"""
Model Service for Heart Rate Estimation
Loads TS-CST Net model and performs inference
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from scipy import signal
from typing import Dict, Tuple
import time
import math
from .config import config


def _spectral_snr_to_confidence(snr_db: float) -> float:
    """
    Map in-band spectral SNR (dB) to a 0–1 display score.
    Same logistic as used in HR estimation; kept in one place so SNR and
    confidence cannot drift if we change the curve later.
    """
    return 1.0 / (1.0 + math.exp(-0.8 * (float(snr_db) + 3.0)))


def _ensure_numpy_pickle_compat():
    """
    Some checkpoints were serialized with NumPy internal paths like `numpy._core`.
    On other environments this path may not exist (only `numpy.core` exists),
    causing `No module named 'numpy._core'` during torch.load.
    """
    try:
        import numpy.core as np_core
        # Alias missing module path used in some pickled checkpoints.
        if "numpy._core" not in sys.modules:
            sys.modules["numpy._core"] = np_core
        if hasattr(np_core, "multiarray") and "numpy._core.multiarray" not in sys.modules:
            sys.modules["numpy._core.multiarray"] = np_core.multiarray
    except Exception:
        # Fallback silently; loader will raise real error if still incompatible.
        pass


# ==================== MODEL ARCHITECTURE ====================
# Import from training codebase to ensure exact match and avoid duplication
import sys
import os

_rppg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "2stream_rppg"))
if _rppg_path not in sys.path:
    sys.path.append(_rppg_path)

from nets.models.MTTS_CSTM_Adjust import MTTS_CSTM


# ==================== MODEL SERVICE ====================

class HeartRateEstimator:
    """Service class for heart rate estimation"""
    
    def __init__(self, model_path: str = None, device: str = None):
        """
        Initialize the estimator
        
        Args:
            model_path: Path to .pth model file
            device: 'cuda' or 'cpu'
        """
        self.device = device or config.DEVICE
        self.model_path = model_path or str(config.MODEL_PATH)
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the trained model"""
        print(f"Loading model from {self.model_path}...")
        _ensure_numpy_pickle_compat()
        
        # Population statistics computed from real webcam face crops (36x36, RGB, [0,1] space).
        # The model divides raw pixel values by 255 internally, so these stats are in [0,1].
        # Appearance stream: mean/std of raw face frames
        # Motion stream: mean/std of frame differences (near-zero mean, small std)
        pop_mean = [
            [0.599, 0.518, 0.506],   # appearance: R,G,B mean
            [-0.000476, -0.000676, -0.000727],  # motion: near-zero
        ]
        pop_std = [
            [0.177, 0.181, 0.180],   # appearance
            [0.0285, 0.0264, 0.0280],  # motion: small (frame differences are tiny)
        ]
        self.model = MTTS_CSTM(
            frame_depth=10,
            pop_mean=pop_mean,
            pop_std=pop_std,
            eca=False,
            shift_factor=0.25,
            skip=True,
            group_on=False
        )
        
        # Apply dynamic quantization ONLY if we are loading the INT8 quantized model
        if "quantized_int8" in str(config.MODEL_PATH):
            self.model = torch.quantization.quantize_dynamic(
                self.model, 
                {torch.nn.Linear}, 
                dtype=torch.qint8
            )
        
        # Load weights — handle both full checkpoint dict and bare state_dict files
        try:
            checkpoint = torch.load(
                self.model_path,
                map_location="cpu",   # always load to CPU first
                weights_only=False,   # needed for older checkpoints with numpy metadata
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {e}") from e
        
        # Extract state dict from various checkpoint formats
        if isinstance(checkpoint, dict):
            if "model" in checkpoint:
                state_dict = checkpoint["model"]          # format used by main_Hao_Summary.py
            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint                   # bare state dict
        else:
            state_dict = checkpoint
        
        # If state dict is FP16 (saved with .half()), upcast to FP32 for accurate CPU inference
        state_dict_fp32 = {k: v.float() if v.dtype == torch.float16 else v
                           for k, v in state_dict.items()}
        
        self.model.load_state_dict(state_dict_fp32)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully on {self.device}")
    
    @torch.no_grad()
    def predict(self, faces_array: np.ndarray) -> Dict:
        """
        Predict heart rate from preprocessed frames
        
        Args:
            faces_array: (T, C, H, W) frames
            
        Returns:
            Dict with heart rate info
        """
        t0 = time.perf_counter()

        T_total = len(faces_array)
        window_length = 10
        stride = 1  # 1-frame stride for smooth overlapping
        
        ppg_accum = np.zeros(T_total - 1)
        counts = np.zeros(T_total - 1)
        t1 = time.perf_counter()
        
        for i in range(0, T_total - window_length, stride):
            chunk = faces_array[i:i + window_length + 1] # shape (11, C, H, W)
            if len(chunk) < window_length + 1:
                break
                
            motion_frames = chunk[1:] - chunk[:-1]
            app_frames = chunk[:-1]
            
            x = np.stack([motion_frames, app_frames], axis=0) # (2, 10, C, H, W)
            x_tensor = torch.from_numpy(x).unsqueeze(0).float().to(self.device) # (1, 2, 10, C, H, W)
            
            out = self.model(x_tensor) # out shape: (1, 10)
            pred = out[0].cpu().numpy()
            
            ppg_accum[i:i + window_length] += pred
            counts[i:i + window_length] += 1
            
        valid_mask = counts > 0
        ppg_signal = ppg_accum[valid_mask] / counts[valid_mask]
        t2 = time.perf_counter()
        
        # Calculate heart rate from PPG signal
        hr_result = self._estimate_hr_from_ppg(ppg_signal, config.FRAME_RATE)
        t3 = time.perf_counter()

        hr_result["benchmark"] = {
            "input_prep_ms": (t1 - t0) * 1000.0,
            "forward_ms": (t2 - t1) * 1000.0,
            "postprocess_ms": (t3 - t2) * 1000.0,
            "inference_ms": (t3 - t1) * 1000.0,
        }
        
        return hr_result
    
    def _estimate_hr_from_ppg(self, ppg_signal: np.ndarray, fps: float) -> Dict:
        """
        Estimate heart rate from PPG signal using FFT
        
        Args:
            ppg_signal: Predicted PPG waveform
            fps: Frame rate
            
        Returns:
            Dict with HR info
        """
        # Detrend signal
        ppg_detrend = signal.detrend(ppg_signal)
        
        # Bandpass filter (0.7-3.5 Hz = 42-210 BPM)
        low_freq = 0.7
        high_freq = 3.5
        nyquist = fps / 2
        
        b, a = signal.butter(
            4,
            [low_freq / nyquist, high_freq / nyquist],
            btype='band'
        )
        ppg_filtered = signal.filtfilt(b, a, ppg_detrend)
        
        # FFT with Hamming window to reduce spectral leakage
        n = len(ppg_filtered)
        window_func = np.hamming(n)
        fft_result = np.fft.rfft(ppg_filtered * window_func)
        frequencies = np.fft.rfftfreq(n, d=1/fps)
        
        # Find valid frequency range
        valid_mask = (frequencies >= low_freq) & (frequencies <= high_freq)
        valid_freqs = frequencies[valid_mask]
        valid_power = np.abs(fft_result[valid_mask])
        
        # Find dominant frequency
        if len(valid_power) > 0:
            peak_idx = np.argmax(valid_power)
            dominant_freq = valid_freqs[peak_idx]
            hr = dominant_freq * 60  # Convert to BPM
        else:
            hr = 0
        
        # Clamp to valid range
        hr = np.clip(hr, config.HR_MIN, config.HR_MAX)
        
        # Calculate robust confidence (Spectral SNR)
        if len(valid_power) > 0:
            # 1. Define frequency window around the peak
            # Since a 4-second clip has poor frequency resolution (~0.23 Hz/bin),
            # we need a wider window to capture the main lobe (e.g., +/- 0.35 Hz)
            window = 0.35
            peak_mask = (valid_freqs >= dominant_freq - window) & (valid_freqs <= dominant_freq + window)
            
            # 2. Calculate signal power (energy around the peak)
            signal_power = np.sum(valid_power[peak_mask] ** 2)
            
            # 3. Calculate noise power (energy of all other frequencies in the band)
            noise_mask = ~peak_mask
            noise_power = np.sum(valid_power[noise_mask] ** 2)
            
            if noise_power > 0:
                snr_ratio = signal_power / noise_power
                snr_db = 10 * np.log10(snr_ratio)  # Convert to dB
            else:
                snr_db = 10.0

            snr_val = float(snr_db)
            # Confidence always derived from the same snr_val (single source of truth)
            confidence = _spectral_snr_to_confidence(snr_val)
        else:
            confidence = 0.0
            snr_val = 0.0
        
        return {
            'heart_rate': float(hr),
            'confidence': float(confidence),
            'snr_db': float(snr_val),
            'unit': 'BPM',
            'ppg_signal': ppg_filtered.tolist()
        }


# Global model instance
_estimator = None

def get_estimator() -> HeartRateEstimator:
    """Get or create model instance (singleton)"""
    global _estimator
    if _estimator is None:
        _estimator = HeartRateEstimator()
    return _estimator

