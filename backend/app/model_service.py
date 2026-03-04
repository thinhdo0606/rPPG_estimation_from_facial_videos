"""
Model Service for Heart Rate Estimation
Loads TS-CST Net model and performs inference
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import signal
from typing import Dict, Tuple
from .config import config


# ==================== MODEL ARCHITECTURE ====================
# Copy from training code - must match exactly!

class TemporalShift(nn.Module):
    """TSM - Shift channels along temporal dimension."""
    
    def __init__(self, n_segment=8, n_div=8):
        super().__init__()
        self.n_segment = n_segment
        self.n_div = n_div
    
    def forward(self, x):
        if x.dim() == 5:
            B, T, C, H, W = x.shape
        else:
            BT, C, H, W = x.shape
            T = self.n_segment
            B = BT // T
            x = x.view(B, T, C, H, W)
        
        fold = C // self.n_div
        out = torch.zeros_like(x)
        out[:, 1:, :fold] = x[:, :-1, :fold]
        out[:, :-1, fold:2*fold] = x[:, 1:, fold:2*fold]
        out[:, :, 2*fold:] = x[:, :, 2*fold:]
        
        return out


class CNNBlock(nn.Module):
    """Basic CNN Block: Conv -> BN -> ReLU"""
    
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class TMB(nn.Module):
    """Temporal Module Block with temporal attention."""
    
    def __init__(self, channels, n_segment=128):
        super().__init__()
        self.n_segment = n_segment
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.temporal_conv = nn.Conv1d(channels, channels, 3, padding=1, groups=channels)
    
    def forward(self, x):
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
            reshape = True
        else:
            BT, C, H, W = x.shape
            T = self.n_segment
            B = BT // T
            reshape = False
        
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        
        _, C, H_out, W_out = out.shape
        out_t = out.view(B, T, C, H_out, W_out).mean(dim=[3, 4])
        out_t = out_t.permute(0, 2, 1)
        out_t = torch.sigmoid(self.temporal_conv(out_t))
        out_t = out_t.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
        
        out = out.view(B, T, C, H_out, W_out) * out_t
        out = out.view(B * T, C, H_out, W_out)
        
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        
        if reshape:
            out = out.view(B, T, C, H_out, W_out)
        return out


class TSMBlock(nn.Module):
    """TSM + Conv Block"""
    
    def __init__(self, in_ch, out_ch, stride=1, n_segment=128):
        super().__init__()
        self.tsm = TemporalShift(n_segment=n_segment)
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = self.tsm(x)
            x = x.view(B * T, C, H, W)
            x = self.relu(self.bn(self.conv(x)))
            _, C_out, H_out, W_out = x.shape
            return x.view(B, T, C_out, H_out, W_out)
        else:
            x = self.tsm(x)
            return self.relu(self.bn(self.conv(x)))


class SpatialAttention(nn.Module):
    """Spatial attention to generate attention mask."""
    
    def __init__(self, in_ch, reduction=4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, in_ch // reduction, 1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_ch // reduction, 1, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        return self.sigmoid(self.conv2(self.relu(self.conv1(x))))


class AttentionMask(nn.Module):
    """Cross-stream attention from spatial to temporal."""
    
    def __init__(self, in_ch):
        super().__init__()
        self.spatial_attn = SpatialAttention(in_ch)
        self.conv = nn.Conv2d(in_ch, in_ch, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, spatial_feat, temporal_feat):
        spatial_mask = self.spatial_attn(spatial_feat)
        attn = self.sigmoid(self.conv(spatial_feat))
        
        if temporal_feat.dim() == 5:
            B, T, C, H, W = temporal_feat.shape
            spatial_mask = spatial_mask.unsqueeze(1).expand(-1, T, -1, -1, -1)
            attn = attn.unsqueeze(1).expand(-1, T, -1, -1, -1)
            return temporal_feat * spatial_mask * attn
        else:
            BT = temporal_feat.shape[0]
            B = spatial_feat.shape[0]
            T = BT // B
            spatial_mask = spatial_mask.repeat(T, 1, 1, 1)
            attn = attn.repeat(T, 1, 1, 1)
            return temporal_feat * spatial_mask * attn


class AvgPoolDropout(nn.Module):
    """Average Pooling + Dropout"""
    
    def __init__(self, pool_size=2, dropout=0.25):
        super().__init__()
        self.avgpool = nn.AvgPool2d(pool_size, pool_size)
        self.dropout = nn.Dropout2d(dropout)
    
    def forward(self, x):
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
            x = self.dropout(self.avgpool(x))
            _, C, H_out, W_out = x.shape
            return x.view(B, T, C, H_out, W_out)
        return self.dropout(self.avgpool(x))


class SpatialStream(nn.Module):
    """Spatial Attention Stream"""
    
    def __init__(self, in_ch=3, dropout=0.25):
        super().__init__()
        self.stage1 = nn.Sequential(
            CNNBlock(in_ch, 32),
            CNNBlock(32, 32)
        )
        self.pool1 = AvgPoolDropout(2, dropout)
        self.attn1 = AttentionMask(32)
        
        self.stage2 = nn.Sequential(
            CNNBlock(32, 64, stride=2),
            CNNBlock(64, 64)
        )
        self.pool2 = AvgPoolDropout(2, dropout)
        self.attn2 = AttentionMask(64)
    
    def forward(self, mean_frame):
        if mean_frame.dim() == 5:
            mean_frame = mean_frame.squeeze(1)
        
        f1 = self.pool1(self.stage1(mean_frame))
        f2 = self.pool2(self.stage2(f1))
        
        return f1, f2, self.attn1, self.attn2


class TemporalStream(nn.Module):
    """Temporal Stream"""
    
    def __init__(self, in_ch=3, n_segment=128, dropout=0.25):
        super().__init__()
        self.n_segment = n_segment
        
        self.conv1 = CNNBlock(in_ch, 32)
        self.tmb1 = TMB(32, n_segment)
        self.pool1 = AvgPoolDropout(2, dropout)
        
        self.tsm2 = TSMBlock(32, 64, stride=2, n_segment=n_segment)
        self.tmb2 = TMB(64, n_segment)
        self.pool2 = AvgPoolDropout(2, dropout)
    
    def forward(self, diff_frames, sf1, sf2, attn1, attn2):
        B, T, C, H, W = diff_frames.shape
        
        x = diff_frames.view(B * T, C, H, W)
        x = self.conv1(x)
        _, C1, H1, W1 = x.shape
        x = x.view(B, T, C1, H1, W1)
        x = self.tmb1(x)
        x = self.pool1(x)
        x = attn1(sf1, x)
        
        x = self.tsm2(x)
        x = self.tmb2(x)
        x = self.pool2(x)
        x = attn2(sf2, x)
        
        return x


class TSCSTNet(nn.Module):
    """TS-CST Net: Two-Stream Convolutional Spatial-Temporal Network"""
    
    def __init__(self, clip_length=128, hidden_dim=128, dropout=0.5):
        super().__init__()
        
        self.spatial_stream = SpatialStream(dropout=0.25)
        self.temporal_stream = TemporalStream(n_segment=clip_length, dropout=0.25)
        
        self.fc1 = nn.Linear(64, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, diff_frames, mean_frame):
        B, T, _, _, _ = diff_frames.shape
        
        sf1, sf2, attn1, attn2 = self.spatial_stream(mean_frame)
        x = self.temporal_stream(diff_frames, sf1, sf2, attn1, attn2)
        
        x = x.view(B * T, 64, -1).mean(dim=2)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x.view(B, T)


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
        
        # Create model
        self.model = TSCSTNet(
            clip_length=config.CLIP_LENGTH,
            hidden_dim=128,
            dropout=0.5
        )
        
        # Load weights
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully on {self.device}")
    
    @torch.no_grad()
    def predict(self, diff_frames: np.ndarray, mean_frame: np.ndarray) -> Dict:
        """
        Predict heart rate from preprocessed frames
        
        Args:
            diff_frames: (T-1, C, H, W) difference frames
            mean_frame: (1, C, H, W) mean frame
            
        Returns:
            Dict with heart rate info
        """
        # Add batch dimension
        diff_tensor = torch.from_numpy(diff_frames).unsqueeze(0).float().to(self.device)
        mean_tensor = torch.from_numpy(mean_frame).unsqueeze(0).float().to(self.device)
        
        # Forward pass
        ppg_signal = self.model(diff_tensor, mean_tensor)
        ppg_signal = ppg_signal.cpu().numpy()[0]  # (T,)
        
        # Calculate heart rate from PPG signal
        hr_result = self._estimate_hr_from_ppg(ppg_signal, config.FRAME_RATE)
        
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
        
        # FFT
        n = len(ppg_filtered)
        fft_result = np.fft.rfft(ppg_filtered)
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
        
        # Calculate confidence (SNR)
        if len(valid_power) > 0:
            peak_power = valid_power[peak_idx]
            noise_power = np.mean(valid_power)
            snr = peak_power / (noise_power + 1e-8)
            confidence = min(snr / 10, 1.0)  # Normalize to 0-1
        else:
            confidence = 0
        
        return {
            'heart_rate': float(hr),
            'confidence': float(confidence),
            'unit': 'BPM',
            'ppg_signal': ppg_signal.tolist()
        }


# Global model instance
_estimator = None

def get_estimator() -> HeartRateEstimator:
    """Get or create model instance (singleton)"""
    global _estimator
    if _estimator is None:
        _estimator = HeartRateEstimator()
    return _estimator

