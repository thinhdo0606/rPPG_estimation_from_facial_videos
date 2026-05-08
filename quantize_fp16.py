import torch
import os
import sys

def _ensure_numpy_pickle_compat():
    try:
        import numpy.core as np_core
        if "numpy._core" not in sys.modules:
            sys.modules["numpy._core"] = np_core
        if hasattr(np_core, "multiarray") and "numpy._core.multiarray" not in sys.modules:
            sys.modules["numpy._core.multiarray"] = np_core.multiarray
    except Exception:
        pass

def quantize_to_fp16(input_path, output_path):
    print(f"Loading model from {input_path}...")
    _ensure_numpy_pickle_compat()
    
    # Load the checkpoint
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)
    
    # Extract state_dict
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    print(f"Original state dict extracted.")
    
    # Convert all floating-point tensors to FP16
    fp16_state_dict = {}
    for k, v in state_dict.items():
        if v.dtype == torch.float32 or v.dtype == torch.float64:
            fp16_state_dict[k] = v.half()
        else:
            fp16_state_dict[k] = v

    print(f"Converted to FP16.")
    
    # Save the new state_dict
    torch.save(fp16_state_dict, output_path)
    
    # Compare sizes
    old_size = os.path.getsize(input_path) / (1024 * 1024)
    new_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Saved FP16 model to {output_path}")
    print(f"Original size: {old_size:.2f} MB")
    print(f"New size:      {new_size:.2f} MB")
    print(f"Reduction:     {(1 - new_size/old_size)*100:.2f}%")

if __name__ == "__main__":
    base_dir = r"d:\Semester_1_Year_5\Nghien_Cuu_Khoa_Hoc_Thay_Dung\rPPG_AI_Prediction_App\web_app\backend\models"
    input_file = "MTTS_CSTM_UBFC_T_10_shift_0.25_combined_loss_best_model_5.pth"
    output_file = "MTTS_CSTM_UBFC_T_10_shift_0.25_combined_loss_best_model_5_fp16.pth"
    
    input_path = os.path.join(base_dir, input_file)
    output_path = os.path.join(base_dir, output_file)
    
    if os.path.exists(input_path):
        quantize_to_fp16(input_path, output_path)
    else:
        print(f"Error: Could not find input file at {input_path}")
