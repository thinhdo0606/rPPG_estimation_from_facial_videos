"""
Script to extract the correct pop_mean and pop_std directly from the saved model checkpoint.

The model stores the normalization constants as buffer tensors inside:
  transforms_app.transforms[0].mean  / .std
  transforms_motion.transforms[0].mean / .std
"""
import sys, os, torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

src_path = (
    r"D:\Semester_1_Year_5\Nghien_Cuu_Khoa_Hoc_Thay_Dung"
    r"\rPPG_AI_Prediction_App\web_app\backend\models"
    r"\MTTS_CSTM_PURE_T_10_shift_0.5_combined_losslrstep_5_0.7_best_model_2.pth"
)

print("Loading checkpoint...")
ckpt = torch.load(src_path, map_location="cpu", weights_only=False)
sd = ckpt["model"]

# Print all keys that contain mean/std/running
print("\n--- Normalization-related keys in state dict ---")
for k, v in sd.items():
    if any(kw in k for kw in ("mean", "std", "running")):
        print(f"  {k}: {v.tolist()}")
