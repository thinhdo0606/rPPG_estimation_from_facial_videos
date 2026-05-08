import torch
from nets.models.sub_models.AppearanceModel import AppearanceModel_2D
# For ROI=36*36
from nets.models.sub_models.MotionModel import MotionModel_TS_CSTM    # Input_ROI_size=36*36
from nets.models.sub_models.LinearModel import LinearModel_TS_CSTM    # Input_ROI_size=36*36
# You can swap other ROI variants here (54/72) if needed.
from torchvision import transforms
import numpy as np


class MTTS_CSTM(torch.nn.Module):
    def __init__(self, frame_depth, pop_mean, pop_std, eca, shift_factor, skip, group_on):
        super().__init__()
        self.transforms_app = transforms.Compose([
            transforms.Normalize(mean=pop_mean[0], std=pop_std[0]),
            transforms.RandomHorizontalFlip(0.),
            transforms.RandomVerticalFlip(0.)])

        self.transforms_motion = transforms.Compose([
            transforms.Normalize(mean=pop_mean[1], std=pop_std[1]),
            transforms.RandomHorizontalFlip(0.),
            transforms.RandomVerticalFlip(0.)])

        in_planes = 32                  
        self.kernel_size = (3, 3)
        self.attention_mask1 = None
        self.attention_mask2 = None
        self.appearance_model = AppearanceModel_2D(False, in_planes, self.kernel_size)
        self.motion_model = MotionModel_TS_CSTM(False, in_planes, self.kernel_size, frame_depth, shift_factor, skip, group_on) # "skip" is to adjust In-place TSM or Residual TSM
        self.hr_linear_model = LinearModel_TS_CSTM(False, frame_depth)

    def forward(self, x):
        # X has shape: B, 2, window_length (T), 3, H, W
        motion_input, appearance_input = torch.tensor_split(x, 2, dim=1)
        B, one, T, C, H, W = appearance_input.shape
        # print(f"appearance_input.shape={appearance_input.shape}")

        appearance_input = torch.mean(appearance_input.view(B*one, T, C, H, W), dim=1) / 255
        appearance_input = self.transforms_app(appearance_input)  # -> B, C, H, W
        motion_input = motion_input.view(B*one, T, C, H, W) / 255               # mean frame in each batch
        motion_input = self.transforms_motion(motion_input)  # -> B, T, C, H, W

        # print(f"appearance_input_mean.shape={appearance_input.shape}")
        # print(f"np.shape(motion_input)={np.shape(motion_input)}")
        # print(f"np.shape(appearance_input)={np.shape(appearance_input)}")

        self.attention_mask1, self.attention_mask2 = self.appearance_model(appearance_input)  # -> B, C, H, W
        # print(f"np.shape(self.attention_mask1)={np.shape(self.attention_mask1)}")
        # print(f"np.shape(self.attention_mask2)={np.shape(self.attention_mask2)}")
        motion_out = self.motion_model(motion_input, self.attention_mask1, self.attention_mask2)
        # print(f"np.shape(motion_out)={motion_out.shape}")
        self.motion_output = motion_out[0]
        # self.motion_output = self.motion_output.view(B*T, 64, 9, 9)
        hr_out = self.hr_linear_model(motion_out)
        # print(f"np.shape(hr_out)={np.shape(hr_out)}")

        return hr_out

    def get_attention_mask(self):
        return self.attention_mask1, self.attention_mask2
