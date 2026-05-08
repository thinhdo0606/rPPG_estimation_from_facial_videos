"""
Compute realistic pop_mean / pop_std for MTTS_CSTM inference.

The model internally divides pixel values by 255 (see MTTS_CSTM_Adjust.py line 38/40),
so these statistics must be in [0,1] space.

For the PURE dataset (RGB face crops, 36x36):
- Appearance (raw frame / 255): ~skin+background mix
- Motion (frame_diff / 255): tiny differences, mean ~0, std very small

We synthesise approximate values here and also show how to compute them
from actual video frames via cv2 if needed.
"""
import sys, os
import numpy as np
import cv2

# ----------------------------------------------------------------
# Option A: Compute from a sample video/image
# (Run this once to get accurate values for deployment)
# ----------------------------------------------------------------

def compute_stats_from_sample_video(video_path: str, num_frames: int = 500):
    """
    Read up to `num_frames` frames from a webcam or video file,
    crop center 36x36, compute mean/std for app and motion streams
    exactly as the training dataset does.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while cap.isOpened() and count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        # Crop center 36x36
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        face = frame[cy-18:cy+18, cx-18:cx+18]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32)
        frames.append(face)
        count += 1
    cap.release()

    if len(frames) < 2:
        return None, None, None, None

    frames = np.array(frames)  # (T, 36, 36, 3)  [0,255]
    app = frames / 255.0       # [0,1]
    motion = np.diff(frames, axis=0) / 255.0  # frame differences / 255

    app_mean = app.mean(axis=(0, 1, 2)).tolist()      # [R_mean, G_mean, B_mean]
    app_std  = app.std(axis=(0, 1, 2)).tolist()
    motion_mean = motion.mean(axis=(0, 1, 2)).tolist()
    motion_std  = motion.std(axis=(0, 1, 2)).tolist()

    return app_mean, app_std, motion_mean, motion_std


if __name__ == "__main__":
    # Try to read from default webcam (0) for 5 seconds at ~30fps = 150 frames
    print("Reading from webcam (30s)...")
    cap = cv2.VideoCapture(0)
    frames = []
    count = 0
    while cap.isOpened() and count < 300:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        # Take center crop
        size = min(h, w) // 2
        cx, cy = w // 2, h // 2
        face = frame[cy-size//2:cy+size//2, cx-size//2:cx+size//2]
        if face.size == 0:
            continue
        face = cv2.resize(face, (36, 36))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32)
        frames.append(face)
        count += 1
    cap.release()

    if len(frames) < 2:
        print("Could not read webcam frames.")
    else:
        frames = np.array(frames)  # (T, 36, 36, 3)
        app = frames / 255.0
        motion = np.diff(frames, axis=0) / 255.0

        app_mean   = app.mean(axis=(0, 1, 2)).tolist()
        app_std    = app.std(axis=(0, 1, 2)).tolist()
        motion_mean = motion.mean(axis=(0, 1, 2)).tolist()
        motion_std  = motion.std(axis=(0, 1, 2)).tolist()

        print(f"\n=== Results (use these in model_service.py) ===")
        print(f"pop_mean = [{app_mean}, {motion_mean}]")
        print(f"pop_std  = [{app_std},  {motion_std}]")
