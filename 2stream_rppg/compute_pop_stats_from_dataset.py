"""
Compute EXACT pop_mean/pop_std from the PURE training dataset HDF5 files.
These are the same values that were computed during training in main_Hao_Summary.py.

Usage:
    python compute_pop_stats_from_dataset.py

Edit PURE_HDF5_DIR below to point to where your .hdf5 files are.
"""
import sys, os
import numpy as np
import torch
import h5py
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
# Point this to the directory containing PURE_train_1.hdf5 ... PURE_train_5.hdf5
# (same as save_root_path in main_Hao_Summary.py)
PURE_HDF5_DIR = os.path.join(os.path.expanduser("~"),
                              "Thinh_Two_Stream_rppg", "Dataset_rppg")
K_FOLD   = 5       # number of folds
WINDOW   = 10      # frame_depth / window_length used in training
# ─────────────────────────────────────────────────────────────────────────────


def compute_from_hdf5():
    app_means, app_stds = [], []
    motion_means, motion_stds = [], []

    for fold in range(1, K_FOLD + 1):
        path = os.path.join(PURE_HDF5_DIR, f"PURE_train_{fold}.hdf5")
        if not os.path.exists(path):
            print(f"  [SKIP] not found: {path}")
            continue

        print(f"  Reading fold {fold}: {path}")
        with h5py.File(path, "r") as f:
            for subject_key in f.keys():
                video = f[subject_key]["video"][:]    # (N, H, W, 3)  uint8 [0,255]

                # Appearance: raw frames / 255  → [0,1]
                app = video.astype(np.float32) / 255.0
                # Motion: frame differences / 255
                motion = np.diff(video.astype(np.float32), axis=0) / 255.0  # (N-1, H, W, 3)

                # Per-channel mean/std over (frames, H, W)
                app_means.append(app.mean(axis=(0, 1, 2)))        # shape (3,)
                app_stds.append(app.std(axis=(0, 1, 2)))
                if len(motion) > 0:
                    motion_means.append(motion.mean(axis=(0, 1, 2)))
                    motion_stds.append(motion.std(axis=(0, 1, 2)))

    if not app_means:
        print("\n[ERROR] No HDF5 files found. Check PURE_HDF5_DIR path.")
        return

    app_mean   = np.array(app_means).mean(axis=0).tolist()
    app_std    = np.array(app_stds).mean(axis=0).tolist()
    motion_mean = np.array(motion_means).mean(axis=0).tolist()
    motion_std  = np.array(motion_stds).mean(axis=0).tolist()

    print("\n" + "="*60)
    print("COPY THESE VALUES INTO model_service.py:")
    print("="*60)
    print(f"pop_mean = [")
    print(f"    {app_mean},   # appearance R,G,B")
    print(f"    {motion_mean},  # motion R,G,B")
    print(f"]")
    print(f"pop_std = [")
    print(f"    {app_std},    # appearance")
    print(f"    {motion_std},   # motion")
    print(f"]")


if __name__ == "__main__":
    print(f"Looking for PURE HDF5 files in: {PURE_HDF5_DIR}")
    compute_from_hdf5()
