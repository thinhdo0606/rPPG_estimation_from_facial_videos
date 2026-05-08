import copy
import datetime
import time

import h5py
import torch
import optim
from loss2 import loss_fn
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from dataset.dataset_loader2 import dataset_loader
# from dataset.dataset_loader2_val import dataset_loader as dataset_loader_v    # for revising overlapping on Validation
from log import log_info_time
from models2 import is_model_support, get_model
from torch.optim import lr_scheduler
from utils.funcs2 import plot_graph, plot_loss_graph, BPF_dict, normalize
from utils.eval_metrics2 import *
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless Linux server (no display)
import matplotlib.pyplot as plt
from torchvision import utils
import torch.nn as nn
import os
import random
import math


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def visTensor(tensor, ch=0, allkernels=False, nrow=8, padding=1):
    n, c, w, h = tensor.shape

    if allkernels:
        tensor = tensor.view(n * c, -1, w, h)
    elif c != 3:
        tensor = tensor[:, ch, :, :].unsqueeze(dim=1)

    rows = np.min((tensor.shape[0] // nrow + 1, 64))
    grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    plt.figure(figsize=(nrow, rows))
    plt.imshow(grid.cpu().numpy().transpose((1, 2, 0)))

# # torch.backends.cudnn.enabled = True
def setup_seed(seed):
    print('fix random seed')
    os.environ['PYTHONHASHSEED'] = str(seed)    # set random seed for python environment
    torch.manual_seed(seed)                     # set random seed for current CPU
    torch.cuda.manual_seed(seed)                # set random seed for current GPU
    torch.cuda.manual_seed_all(seed)            # set random seed for all GPU
    np.random.seed(seed)                        # set random seed for numpy
    random.seed(seed)                           # set random seed for python
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_evaluation_plots(groundtruth, prediction, fs_used, NF, path, model_name, dataset_name, windows=[30, 20, 10]):
    """Vẽ và lưu 3 đồ thị: BVP Waveform, Bland-Altman, Scatter Correlation."""
    from scipy.signal import welch
    os.makedirs(path, exist_ok=True)

    # 1. BVP Waveform Comparison (10 giây đầu của người đầu tiên)
    try:
        if len(groundtruth.keys()) > 0:
            gt_sig = groundtruth[0]
            pred_sig = prediction[0]
            fs_val = fs_used[0] if isinstance(fs_used, list) else fs_used
            plot_len = min(int(10 * fs_val), len(gt_sig))
            t = np.arange(plot_len) / fs_val

            plt.figure(figsize=(10, 4))
            plt.plot(t, gt_sig[:plot_len], label='Ground Truth BVP', color='black', linewidth=1.2, alpha=0.85)
            plt.plot(t, pred_sig[:plot_len], label='Predicted BVP', color='red', linewidth=1.2, alpha=0.85)
            plt.title(f'BVP Waveform Comparison (10s) - Fold {NF}', fontsize=13)
            plt.xlabel('Time (s)', fontsize=11)
            plt.ylabel('Normalized Amplitude', fontsize=11)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.4)
            plt.tight_layout()
            plt.savefig(os.path.join(path, f"Waveform_{model_name}_{dataset_name}_Fold_{NF}.png"), dpi=150)
            plt.close()
            print(f"[Plot] Saved Waveform Fold {NF}")
    except Exception as e:
        print(f"[Plot] Waveform failed: {e}")

    # Helper: tính danh sách HR từ tín hiệu PPG theo cửa sổ win giây
    def get_hr_lists(win):
        hr_gt_list, hr_pred_list = [], []
        for i in range(len(groundtruth.keys())):
            target_sig = groundtruth[i]
            predict_sig = prediction[i]
            fs_i = float(fs_used[i] if isinstance(fs_used, list) else fs_used)
            samples = int(fs_i * win)
            step_s = int(fs_i * 1)  # bước 1 giây
            signal_length = len(target_sig)
            if samples > signal_length:
                continue
            seglength = samples if samples < 256 else 256
            overlap = int(0.8 * samples) if samples < 256 else 200
            nfft = int(np.ceil((60 * 2 * (fs_i / 2)) / 0.5))
            for j in range(0, signal_length - samples + 1, step_s):
                pred_seg = predict_sig[j:j + samples]
                gt_seg   = target_sig[j:j + samples]
                pf, pp = welch(pred_seg, fs=fs_i, nperseg=seglength, noverlap=overlap, nfft=nfft)
                gf, gp = welch(gt_seg,   fs=fs_i, nperseg=seglength, noverlap=overlap, nfft=nfft)
                hr_pred_list.append(pf[np.argmax(pp)] * 60)
                hr_gt_list.append(gf[np.argmax(gp)] * 60)
        return np.array(hr_gt_list), np.array(hr_pred_list)

    # 2. Bland-Altman và 3. Scatter — mỗi cái 1x3 subplots cho 30s/20s/10s
    try:
        fig_ba, axes_ba = plt.subplots(1, 3, figsize=(18, 5))
        fig_sc, axes_sc = plt.subplots(1, 3, figsize=(18, 5))

        for idx, win in enumerate(windows):
            gt_hr, pred_hr = get_hr_lists(win)
            if len(gt_hr) == 0:
                continue

            # Bland-Altman
            mean_hr = (gt_hr + pred_hr) / 2.0
            diff_hr = pred_hr - gt_hr
            md = np.mean(diff_hr)
            sd = np.std(diff_hr)
            axes_ba[idx].scatter(mean_hr, diff_hr, alpha=0.4, color='steelblue', s=12)
            axes_ba[idx].axhline(md,           color='navy',  linewidth=1.5, label=f'Mean: {md:.2f}')
            axes_ba[idx].axhline(md + 1.96*sd, color='red', linestyle='--', linewidth=1.2, label=f'+1.96SD: {md+1.96*sd:.2f}')
            axes_ba[idx].axhline(md - 1.96*sd, color='red', linestyle='--', linewidth=1.2, label=f'-1.96SD: {md-1.96*sd:.2f}')
            axes_ba[idx].set_title(f'Bland-Altman ({win}s Window)', fontsize=11)
            axes_ba[idx].set_xlabel('Average HR (BPM)', fontsize=10)
            axes_ba[idx].set_ylabel('Difference Pred−GT (BPM)', fontsize=10)
            axes_ba[idx].legend(fontsize=8)
            axes_ba[idx].grid(True, alpha=0.3)

            # Scatter Plot
            min_v = min(float(np.min(gt_hr)), float(np.min(pred_hr))) - 5
            max_v = max(float(np.max(gt_hr)), float(np.max(pred_hr))) + 5
            axes_sc[idx].scatter(gt_hr, pred_hr, alpha=0.4, color='seagreen', s=12)
            axes_sc[idx].plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=1.5, label='y = x (perfect)')
            corr = np.corrcoef(gt_hr, pred_hr)[0, 1]
            axes_sc[idx].set_title(f'Scatter ({win}s) | r={corr:.3f}', fontsize=11)
            axes_sc[idx].set_xlabel('Ground Truth HR (BPM)', fontsize=10)
            axes_sc[idx].set_ylabel('Predicted HR (BPM)', fontsize=10)
            axes_sc[idx].legend(fontsize=8)
            axes_sc[idx].grid(True, alpha=0.3)

        fig_ba.suptitle(f'Bland-Altman Plots — {model_name} {dataset_name} Fold {NF}', fontsize=14)
        fig_ba.tight_layout()
        fig_ba.savefig(os.path.join(path, f"BlandAltman_{model_name}_{dataset_name}_Fold_{NF}.png"), dpi=150)
        plt.close(fig_ba)
        print(f"[Plot] Saved BlandAltman Fold {NF}")

        fig_sc.suptitle(f'Scatter Correlation Plots — {model_name} {dataset_name} Fold {NF}', fontsize=14)
        fig_sc.tight_layout()
        fig_sc.savefig(os.path.join(path, f"Scatter_{model_name}_{dataset_name}_Fold_{NF}.png"), dpi=150)
        plt.close(fig_sc)
        print(f"[Plot] Saved Scatter Fold {NF}")

    except Exception as e:
        print(f"[Plot] Bland-Altman/Scatter failed: {e}")


def compute_hr_metrics_from_validation(inference_array, target_array, valid_dataset, window_length):
    """
    Build per-video signals and compute HR metrics for windows 30s / 20s / 10s.
    Return None if signals cannot be constructed.
    """
    result = {}
    groundtruth = {}
    fs_used = []
    start_idx = 0
    out_vid_idx = 0

    n_frames_per_video = valid_dataset.n_frames_per_video
    vid_fs = valid_dataset.video_fs

    total_len = min(len(inference_array), len(target_array))

    for i, value in enumerate(n_frames_per_video):
        # __getitem__ uses x[idx:idx+wl+1], so effective predictable frames are based on (value-1).
        value_eff = ((int(value) - 1) // window_length) * window_length
        if value_eff <= 0:
            continue

        end_idx = min(start_idx + value_eff, total_len)
        if end_idx <= start_idx:
            break

        pred_seg = np.asarray(inference_array[start_idx:end_idx]).squeeze()
        gt_seg = np.asarray(target_array[start_idx:end_idx]).squeeze()
        if len(pred_seg) < 2 or len(gt_seg) < 2:
            start_idx = end_idx
            continue

        result[out_vid_idx] = normalize(pred_seg)
        groundtruth[out_vid_idx] = gt_seg
        fs_used.append(vid_fs[i] if isinstance(vid_fs, list) else vid_fs)
        out_vid_idx += 1
        start_idx = end_idx

    if len(result) == 0:
        return None

    result = BPF_dict(result, fs_used)
    groundtruth = BPF_dict(groundtruth, fs_used)

    res30 = HR_Metric(groundtruth, result, fs_used, 30, 1)
    mae30, rmse30 = res30[0], res30[1]
    pearson30 = Pearson_Corr(groundtruth, result)
    res20 = HR_Metric(groundtruth, result, fs_used, 20, 1)
    mae20, rmse20 = res20[0], res20[1]
    res10 = HR_Metric(groundtruth, result, fs_used, 10, 1)
    mae10, rmse10 = res10[0], res10[1]

    return {
        "mae30":     float(mae30),
        "rmse30":    float(rmse30),
        "pearson30": float(pearson30),
        "mae20":     float(mae20),
        "rmse20":    float(rmse20),
        "mae10":     float(mae10),
        "rmse10":    float(rmse10),
        "_plot_data": (groundtruth, result, fs_used),  # dùng để vẽ 3 đồ thị
    }
    
def main():

    # setup_seed(0)
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)

    ''' MSE is Mean Square Error   '''

    '''Setting up'''
    __TIME__ = True
    train = 0  # 0: train from the scratch, 1: continue to train from a pth file, 2: test a model
    model_name = "MTTS_CSTM"

    # Where your PURE_train_*.hdf5 / PURE_test_*.hdf5 are located
    # dataset_loader2 will build file paths as: save_root_path + "PURE_train_{fold}.hdf5"
    # save_root_path = os.path.join(os.path.expanduser("~"), "Thinh_Two_Stream_rppg", "Dataset_rppg") + "/"
    save_root_path = "/media/user/DATA/pytorch_rppgs/Rppg_database_for_training_testing/Baseline_FiveFold(Testing_Unseen_Rand)/PURE/"

    # Save checkpoints here (created by you before running, if needed)
    # checkpoint_path = os.path.join(
    #     os.path.expanduser("~"),
    #     "Thinh_Two_Stream_rppg",
    #     "Dataset_rppg",
    #     "2stream_rppg_checkpoints",
    #     "retrain_PURE_36",
    # ) + "/"
    checkpoint_path = "Thinh_Checkspoint_PURE/"



    # dataset_name = [["PURE"], ["MMSE"], ["UBFC"], ["MANHOB_HCI"]]
    print("Checkpoint Path: ", checkpoint_path)
    dataset_name = [["PURE"]]
    # window_length = [5, 7, 10, 13, 15, 20]              # T (window_length)
    # shift_factor =  [0.2, 0.15, 0.1, 0.08, 0.07, 0.06]             # n/T (the ratio of shift) default: 0.25
    window_length_l = [10]              # T (window_length)
    shift_factor_l =  [0.25]             # n/T (the ratio of shift) default: 0.25
    loss_metric = "combined_loss"      # "combined_loss" "snr" "mse"
    optimi = "ada_delta"
    ROI = 36                            # Adjust faceROI to 36 | 54 | 72
    batch_size = 32                     # ImgROI=36: 32   ImgROI=72: 16

    tot_epochs = 25
    model_list = ["MTTS", "TSDAN", "MTTS_CSTM", "SlowFast_FD", "SlowFast_AM"]
    # fs = 30                                                                                           

    # Decay parameter
    learning_rate = 0.05
    Gamma = 0.8
    Step = 4
    # Exponential
    # Gamma = 0.8

    skip_connection = os.getenv("ABLATION_SKIP_CONNECTION", "1") == "1"  # True: Residual  False: In-place
    new_group_tsm = False
    k_fold = int(os.getenv("ABLATION_K_FOLD", "5"))
    ablation_tag = os.getenv("ABLATION_TAG", "").strip()
    if ablation_tag:
        checkpoint_path = os.path.join(checkpoint_path, ablation_tag) + "/"
    os.makedirs(checkpoint_path, exist_ok=True)
    print(f"Ablation config | skip_connection={skip_connection} | disable_tam={os.getenv('RPPG_DISABLE_TAM', '0')} | k_fold={k_fold} | tag={ablation_tag or 'default'}")
    cuda_device=0
    print(f"GPU device={cuda_device}")  
    setup_seed(20)

    if __TIME__:

        start_time = time.time()

    if train == 0 or train == 1:
        # for i in range(5):
        # checkpoint_name = "TS_CST_MMSE_T_10_shift_0.5_best_model" + str(i) + ".pth"
        # checkpoint_name = "TS_CST_MMSE_T_10_shift_0.5_best_model.pth"    
        for (window_length, shift_factor) in zip(window_length_l, shift_factor_l):
            for dataset in dataset_name:
                fold_best_metrics = []
                # print(f"type(dataset)={type(dataset)}, dataset={dataset}")
                for NF in range(1, k_fold + 1):
                    # Anh
                    # print("NF={}".format(NF))
                    # setup_seed(20)
                    train_dataset, valid_dataset = dataset_loader(train, save_root_path, model_name, dataset, window_length,
                                                                    fold=NF, SW=window_length, ImgROI=ROI)
                    # Hao
                    # train_dataset, valid_dataset = dataset_loader_v(train, save_root_path, model_name, dataset, window_length,                  
                    #                                fold=NF)
                    # Random Seed
                    # g = torch.Generator()
                    # g.manual_seed(0)
                    # Original
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                num_workers=4, pin_memory=True, drop_last=False)

                    validation_loader = DataLoader(valid_dataset, batch_size=batch_size, sampler=SequentialSampler(valid_dataset),
                                                    num_workers=4, pin_memory=True, drop_last=False)
                    # Random Seed
                    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                    #                             num_workers=0, worker_init_fn=random.seed(20), pin_memory=True, drop_last=False)

                    # validation_loader = DataLoader(valid_dataset, batch_size=batch_size, sampler=SequentialSampler(valid_dataset),
                    #                                 num_workers=0, worker_init_fn=random.seed(20), pin_memory=True, drop_last=False)

                    app_mean = []
                    app_std = []
                    motion_mean = []
                    motion_std = []

                    with tqdm(total=len(train_dataset) + len(valid_dataset), position=0, leave=True,
                                desc='Calculating population statistics') as pbar:
                        for data in train_loader:
                            # TSDAN is used to be an evaluation model MTTS_CSTM = TS_CST
                            if model_name in ['TSDAN', 'MTTS', 'MTTS_CSTM']:
                                data = data[0]  # -> (Batch, 2, T, H, W, 3)
                                motion_data, app_data = torch.tensor_split(data, 2, dim=1)
                                B, one, T, C, H, W = motion_data.shape

                                motion_data = motion_data.view(B * one, T, C, H, W)
                                app_data = app_data.view(B * one, T, C, H, W)
                                motion_data = motion_data.reshape(B * T, C, H, W)
                                app_data = app_data.reshape(B * T, C, H, W)

                                batch_motion_mean = torch.mean(motion_data, dim=(0, 2, 3)).tolist()
                                batch_motion_std = torch.std(motion_data, dim=(0, 2, 3)).tolist()
                                batch_app_mean = torch.mean(app_data, dim=(0, 2, 3)).tolist()
                                batch_app_std = torch.std(app_data, dim=(0, 2, 3)).tolist()

                                app_mean.append(batch_app_mean)
                                app_std.append(batch_app_std)
                                motion_mean.append(batch_motion_mean)
                                motion_std.append(batch_motion_std)

                            elif model_name == 'SlowFast_FD':
                                motion_data = data[0][0]
                                app_data = data[0][2]
                                B, T, C, H, W = motion_data.shape
                                motion_data = motion_data.view(B * T, C, H, W)
                                app_data = app_data.view(B * T, C, H, W)

                                batch_motion_mean = torch.mean(motion_data, dim=(0, 2, 3)).tolist()
                                batch_motion_std = torch.std(motion_data, dim=(0, 2, 3)).tolist()
                                batch_app_mean = torch.mean(app_data, dim=(0, 2, 3)).tolist()
                                batch_app_std = torch.std(app_data, dim=(0, 2, 3)).tolist()

                                app_mean.append(batch_app_mean)
                                app_std.append(batch_app_std)
                                motion_mean.append(batch_motion_mean)
                                motion_std.append(batch_motion_std)

                            elif model_name == 'SlowFast_AM':
                                motion_data = data[0][0]
                                app_data = data[0][2]
                                B, T, C, H, W = motion_data.shape
                                motion_data = motion_data.view(B * T, C, H, W)
                                app_data = app_data.view(B * T, C, H, W)

                                batch_motion_mean = torch.mean(motion_data, dim=(0, 2, 3)).tolist()
                                batch_motion_std = torch.std(motion_data, dim=(0, 2, 3)).tolist()
                                batch_app_mean = torch.mean(app_data, dim=(0, 2, 3)).tolist()
                                batch_app_std = torch.std(app_data, dim=(0, 2, 3)).tolist()

                                app_mean.append(batch_app_mean)
                                app_std.append(batch_app_std)
                                motion_mean.append(batch_motion_mean)
                                motion_std.append(batch_motion_std)

                            elif model_name in ['STM_Phys', 'New']:
                                data = data[0].numpy()  # B, T+1, H, W, C
                                if window_length == 10:
                                    data = data[:, :-1, :, :, :]
                                else:
                                    data = data[:, :-2, :, :, :]
                                B, T, C, H, W = data.shape
                                data = np.reshape(data, (B * T, C, H, W))
                                batch_app_mean = np.mean(data, axis=(0, 2, 3))
                                batch_app_std = np.std(data, axis=(0, 2, 3))
                                app_mean.append(batch_app_mean)
                                app_std.append(batch_app_std)

                            pbar.update(B)

                        for i, data in enumerate(validation_loader):
                            if model_name in ['TSDAN', 'MTTS', 'MTTS_CSTM']:
                                data = data[0]  # shape (Batch, T+1, H, W, 6)
                                motion_data, app_data = torch.tensor_split(data, 2, dim=1)
                                B, one, T, C, H, W = motion_data.shape

                                motion_data = motion_data.view(B * one, T, C, H, W)
                                app_data = app_data.view(B * one, T, C, H, W)
                                motion_data = motion_data.reshape(B * T, C, H, W)
                                app_data = app_data.reshape(B * T, C, H, W)

                                batch_motion_mean = torch.mean(motion_data, dim=(0, 2, 3)).tolist()
                                batch_motion_std = torch.std(motion_data, dim=(0, 2, 3)).tolist()
                                batch_app_mean = torch.mean(app_data, dim=(0, 2, 3)).tolist()
                                batch_app_std = torch.std(app_data, dim=(0, 2, 3)).tolist()

                                app_mean.append(batch_app_mean)
                                app_std.append(batch_app_std)
                                motion_mean.append(batch_motion_mean)
                                motion_std.append(batch_motion_std)

                            elif model_name == 'SlowFast_FD':
                                motion_data = data[0][0]
                                app_data = data[0][2]
                                B, T, C, H, W = motion_data.shape
                                motion_data = motion_data.view(B * T, C, H, W)
                                app_data = app_data.view(B * T, C, H, W)

                                batch_motion_mean = torch.mean(motion_data, dim=(0, 2, 3)).tolist()
                                batch_motion_std = torch.std(motion_data, dim=(0, 2, 3)).tolist()
                                batch_app_mean = torch.mean(app_data, dim=(0, 2, 3)).tolist()
                                batch_app_std = torch.std(app_data, dim=(0, 2, 3)).tolist()

                                app_mean.append(batch_app_mean)
                                app_std.append(batch_app_std)
                                motion_mean.append(batch_motion_mean)
                                motion_std.append(batch_motion_std)

                            elif model_name == 'SlowFast_AM':
                                motion_data = data[0][0]
                                app_data = data[0][2]
                                B, T, C, H, W = motion_data.shape
                                motion_data = motion_data.view(B * T, C, H, W)
                                app_data = app_data.view(B * T, C, H, W)

                                batch_motion_mean = torch.mean(motion_data, dim=(0, 2, 3)).tolist()
                                batch_motion_std = torch.std(motion_data, dim=(0, 2, 3)).tolist()
                                batch_app_mean = torch.mean(app_data, dim=(0, 2, 3)).tolist()
                                batch_app_std = torch.std(app_data, dim=(0, 2, 3)).tolist()

                                app_mean.append(batch_app_mean)
                                app_std.append(batch_app_std)
                                motion_mean.append(batch_motion_mean)
                                motion_std.append(batch_motion_std)

                            elif model_name in ['STM_Phys', 'New']:
                                data = data[0].numpy()  # B, T+1, H, W, C
                                if window_length == 10:
                                    data = data[:, :-1, :, :, :]
                                else:
                                    data = data[:, :-2, :, :, :]
                                B, T, C, H, W = data.shape
                                data = np.reshape(data, (B * T, C, H, W))

                                batch_app_mean = np.mean(data, axis=(0, 2, 3))
                                batch_app_std = np.std(data, axis=(0, 2, 3))

                                app_mean.append(batch_app_mean)
                                app_std.append(batch_app_std)

                            pbar.update(B)
                        pbar.close()

                    if model_name in ['TSDAN', 'MTTS', 'MTTS_CSTM', 'SlowFast_FD', "SlowFast_AM"]:
                        # shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
                        app_mean = np.array(app_mean).mean(axis=0) / 255
                        app_std = np.array(app_std).mean(axis=0) / 255
                        motion_mean = np.array(motion_mean).mean(axis=0) / 255
                        motion_std = np.array(motion_std).mean(axis=0) / 255
                        pop_mean = np.stack((app_mean, motion_mean))  # 0 is app, 1 is motion
                        pop_std = np.stack((app_std, motion_std))

                    elif model_name in ['STM_Phys', 'New']:
                        pop_mean = np.array(app_mean).mean(axis=0) / 255
                        pop_std = np.array(app_std).mean(axis=0) / 255



                    device = torch.device(f"cuda:{cuda_device}" if torch.cuda.is_available() else "cpu")  # test
                    is_model_support(model_name, model_list)

                    model = get_model(model_name, pop_mean, pop_std, frame_depth=window_length, skip=skip_connection,
                                        shift_factor=shift_factor, group_on=new_group_tsm)
                    model.to(device)

                    criterion = loss_fn(loss_metric)
                    optimizer = optim.optimizer(model.parameters(), learning_rate, optim = optimi)
                    # Decay
                    scheduler = lr_scheduler.StepLR(optimizer, step_size=Step, gamma=Gamma)
                    # Exponential
                    # scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=Gamma, last_epoch=-1)
                    #Plateau
                    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
                    min_val_loss = 10000
                    min_avg_eva = float('inf')
                    min_val_loss_model = None
                    best_fold_metric = None
                    # torch.backends.cudnn.benchmark = True

                    train_loss = []
                    valid_loss = []

                    if __TIME__:
                        log_info_time("Preprocessing time \t: ", datetime.timedelta(seconds=time.time() - start_time))

                    if __TIME__:
                        start_time = time.time()
                        
                    if train == 0 or train == 1:    
                        torch.autograd.set_grad_enabled(True)
                        if train == 0:
                            start_epoch = 1
                        else:
                            checkpoint = torch.load(checkpoint_path + checkpoint_name)
                            model.load_state_dict(checkpoint["model"])
                            start_epoch = checkpoint["epoch"] + 1
                            optimizer.load_state_dict(checkpoint["optimizer"])
                            # Decay
                            scheduler.load_state_dict(checkpoint['scheduler'])
                            train_loss = checkpoint["train_loss"]
                            valid_loss = checkpoint["valid_loss"]
                            min_val_loss = valid_loss[-1]
                            min_val_loss_model = copy.deepcopy(model)

                        # if len(train_dataset) % batch_size == 1:            # for AFF MANHOB_HCI
                        #     total = len(train_loader) - 1
                        # else:
                        #     total = len(train_loader)

                        path = checkpoint_path + model_name + "_" + "_".join(dataset) + "_T_" + str(T) + "_shift_" + str(shift_factor) + '_' + str(loss_metric) + "_best_model_" + str(NF) + ".txt"
                        f = open(path, 'w')

                        for epoch in np.arange(start_epoch, tot_epochs + 1):
                            with tqdm(train_loader, desc="Train ", total=len(train_loader), colour='red') as tepoch:
                                model.train()
                                running_loss = 0.0
                                for inputs, target in tepoch:
                                    # if inputs[0].shape[0] == 1:
                                    #     continue
                                    optimizer.zero_grad(set_to_none=True)
                                    if torch.isnan(target).any():
                                        print('A1')
                                        return
                                    if torch.isinf(target).any():
                                        print('B1')
                                        return
                                    tepoch.set_description(f"Train Epoch {epoch}")

                                    if model_name == 'SlowFast_FD' or model_name == 'SlowFast_AM':
                                        inputs = [_inputs.cuda() for _inputs in inputs]
                                    else:
                                        inputs = inputs.to(device)
                                    target = target.to(device)
                                    outputs = model(inputs)

                                    if torch.isnan(outputs).any():
                                        print('A2')
                                        return
                                    if torch.isinf(outputs).any():
                                        print('B2')
                                        return

                                    if loss_metric == "snr":
                                        loss = criterion(outputs, target, fs)
                                    else:
                                        loss = criterion(outputs, target)
                                    loss.backward()
                                    optimizer.step()

                                    running_loss += loss.item() * target.size(0) * target.size(1)
                                    del loss, outputs, inputs, target
                                    tepoch.set_postfix(loss='%.6f' % (running_loss / len(train_loader) / window_length / batch_size))
                                train_loss.append(running_loss / len(train_loader) / window_length / batch_size)

                            if epoch == tot_epochs and min_val_loss_model is not None:
                                model = min_val_loss_model

                            # Decay
                            scheduler.step()
                            # print(f"Epoch {epoch} Learning Rate is {scheduler.get_last_lr()}")
                            with tqdm(validation_loader, desc="Validation ", total=len(validation_loader), colour='green') as tepoch:
                                model.eval()
                                running_loss = 0.0
                                hr_metrics = None

                                # if epoch == tot_epochs:
                                inference_array = []
                                target_array = []
                                # torch.set_anomaly_enabled(True)

                                with torch.no_grad():
                                    for inputs, target in tepoch:
                                        tepoch.set_description(f"Validation")
                                        if torch.isnan(target).any():
                                            print('A3')
                                            return
                                        if torch.isinf(target).any():
                                            print('B3')
                                            return
                                        if model_name == 'SlowFast_FD' or model_name == 'SlowFast_AM':
                                            inputs = [_inputs.cuda() for _inputs in inputs]
                                        else:
                                            inputs = inputs.to(device)

                                        target = target.to(device)
                                        outputs = model(inputs)
                                        if torch.isnan(outputs).any():
                                            print('A4')
                                            return
                                        if torch.isinf(outputs).any():
                                            print('B4')
                                            return
                                        if loss_metric == "snr":
                                            loss = criterion(outputs, target, fs)
                                        else:
                                            loss = criterion(outputs, target)

                                        running_loss += loss.item() * target.size(0) * target.size(1)
                                        tepoch.set_postfix(
                                            loss='%.6f' % (running_loss / len(validation_loader) / window_length / batch_size))

                                        # if epoch == tot_epochs:
                                        # inference_array.extend(normalize(torch.squeeze(outputs).cpu().detach().numpy()))
                                        # target_array.extend(normalize(torch.squeeze(target).cpu().detach().numpy()))
                                        inference_array = np.append(inference_array,
                                                                    np.reshape(outputs.cpu().detach().numpy(), (1, -1)))
                                        target_array = np.append(target_array, np.reshape(target.cpu().detach().numpy(), (1, -1)))

                                    valid_loss.append(running_loss / len(validation_loader) / window_length / batch_size)

                                    # Print additional waveform-domain metrics for quick monitoring.
                                    try:
                                        diff = inference_array - target_array
                                        val_mae = float(np.mean(np.abs(diff)))
                                        val_rmse = float(np.sqrt(np.mean(diff ** 2)))
                                        val_pearson = float(np.corrcoef(target_array, inference_array)[0][1])
                                        print(
                                            f"[Eval] Fold NF={NF} | Epoch {epoch} | Val loss={valid_loss[-1]:.6f} "
                                            f"| Val MAE={val_mae:.4f} | Val RMSE={val_rmse:.4f} | Val Pearson={val_pearson:.4f}"
                                        )
                                    except Exception as e:
                                        print(f"[Eval] Fold NF={NF} | Epoch {epoch} | (metric print failed: {e})")

                                    # HR-based metrics (paper-style windows: 30s/20s/10s)
                                    hr_metrics = compute_hr_metrics_from_validation(
                                        inference_array, target_array, valid_dataset, window_length
                                    )
                                    if hr_metrics is not None:
                                        print(
                                            f"[HR Eval] Fold NF={NF} | Epoch {epoch} "
                                            f"| MAE30={hr_metrics['mae30']:.3f} RMSE30={hr_metrics['rmse30']:.3f} P30={hr_metrics['pearson30']:.3f} "
                                            f"| MAE20={hr_metrics['mae20']:.3f} RMSE20={hr_metrics['rmse20']:.3f} "
                                            f"| MAE10={hr_metrics['mae10']:.3f} RMSE10={hr_metrics['rmse10']:.3f}"
                                        )
                            # Pleteau
                            # scheduler.step(valid_loss[-1])
                            # print(f"Epoch {epoch} Learning Rate is {learning_rate}")
                            f.write(str(valid_loss[-1])+"\n")
                            
                            #--- Save checkpoint used validation loss ---#
                            if min_val_loss > valid_loss[-1]:  # save the train model
                                min_val_loss = valid_loss[-1]
                                min_val_loss_model = copy.deepcopy(model)
                                best_fold_metric = hr_metrics

                                # --- Vẽ 3 đồ thị đánh giá tại best epoch ---
                                if best_fold_metric is not None and "_plot_data" in best_fold_metric:
                                    gt_plt, res_plt, fs_plt = best_fold_metric["_plot_data"]
                                    try:
                                        generate_evaluation_plots(
                                            gt_plt, res_plt, fs_plt,
                                            NF, checkpoint_path,
                                            model_name, dataset[0],
                                            windows=[30, 20, 10]
                                        )
                                    except Exception as e:
                                        print(f"[Plot] generate_evaluation_plots failed: {e}")
                                
                                checkpoint = {'epoch': epoch,
                                                'model': model.state_dict(),
                                                'optimizer': optimizer.state_dict(),
                                                # Decay
                                                'scheduler': scheduler.state_dict(),
                                                'train_loss': train_loss,
                                                'valid_loss': valid_loss}
                                # QAnh 5-fold
                                print(f"you update this checkpoint on epoch{epoch}")

                                torch.save(checkpoint,
                                            checkpoint_path + "/" + model_name + "_" + "_".join(dataset) + "_T_" + str(T) + "_shift_" + str(shift_factor) + '_' + str(loss_metric) + "_best_model_" + str(NF) + ".pth")


                            #--- Save checkpoint used average evaluation matrix(MAE+RMSE in window length=30,20,10) ---#
                            # inference_array_avg=[]
                            # target_array_avg=[]
                            # Step = 10//Test_SW
                            # inference_len = len(inference_array)
                            # for i in range(0, window_length[Index], 1):
                            #     sum_value_inf = 0
                            #     sum_value_tar = 0
                            #     # print(f"i={i}")
                            #     for index in range(0,i//Test_SW+1,1):
                            #         sum_value_inf = sum_value_inf + inference_array[(index*(10-Test_SW))+i]
                            #         sum_value_tar = sum_value_tar + target_array[(index*(10-Test_SW))+i]
                            #     inference_array_avg.append(sum_value_inf/(i//Test_SW+1))
                            #     target_array_avg.append(sum_value_tar/(i//Test_SW+1))                

                            # #-- the middle step --#
                            # for i in range(20-Test_SW, inference_len-((Step-1)*10), 10):
                            #     for j in range(0, Test_SW, 1):
                            #         sum_value_inf = 0
                            #         sum_value_tar = 0
                            #         for index in range(0,Step,1):
                            #             sum_value_inf = sum_value_inf + inference_array[(index*(10-Test_SW))+i+j]
                            #             # sum_value_inf = sum_value_inf + inference_array[(index*10)+j]
                            #             sum_value_tar = sum_value_tar + target_array[(index*(10-Test_SW))+i+j]
                            #         inference_array_avg.append(sum_value_inf/Step)
                            #         target_array_avg.append(sum_value_tar/Step)

                            # #-- the end step --#
                            # for i in range(-10+Test_SW, 0):
                            #     sum_value_inf = 0
                            #     sum_value_tar = 0
                            #     # print(f"i={i}")
                            #     # print(f"target_array[{i}] = {target_array[i]}")
                            #     avg_number = math.ceil(abs(i)/Test_SW)
                            #     for index in range(0,avg_number,1):
                            #         # print(f"target_array[i-(index*(10-Test_SW))] = {target_array[i-(index*(10-Test_SW))]}")
                            #         position = i-(index*(10-Test_SW))
                            #         sum_value_inf = sum_value_inf + inference_array[position]
                            #         sum_value_tar = sum_value_tar + target_array[position]              
                            #     inference_array_avg.append(sum_value_inf/avg_number)
                            #     target_array_avg.append(sum_value_tar/avg_number)     

                            # result = {}
                            # groundtruth = {}
                            # start_idx = 0
                            # n_frames_per_video = valid_dataset.n_frames_per_video   # show the total frame of each video 

                            # for i, value in enumerate(n_frames_per_video):

                            #     result[i] = normalize(inference_array_avg[start_idx:start_idx + value])
                            #     groundtruth[i] = target_array_avg[start_idx:start_idx + value]
                            #     start_idx += value

                            # result = BPF_dict(result, 25)
                            # groundtruth = BPF_dict(groundtruth, 25)
                            # mae_30, rmse_30, acc3, acc5, acc10 = HR_Metric(groundtruth, result, 25, 30, 1)   # for pure/ubfc
                            # pearson = Pearson_Corr(groundtruth, result)
                            # print('MAE 30s: ' + str(round(mae_30, 3)))
                            # print('RMSE 30s: ' + str(round(rmse_30, 3)))
                            # mae_20, rmse_20, acc3, acc5, acc10c = HR_Metric(groundtruth, result, 25, 20, 1)
                            # mae_10, rmse_10, acc3, acc5, acc10 = HR_Metric(groundtruth, result, 25, 10, 1)
                            # avg_eva = (mae_30 + rmse_30 + mae_20 + rmse_20 + mae_10 + rmse_10) / 6
                            # print(f"epoch{epoch}, avg_eva={avg_eva}")
                            # if avg_eva < min_avg_eva:
                            #     min_avg_eva = avg_eva
                            #     checkpoint = {'epoch': epoch,
                            #                     'model': model.state_dict(),
                            #                     'optimizer': optimizer.state_dict(),
                            #                     # Decay
                            #                     'scheduler': scheduler.state_dict(),
                            #                     'loss': loss,
                            #                     'train_loss': train_loss,
                            #                     'valid_loss': valid_loss}
                            #     # QAnh 5-fold
                            #     print(f"you update this checkpoint on epoch{epoch}")

                            #     torch.save(checkpoint,
                            #                 checkpoint_path + "/" + model_name + "_" + "_".join(dataset) + "_T_" + str(T) + "_shift_" + str(shift_factor[Index]) + '_' + str(loss_metric) + "_best_model_" + str(NF) + ".pth")
                            #     # first_layer = model.motion_model_fast.block_1.tsm.conv_2d[0].weight.data.clone()
                            #     # visTensor(first_layer, ch=0, allkernels=False)
                            #     #
                            #     # plt.axis('off')
                            #     # plt.ioff()
                            #     # plt.show()

                            #     result = {}
                            #     groundtruth = {}
                            #     start_idx = 0
                            #     n_frames_per_video = valid_dataset.n_frames_per_video
                            #     vid_fs = valid_dataset.video_fs
                            #     for i, value in enumerate(n_frames_per_video):
                            #         # if dataset_name == 'PURE':
                            #         #     result[i] = inference_array[start_idx:start_idx + value]
                            #         # else:
                            #         result[i] = normalize(inference_array[start_idx:start_idx + value])
                            #         groundtruth[i] = target_array[start_idx:start_idx + value]
                            #         start_idx += value

                            #     # plot_loss_graph(train_loss, valid_loss)
                            #     # plot_graph(0, 500, groundtruth[3], result[3])
                            #     result = BPF_dict(result, vid_fs)
                            #     groundtruth = BPF_dict(groundtruth, vid_fs)
                            #     # plot_graph(0, 500, groundtruth[3], result[3])

                            #     mae, rmse, acc3, acc5, acc10 = HR_Metric(groundtruth, result, vid_fs, 30, 1)
                            #     pearson = Pearson_Corr(groundtruth, result)

                            #     # print(checkpoint_name)

                            #     print('MAE 30s: ' + str(round(mae, 3)))
                            #     print('RMSE 30s: ' + str(round(rmse, 3)))
                            #     # print('Accuracy 3 30s: ' + str(round(acc3, 3)))
                            #     # print('Accuracy 5 30s: ' + str(round(acc5, 3)))
                            #     # print('Accuracy 10 30s: ' + str(round(acc10, 3)))

                            #     print('Pearson 30s: ' + str(round(pearson, 3)))

                            #     mae, rmse, acc3, acc5, acc10c = HR_Metric(groundtruth, result, vid_fs, 20, 1)
                            #     print('MAE 20s: ' + str(round(mae, 3)))
                            #     print('RMSE 20s: ' + str(round(rmse, 3)))
                            #     # print('Accuracy 3 30s: ' + str(round(acc3, 3)))
                            #     # print('Accuracy 5 30s: ' + str(round(acc5, 3)))
                            #     # print('Accuracy 10 30s: ' + str(round(acc10, 3)))

                            #     mae, rmse, acc3, acc5, acc10 = HR_Metric(groundtruth, result, vid_fs, 10, 1)
                            #     print('MAE 10s: ' + str(round(mae, 3)))
                            #     print('RMSE 10s: ' + str(round(rmse, 3)))
                            #     # print('Accuracy 3 30s: ' + str(round(acc3, 3)))
                            #     # print('Accuracy 5 30s: ' + str(round(acc5, 3)))
                            #     # print('Accuracy 10 30s: ' + str(round(acc10, 3)))

                            # if epoch % 5 == 0:
                            #     checkpoint = {'epoch': epoch,
                            #                   'model': model.state_dict(),
                            #                   'optimizer': optimizer.state_dict(),
                            #                   # 'scheduler': scheduler.state_dict(),
                            #                   'loss': loss,
                            #                   'train_loss': train_loss,
                            #                   'valid_loss': valid_loss}
                            #     torch.save(checkpoint, checkpoint_path + "/" + model_name + "_" + dataset_name + "_" + str(epoch) + "_" +
                            #                str(running_loss / len(validation_loader) / window_length / batch_size) + '.pth')


                            # temporary closed
                            # if epoch == tot_epochs:
                            #     # first_layer = model.motion_model_fast.block_1.tsm.conv_2d[0].weight.data.clone()
                            #     # visTensor(first_layer, ch=0, allkernels=False)
                            #     #
                            #     # plt.axis('off')
                            #     # plt.ioff()
                            #     # plt.show()

                            #     result = {}
                            #     groundtruth = {}
                            #     start_idx = 0
                            #     n_frames_per_video = valid_dataset.n_frames_per_video
                            #     vid_fs = valid_dataset.video_fs
                            #     for i, value in enumerate(n_frames_per_video):
                            #         # if dataset_name == 'PURE':
                            #         #     result[i] = inference_array[start_idx:start_idx + value]
                            #         # else:
                            #         result[i] = normalize(inference_array[start_idx:start_idx + value])
                            #         groundtruth[i] = target_array[start_idx:start_idx + value]
                            #         start_idx += value

                            #     # plot_loss_graph(train_loss, valid_loss)
                            #     # plot_graph(0, 500, groundtruth[3], result[3])
                            #     result = BPF_dict(result, vid_fs)
                            #     groundtruth = BPF_dict(groundtruth, vid_fs)
                            #     # plot_graph(0, 500, groundtruth[3], result[3])

                            #     mae, rmse, acc3, acc5, acc10 = HR_Metric(groundtruth, result, vid_fs, 30, 1)
                            #     pearson = Pearson_Corr(groundtruth, result)

                            #     # print(checkpoint_name)

                            #     print('MAE 30s: ' + str(round(mae, 3)))
                            #     print('RMSE 30s: ' + str(round(rmse, 3)))
                            #     # print('Accuracy 3 30s: ' + str(round(acc3, 3)))
                            #     # print('Accuracy 5 30s: ' + str(round(acc5, 3)))
                            #     # print('Accuracy 10 30s: ' + str(round(acc10, 3)))

                            #     print('Pearson 30s: ' + str(round(pearson, 3)))

                            #     mae, rmse, acc3, acc5, acc10c = HR_Metric(groundtruth, result, vid_fs, 20, 1)
                            #     print('MAE 20s: ' + str(round(mae, 3)))
                            #     print('RMSE 20s: ' + str(round(rmse, 3)))
                            #     # print('Accuracy 3 30s: ' + str(round(acc3, 3)))
                            #     # print('Accuracy 5 30s: ' + str(round(acc5, 3)))
                            #     # print('Accuracy 10 30s: ' + str(round(acc10, 3)))

                            #     mae, rmse, acc3, acc5, acc10 = HR_Metric(groundtruth, result, vid_fs, 10, 1)
                            #     print('MAE 10s: ' + str(round(mae, 3)))
                            #     print('RMSE 10s: ' + str(round(rmse, 3)))
                            #     # print('Accuracy 3 30s: ' + str(round(acc3, 3)))
                            #     # print('Accuracy 5 30s: ' + str(round(acc5, 3)))
                            #     # print('Accuracy 10 30s: ' + str(round(acc10, 3)))

                            # # torch.cuda.empty_cache()
                            # # model=None

                        f.close()
                        
                        # --- Thêm code vẽ biểu đồ Learning Curve ---
                        try:
                            plt.figure(figsize=(10, 6))
                            epochs_range = range(1, len(train_loss) + 1)
                            plt.plot(epochs_range, train_loss, label='Train Loss', color='blue', marker='o')
                            plt.plot(epochs_range, valid_loss, label='Validation Loss', color='red', marker='s')
                            plt.title(f'Learning Curve - Fold {NF}')
                            plt.xlabel('Epochs')
                            plt.ylabel('Loss')
                            plt.xticks(epochs_range)
                            plt.legend()
                            plt.grid(True)
                            
                            plot_filename = f"Learning_Curve_{model_name}_{dataset[0]}_Fold_{NF}.png"
                            plot_path = os.path.join(checkpoint_path, plot_filename)
                            plt.savefig(plot_path)
                            plt.close()
                            print(f"Saved Learning Curve to {plot_path}")
                        except Exception as e:
                            print(f"Failed to save Learning Curve: {e}")
                        # ----------------------------------------
                        if best_fold_metric is not None:
                            # Loại bỏ _plot_data (dữ liệu tín hiệu lớn) trước khi lưu vào summary
                            best_fold_metric_clean = {k: v for k, v in best_fold_metric.items() if k != "_plot_data"}
                            fold_best_metrics.append(best_fold_metric_clean)
                            print(
                                f"[Fold Summary] Fold NF={NF} (best by val_loss) "
                                f"| MAE30={best_fold_metric['mae30']:.3f} RMSE30={best_fold_metric['rmse30']:.3f} P30={best_fold_metric['pearson30']:.3f} "
                                f"| MAE20={best_fold_metric['mae20']:.3f} RMSE20={best_fold_metric['rmse20']:.3f} "
                                f"| MAE10={best_fold_metric['mae10']:.3f} RMSE10={best_fold_metric['rmse10']:.3f}"
                            )
                        if __TIME__:
                            log_info_time("Total training time \t: ", datetime.timedelta(seconds=time.time() - start_time))

                # Final 5-fold average/std summary for paper reporting
                if len(fold_best_metrics) > 0:
                    keys = ["mae30", "rmse30", "pearson30", "mae20", "rmse20", "mae10", "rmse10"]
                    agg = {k: np.array([m[k] for m in fold_best_metrics], dtype=np.float64) for k in keys}
                    print("\n================ 5-Fold Final Summary (best val_loss per fold) ================")
                    print(f"Folds counted: {len(fold_best_metrics)}")
                    print(f"MAE 30s    : {agg['mae30'].mean():.3f} ± {agg['mae30'].std(ddof=0):.3f}")
                    print(f"RMSE 30s   : {agg['rmse30'].mean():.3f} ± {agg['rmse30'].std(ddof=0):.3f}")
                    print(f"Pearson 30s: {agg['pearson30'].mean():.3f} ± {agg['pearson30'].std(ddof=0):.3f}")
                    print(f"MAE 20s    : {agg['mae20'].mean():.3f} ± {agg['mae20'].std(ddof=0):.3f}")
                    print(f"RMSE 20s   : {agg['rmse20'].mean():.3f} ± {agg['rmse20'].std(ddof=0):.3f}")
                    print(f"MAE 10s    : {agg['mae10'].mean():.3f} ± {agg['mae10'].std(ddof=0):.3f}")
                    print(f"RMSE 10s   : {agg['rmse10'].mean():.3f} ± {agg['rmse10'].std(ddof=0):.3f}")
                    print("===============================================================================\n")


                    if train == 2:
                        checkpoint = torch.load(checkpoint_path + checkpoint_name)
                        model.load_state_dict(checkpoint["model"])
                        model.eval()
                        if __TIME__:
                            start_time = time.time()
                        test_dataset = dataset_loader(train, save_root_path=save_root_path, model_name=model_name,
                                                        dataset_name=dataset_name, window_length=window_length)

                        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
                                                    drop_last=True)

                        with tqdm(test_loader, desc="test ", total=len(test_loader)) as tepoch:
                            inference_array = []
                            target_array = []
                            with torch.no_grad():
                                for inputs, target in tepoch:
                                    tepoch.set_description("test")
                                    inputs = inputs.to(device)
                                    target = target.to(device)
                                    outputs = model(inputs)

                                    inference_array.extend(outputs.cpu().detach().numpy())
                                    target_array.extend(target.cpu().detach().numpy())

                                if __TIME__:
                                    log_info_time("inference time \t: ", datetime.timedelta(seconds=time.time() - start_time))

                        # plot_graph(0, 300, target_array, inference_array)
                        if len(np.shape(inference_array)) >= 2:
                            inference_array = np.reshape(inference_array, (-1, 1))
                        if len(np.shape(target_array)) >= 2:
                            target_array = np.reshape(target_array, (-1, 1))
                        print('MAE: ' + str(MAE(target_array, inference_array)[0]))
                        print('RMSE: ' + str(RMSE(target_array, inference_array)))
                        print('Pearson: ' + str(pearson_corr(target_array, inference_array)[0]))


if __name__ == '__main__':
    main()
    
