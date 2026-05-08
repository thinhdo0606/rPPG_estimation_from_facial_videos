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
import matplotlib.pyplot as plt
from torchvision import utils
import torch.nn as nn
import os
import random
import math
from FCN import FCN8s, FCN16s, FCN32s, FCNs, VGGNet
from Skin_Seg import main as Skin_segmentation
import cv2



# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
    save_root_path = os.path.join(os.path.expanduser("~"), "Thinh_Two_Stream_rppg", "Dataset_rppg") + "/"

    checkpoint_path = os.path.join(
        os.path.expanduser("~"),
        "Thinh_Two_Stream_rppg",
        "Dataset_rppg",
        "2stream_rppg_checkpoints",
        "retrain_PURE_72",
    ) + "/"
    # checkpoint_path = "Hao_Checkpoints/MMSE_72_motion_normalized/"


    # dataset_name = [["PURE"], ["MMSE"], ["UBFC"], ["MANHOB_HCI"]]
    print("Checkpoint Path: ", checkpoint_path)
    dataset_name = [["PURE"]]
    # window_length = [5, 7, 10, 13, 15, 20]              # T (window_length)
    # shift_factor =  [0.2, 0.15, 0.1, 0.08, 0.07, 0.06]             # n/T (the ratio of shift) default: 0.25
    window_length_l = [10]              # T (window_length)q
    shift_factor_l =  [0.25]             # n/T (the ratio of shift) default: 0.25
    loss_metric = "combined_loss"      # "combined_loss" "snr" "mse"
    optimi = "ada_delta"
    ROI = 72                            # Adjust faceROI to 36 | 72
    batch_size = 16                     # ImgROI=36: 32   ImgROI=72: 16   ImgROI_deeper=72: 8

    tot_epochs = 10
    model_list = ["MTTS", "TSDAN", "MTTS_CSTM", "SlowFast_FD", "SlowFast_AM"]
    # fs = 30                                                                                           

    # Decay parameter
    learning_rate = 0.5
    Gamma = 0.8
    Step = 4
    # Exponential
    # Gamma = 0.8

    skip_connection = True      # True: Residual  False: In-place
    new_group_tsm = False
    cuda_device=0
    print(f"GPU device={cuda_device}")  
    setup_seed(20)              # Initial seed:20 | different seed:30

    if __TIME__:

        start_time = time.time()

    if train == 0 or train == 1:
        # for i in range(5):
        # checkpoint_name = "TS_CST_MMSE_T_10_shift_0.5_best_model" + str(i) + ".pth"
        # checkpoint_name = "TS_CST_MMSE_T_10_shift_0.5_best_model.pth"    
        for (window_length, shift_factor) in zip(window_length_l, shift_factor_l):
            for dataset in dataset_name:
                # print(f"type(dataset)={type(dataset)}, dataset={dataset}")
                for NF in range(4, 6):
                    # setup_seed(20)
                    print(f"window_length[Index]={window_length}")
                    train_dataset, valid_dataset = dataset_loader(train, save_root_path, model_name, dataset, window_length,
                                                                    fold=NF, SW=window_length, ImgROI=ROI)

                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                num_workers=2, pin_memory=True, drop_last=False)

                    validation_loader = DataLoader(valid_dataset, batch_size=batch_size, sampler=SequentialSampler(valid_dataset),
                                                    num_workers=2, pin_memory=True, drop_last=False)
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
                                
                                # motion_data = motion_data.view(B * one, T, C, H, W)
                                # app_data = app_data.view(B * one, T, C, H, W)
                                # motion_data = motion_data.reshape(B * T, C, H, W)
                                # app_data = app_data.reshape(B * T, C, H, W)

                                # batch_motion_mean = torch.mean(motion_data, dim=(0, 2, 3)).tolist()
                                # batch_motion_std = torch.std(motion_data, dim=(0, 2, 3)).tolist()
                                # batch_app_mean = torch.mean(app_data, dim=(0, 2, 3)).tolist()
                                # batch_app_std = torch.std(app_data, dim=(0, 2, 3)).tolist()

                                # frames = self.video[idx:idx + self.window_length + 1]
                                # average_frame_np = np.array(batch_app_mean)
                                # average_frame_np = cv2.normalize(average_frame_np, None, 0, 255, cv2.NORM_MINMAX)
                                # average_frame_np = average_frame_np.astype(np.uint8)
                                # cv2.imshow("avg frame", average_frame_np)
                                # cv2.waitKey(0)
                                # cv2.destroyAllWindows()

                                # app_mean.append(batch_app_mean)
                                # app_std.append(batch_app_std)
                                # motion_mean.append(batch_motion_mean)
                                # motion_std.append(batch_motion_std)
                                
                                #---Training for central frame---#
                                # central_frame_index = T//2
                                # central_frame = app_data[:, :, central_frame_index, :, :, :]
                                # central_frame = central_frame.view(B * one, C, H, W)
                                # motion_data = motion_data.view(B * one, T, C, H, W)
                                # motion_data = motion_data.reshape(B * T, C, H, W)

                                # batch_motion_mean = torch.mean(motion_data, dim=(0, 2, 3)).tolist()
                                # batch_motion_std = torch.std(motion_data, dim=(0, 2, 3)).tolist()
                                # batch_app_mean = torch.mean(central_frame, dim=(0, 2, 3)).tolist()
                                # batch_app_std = torch.std(central_frame, dim=(0, 2, 3)).tolist()
                                #----------------------------#

                                #---Training for central frame + Skin Segmentation---#
                                central_frame_index = T//2
                                central_frame = app_data[:, :, central_frame_index, :, :, :]
                                central_frame = central_frame.view(B * one, C, H, W)
                                motion_data = motion_data.view(B * one, T, C, H, W)
                                # app_data = app_data.view(B * one, T, C, H, W)
                                motion_data = motion_data.reshape(B * T, C, H, W)
                                # app_data = app_data.reshape(B * T, C, H, W)

                                # print(app_mean.shape)
                                #-- Skin Segmentation --#
                                # Initialize a list to store the results from Skin_Seg function
                                processed_c_frames = []

                                # Loop through each central frame and apply the Skin_Seg function
                                for i in range(central_frame.size(0)):  # Iterate through the batch
                                    frame = central_frame[i]
                                    # print(frame.shape)
                                    frame = frame.squeeze(0)   # reduce the first dimension
                                    frame_np = frame.permute(1, 2, 0).cpu().numpy()  # Change shape from (C, H, W) to (H, W, C) and move to CPU
                                    frame_np = frame_np.astype(np.uint8)
                                    processed_frame_np = Skin_segmentation(frame_np)

                                    # app_mean = np.mean(frame, axis=0)
                                    # print(type(processed_frame_np))
                                    # frame_test = np.array(processed_frame_np)
                                    # print(frame_test.shape)

                                    # average_frame_np = cv2.normalize(processed_frame_np, None, 0, 255, cv2.NORM_MINMAX)
                                    # cv2.imshow("avg frame", processed_frame_np)
                                    # cv2.waitKey(0)
                                    # cv2.destroyAllWindows()

                                    processed_frame = torch.from_numpy(processed_frame_np).permute(2, 0, 1)  # Change shape back to (C, H, W)
                                    processed_c_frames.append(processed_frame)
                                processed_c_frames = torch.stack(processed_c_frames)

                                # central_frame = central_frame.view(B * one, C, H, W)
                                processed_c_frames = processed_c_frames.view(B * one, C, H, W)
                    
                                batch_motion_mean = torch.mean(motion_data, dim=(0, 2, 3)).tolist()
                                batch_motion_std = torch.std(motion_data, dim=(0, 2, 3)).tolist()
                                batch_app_mean = torch.mean(processed_c_frames, dim=(0, 2, 3)).tolist()
                                batch_app_std = torch.std(processed_c_frames, dim=(0, 2, 3)).tolist()
                                #-- End of Skin Segmentation --#



                                app_mean.append(batch_app_mean)
                                app_std.append(batch_app_std)
                                motion_mean.append(batch_motion_mean)
                                motion_std.append(batch_motion_std)
                                #------------------------------------------------#
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
                        # test = np.array(app_mean)
                        # print(test.shape)
                        for i, data in enumerate(validation_loader):
                            if model_name in ['TSDAN', 'MTTS', 'MTTS_CSTM']:
                                data = data[0]  # shape (Batch, T+1, H, W, 6)
                                motion_data, app_data = torch.tensor_split(data, 2, dim=1)
                                B, one, T, C, H, W = motion_data.shape

                                #---Validation for avg frame---#
                                # motion_data = motion_data.view(B * one, T, C, H, W)
                                # app_data = app_data.view(B * one, T, C, H, W)
                                # motion_data = motion_data.reshape(B * T, C, H, W)
                                # app_data = app_data.reshape(B * T, C, H, W)

                                # batch_motion_mean = torch.mean(motion_data, dim=(0, 2, 3)).tolist()
                                # batch_motion_std = torch.std(motion_data, dim=(0, 2, 3)).tolist()
                                # batch_app_mean = torch.mean(app_data, dim=(0, 2, 3)).tolist()
                                # batch_app_std = torch.std(app_data, dim=(0, 2, 3)).tolist()
                                
                                #---Validation for central frame---#
                                central_frame_index = T//2
                                central_frame = app_data[:, :, central_frame_index, :, :, :]
                                central_frame = central_frame.view(B * one, C, H, W)
                                motion_data = motion_data.view(B * one, T, C, H, W)
                                motion_data = motion_data.reshape(B * T, C, H, W)

                                batch_motion_mean = torch.mean(motion_data, dim=(0, 2, 3)).tolist()
                                batch_motion_std = torch.std(motion_data, dim=(0, 2, 3)).tolist()
                                batch_app_mean = torch.mean(central_frame, dim=(0, 2, 3)).tolist()
                                batch_app_std = torch.std(central_frame, dim=(0, 2, 3)).tolist()
                                #----------------------------#

                                #---Validation for central frame + Skin Segmentation---#
                                # central_frame_index = T//2
                                # central_frame = app_data[:, :, central_frame_index, :, :, :]
                                # motion_data = motion_data.view(B * one, T, C, H, W)
                                # motion_data = motion_data.reshape(B * T, C, H, W)
                                # #-- Skin Segmentation --#
                                # # Initialize a list to store the results from Skin_Seg function
                                # processed_c_frames = []

                                # # Loop through each central frame and apply the Skin_Seg function
                                # for i in range(central_frame.size(0)):  # Iterate through the batch
                                #     frame = central_frame[i]
                                #     # print(frame.shape)
                                #     frame = frame.squeeze(0)   # reduce the first dimension
                                #     frame_np = frame.permute(1, 2, 0).cpu().numpy()  # Change shape from (C, H, W) to (H, W, C) and move to CPU
                                #     processed_frame_np = Skin_segmentation(frame_np)
                                #     processed_frame = torch.from_numpy(processed_frame_np).permute(2, 0, 1)  # Change shape back to (C, H, W)
                                #     processed_c_frames.append(processed_frame)
                                # processed_c_frames = torch.stack(processed_c_frames)

                                # # central_frame = central_frame.view(B * one, C, H, W)
                                # processed_c_frames = processed_c_frames.view(B * one, C, H, W)
                                # #-- End of Skin Segmentation --#

                                # batch_motion_mean = torch.mean(motion_data, dim=(0, 2, 3)).tolist()
                                # batch_motion_std = torch.std(motion_data, dim=(0, 2, 3)).tolist()
                                # batch_app_mean = torch.mean(processed_c_frames, dim=(0, 2, 3)).tolist()
                                # batch_app_std = torch.std(processed_c_frames, dim=(0, 2, 3)).tolist()
                                #------------------------------------------------#


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
                        # app_mean_array = np.array(app_mean)

                    elif model_name in ['STM_Phys', 'New']:
                        pop_mean = np.array(app_mean).mean(axis=0) / 255
                        pop_std = np.array(app_std).mean(axis=0) / 255



                    device = torch.device(f"cuda:{cuda_device}" if torch.cuda.is_available() else "cpu")  # test
                    is_model_support(model_name, model_list)

                    model = get_model(model_name, pop_mean, pop_std, frame_depth=window_length, skip=skip_connection,
                                        shift_factor=shift_factor, group_on=new_group_tsm)
                    # model= nn.DataParallel(model)
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
                            checkpoint = torch.load(checkpoint_path + "/" + model_name + "_" + "_".join(dataset) + "_T_" + str(T) + "_shift_" + str(shift_factor) + '_' + str(loss_metric) + "_best_model_" + str(NF) + ".pth")
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
                            # Pleteau
                            # scheduler.step(valid_loss[-1])
                            # print(f"Epoch {epoch} Learning Rate is {learning_rate}")
                            f.write(str(valid_loss[-1])+"\n")
                            
                            #--- Save checkpoint used validation loss ---#
                            if min_val_loss > valid_loss[-1]:  # save the train model
                                min_val_loss = valid_loss[-1]
                                min_val_loss_model = copy.deepcopy(model)
                                
                                checkpoint = {'epoch': epoch,
                                                'model': model.state_dict(),
                                                'optimizer': optimizer.state_dict(),
                                                # Decay
                                                'scheduler': scheduler.state_dict(),
                                                'loss': loss,
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
                        if __TIME__:
                            log_info_time("Total training time \t: ", datetime.timedelta(seconds=time.time() - start_time))


                    elif train == 2:
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
    
