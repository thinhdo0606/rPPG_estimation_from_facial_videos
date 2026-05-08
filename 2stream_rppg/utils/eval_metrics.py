from math import sqrt
import numpy as np


def HR_Metric(groundtruth, prediction, fs, window, step):
    count = 0
    correct3 = 0
    correct5 = 0
    correct10 = 0
    error_sum = 0.
    squared_error_sum = 0.
    samples = fs * window   # window 10s, 20s, 30s
    step = fs * step
    
    for i in np.arange(len(groundtruth.keys())):
        target_signal = groundtruth[i] 
        predict_signal = prediction[i]
        signal_length = len(target_signal)
        for j in np.arange(0, signal_length, step):
            # if j + samples >= signal_length:
            #     print("Sample length is smaller than window {window} length")
            #     count = 1
            #     # samples = signal_length - j
            #     break
            predict_segment = predict_signal[j:j + samples]
            target_segment = target_signal[j:j + samples]

            predict_fft = np.square(np.abs(np.fft.rfft(predict_segment)))
            gt_fft = np.square(np.abs(np.fft.rfft(target_segment)))
            frequency = (np.linspace(0, fs / 2, len(predict_fft)))

            # start = np.where(frequency >= 0.67)[0][0]
            # end = np.where(frequency >= 2.5)[0][0]
            # predict_fft[:start] = 0
            # predict_fft[end:] = 0
            # gt_fft[:start] = 0
            # gt_fft[end:] = 0
            pre_idx = np.argmax(predict_fft)
            # print(pre_idx)
            gt_idx = np.argmax(gt_fft)
            predict_hr = frequency[pre_idx] * 60
            gt_hr = frequency[gt_idx] * 60

            error_sum += abs(predict_hr - gt_hr)
            if abs(predict_hr - gt_hr) < 3:
                correct3 += 1
            if abs(predict_hr - gt_hr) < 5:
                correct5 += 1
            if abs(predict_hr - gt_hr) < 10:
                correct10 += 1
            squared_error_sum += abs(predict_hr - gt_hr)**2
            count += 1
    mae_loss = error_sum / count
    rmse_loss = sqrt(squared_error_sum / count)
    acc3 = correct3/count
    acc5 = correct5/count
    acc10 = correct10/count

    return mae_loss, rmse_loss, acc3, acc5, acc10


def Pearson_Corr(target, inference):
    pearson = 0
    for i in np.arange(len(target.keys())):
        target_signal = target[i]
        inference_signal = inference[i]
        pearson += np.corrcoef(target_signal, inference_signal)[0][1]
    pearson /= len(target.keys())
    return pearson
