import gc
import multiprocessing
import os
import h5py
import random
import time
import datetime
import natsort
# natsorted() identifies numbers anywhere in a string and sorts them naturally
from log import log_info_time
# from utils.image_preprocess import preprocess_Video_RGB_only
from image_preprocess_Hao import preprocess_Video_RGB_only_Hao
from text_preprocess import *


def preprocessing(init: bool = True,
                  save_root_path: str = "Face_Tracking_Test/",
                  data_root_path: str = "DATASETS/",
                  dataset_name: str = "PURE",
                  cv_ratio: int = 0.8):

    manager = multiprocessing.Manager() #Multiprocessing Manager provides a way of creating centralized Python objects that can be shared safely among processes.

    if dataset_name == "PURE":
        dataset_root_path = data_root_path + dataset_name
        data_list = [data for data in natsort.natsorted(os.listdir(dataset_root_path))]
        random.shuffle(data_list)
        print("data_list_len={}".format(len(data_list)))
    elif dataset_name == "UBFC":
        dataset_root_path = data_root_path + dataset_name
        data_list = [data for data in natsort.natsorted(os.listdir(dataset_root_path)) if data.__contains__("subject")]
        random.shuffle(data_list)
    elif dataset_name == "VIPL":
        dataset_root_path = data_root_path + dataset_name + '/data'
        data_list = [data for data in natsort.natsorted(os.listdir(dataset_root_path))]
        random.shuffle(data_list)
    elif dataset_name == "MMSE":
        dataset_root_path = data_root_path + dataset_name
        data_list = [data for data in natsort.natsorted(os.listdir(dataset_root_path))]
        random.shuffle(data_list)
    elif dataset_name == "cohface" or "MANHOB_HCI":
        dataset_root_path = data_root_path + dataset_name
        data_list = [data for data in natsort.natsorted(os.listdir(dataset_root_path)) if data.isdigit()]
        random.shuffle(data_list)
    else:
        print('Not supported dataset')
        return

    # # TxCxHxW -> C is the RGB color channels HxW(36*36) is the spartial size
    # img_size = 36
    # # chunk_shape = (200, img_size, img_size, 3)

    # threads = 6
    # for i in np.arange(0, len(data_list), threads):
    #     if i + threads > len(data_list):
    #         threads = len(data_list) - i
    #     process = []
    #     return_dict = manager.dict()
    #     for data_path in data_list[i:i+threads]:  # 6 threads deal with each video in data_list(shuffle) for accelerating computation
    #         proc = multiprocessing.Process(target=preprocess_dataset, args=(dataset_root_path + "/" + data_path, True,
    #                                                                         dataset_name, return_dict, img_size))
    #         process.append(proc)
    #         proc.start()
    #     for proc in process:
    #         proc.join()
    #         proc.terminate()

    #     file = h5py.File(save_root_path + dataset_name + ".hdf5", "a")
    #     print(return_dict)
    #     for data_path in return_dict.keys():
    #         dset = file.create_group(data_path)
    #         video_data = return_dict[data_path]['video']
    #         label_data = return_dict[data_path]['label']
    #         video_shape = video_data.shape
    #         label_shape = label_data.shape
    #         dset.create_dataset('video', video_shape, np.uint8, video_data, chunks=video_shape)
    #         dset.create_dataset('label', label_shape, np.float32, label_data, chunks=label_shape)
    #         # dset['video'] = return_dict[data_path]['video']
    #         # dset['label'] = return_dict[data_path]['label']
    #     # for data_path in return_dict.keys():
    #     #   input_vid = return_dict[data_path]['video']
    #         # nofs = input_vid.shape[0]
    #     #     label = return_dict[data_path]['label'].reshape(-1, 1)
    #     #
    #     #     if init:
    #     #         file.create_dataset('video', data=input_vid, shape=(nofs, img_size, img_size, chunk_shape[3]),
    #     #                             dtype=np.uint8, chunks=chunk_shape, maxshape=(None, img_size, img_size, chunk_shape[3]))
    #     #         file.create_dataset('label', data=label, shape=(nofs, 1), dtype=np.float32, chunks=(
    #     #             chunk_shape[0], 1), maxshape=(None, 1))
    #     #         init = False
    #     #     else:
    #     #         file['video'].resize((file['video'].shape[0] + nofs), axis=0)
    #     #         file['video'][-nofs:] = input_vid
    #     #         file['label'].resize((file['label'].shape[0] + nofs), axis=0)
    #     #         file['label'][-nofs:] = label
    #     file.close()
    #     #
    #     del process, return_dict
    #     gc.collect()

    # # file = h5py.File(save_root_path + dataset_name + ".hdf5", "r")
    # # len_dataset = len(file['label'])
    # # cut_index = int(cv_ratio * len_dataset)  # = cut index - 1

    # # read the fully datasets
    # print(save_root_path + dataset_name + ".hdf5")
    # file = h5py.File(save_root_path + dataset_name + ".hdf5", "r")  # file is equal to the entire dataset
    # train_length = int(len(file.keys()) * cv_ratio)
    # # create a empty file for saved training sets
    # train_file = h5py.File(save_root_path + dataset_name + "_train.hdf5", "w")
    # for data_path in list(file.keys())[:train_length]:
    #     file.copy(file[data_path], train_file, data_path)  #  train = 0.8 * file
    # train_file.close()  # close train file

    # # create a empty file for saved test sets
    # test_file = h5py.File(save_root_path + dataset_name + "_test.hdf5", "w")
    # for data_path in list(file.keys())[train_length:]:  # test = the rest of file (0.2 * file)
    #     file.copy(file[data_path], test_file, data_path)
    # test_file.close()  # close test file 
    # file.close()  # close the entire dataset

    # file_train.create_dataset('video', shape=(cut_index, img_size, img_size, chunk_shape[3]), dtype=np.uint8,
    #                           data=file['video'][:cut_index], chunks=chunk_shape)
    # file_train.create_dataset('label', shape=(cut_index, 1), dtype=np.float32,
    #                           data=file['label'][:cut_index], chunks=(chunk_shape[0], 1))
    # file_train.close()

    # file_test = h5py.File(save_root_path + dataset_name + "_test.hdf5", "w")
    # file_test.create_dataset('video', shape=(len_dataset - cut_index, img_size, img_size, chunk_shape[3]),
    #                          dtype=np.uint8, data=file['video'][cut_index:], chunks=chunk_shape)
    # file_test.create_dataset('label', shape=(len_dataset - cut_index, 1), dtype=np.float32,
    #                          data=file['label'][cut_index:], chunks=(chunk_shape[0], 1))
    # file_test.close()
    # file.close()

    log_info_time("Data Processing Time \t: ", datetime.timedelta(seconds=time.time() - start_time))


def preprocess_dataset(path, flag, dataset_name, return_dict, img_size):
    if dataset_name == 'UBFC':
        rst, video = preprocess_Video_RGB_only_Hao(path + "/vid.avi", flag, vid_res=img_size)
        if not rst:
            return
        else:
            label = UBFC_preprocess_Label(path + "/ground_truth.txt", video.shape[0])
            if label is None:
                return
            else:
                return_dict[path.split("/")[-1]] = {'video': video, 'label': label}

    elif dataset_name == 'cohface':
        for i in os.listdir(path):
            rst, video = preprocess_Video_RGB_only_Hao(path + '/' + i + '/data.avi', flag, vid_res=img_size)
            if not rst:
                return
            else:
                label = MTTS_cohface_Label(path + '/' + i + '/data.hdf5', video.shape[0])
                if label is None:
                    return
                else:
                    return_dict[path.split("/")[-1] + '_' + i] = {'video': video, 'label': label}

    elif dataset_name == 'VIPL':
        for v in os.listdir(path):
            for source in os.listdir(path + '/' + v):
                if source != 'source4':
                    rst, video = preprocess_Video_RGB_only_Hao(path + '/' + v + '/' + source + '/video.avi', flag, vid_res=img_size)
                    if not rst:
                        return
                    else:
                        label = MTTS_VIPL_Label(path + '/' + v + '/' + source + '/wave.csv', video.shape[0])
                        if label is None:
                            return
                        else:
                            return_dict[path.split("/")[-1] + '_' + v + '_' + source] = {'video': video, 'label': label}

    elif dataset_name == 'MMSE':
        for v in os.listdir(path):
            rst, video = preprocess_Video_RGB_only_Hao(path + '/' + v + '/video.avi', flag, vid_res=img_size)
            if not rst:
                return
            else:
                label = MTTS_MMSE_Label(path + '/' + v + '/BP_mmHg.txt', video.shape[0])
                if label is None:
                    return
                else:
                    return_dict[path.split("/")[-1] + '_' + v] = {'video': video, 'label': label}

    elif dataset_name == 'MANHOB_HCI':
        rst, video = preprocess_Video_RGB_only_Hao(path + "/vid.avi", flag, vid_res=img_size)
        if not rst:
            return
        else:
            label = MTTS_MANHOB_Label(path + "/ground_truth.txt", video.shape[0])
            if label is None:
                return
            else:
                return_dict[path.split("/")[-1]] = {'video': video, 'label': label}

    if dataset_name == 'PURE':
        rst, video = preprocess_Video_RGB_only_Hao(path + "/" + str(path.split("/")[-1]) + ".avi", flag, vid_res=img_size)
        if not rst:
            return
        else:
            label = PURE_preprocess_label(path + "/" + str(path.split("/")[-1]) + ".json", video.shape[0])
            if label is None:
                return
            else:
                return_dict[path.split("/")[-1]] = {'video': video, 'label': label}

    del video, label, rst
    gc.collect()


if __name__ == '__main__':
    start_time = time.time()
    print("Hello")
    multiprocessing.set_start_method('forkserver')
    preprocessing()
    # filename = "DATASETS/cohface/1/0/data.hdf5"

    # with h5py.File(filename, "r") as f:
    #     # Print all root level object names (aka keys)
    #     # these can be group or dataset names
    #     print("Keys: %s" % f.keys())
        # get first object name/key; may or may NOT be a group
        # a_group_key = list(f.keys())[0]

        # get the object type for a_group_key: usually group or dataset
        # print(type(f[a_group_key]))

        # If a_group_key is a group name,
        # this gets the object names in the group and returns as a list
        # data = list(f[a_group_key])
        #
        # # If a_group_key is a dataset name,
        # # this gets the dataset values and returns as a list
        # data = list(f[a_group_key])
        # # preferred methods to get dataset values:
        # ds_obj = f[a_group_key]  # returns as a h5py dataset object
        # ds_arr = f[a_group_key][()]  # returns as a numpy array
