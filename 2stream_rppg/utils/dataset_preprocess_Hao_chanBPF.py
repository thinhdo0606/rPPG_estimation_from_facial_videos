import gc
import multiprocessing
import os
import h5py
import random
import time
import datetime
import natsort
from log import log_info_time
from text_preprocess import *

img_root_path: str = "Baseline_Size72_BPF_0.67_2/"
dataset_name: str = "MANHOB_HCI"
file_Img = h5py.File(img_root_path + dataset_name + ".hdf5", "r")
data_list = file_Img.keys()
data_list = list(data_list)

def preprocessing(init: bool = True,
                  dataset_root_path: str = "DATASETS/",
                  save_root_path: str = "Baseline_Size72_BPF_0.67_2_norm/"):

    manager = multiprocessing.Manager() #Multiprocessing Manager provides a way of creating centralized Python objects that can be shared safely among processes.

    threads = 6
    for i in np.arange(0, len(data_list), threads):
        if i + threads > len(data_list):
            threads = len(data_list) - i
        process = []
        return_dict = manager.dict()
        for data_path in data_list[i:i+threads]:  # 6 threads deal with each video in data_list(shuffle) for accelerating computation
            proc = multiprocessing.Process(target=preprocess_dataset, args=(dataset_root_path + dataset_name + "/" + data_path, True,
                                                                            data_path, dataset_name, return_dict))
            process.append(proc)
            proc.start()
        for proc in process:
            proc.join()
            proc.terminate()

        file = h5py.File(save_root_path + dataset_name + ".hdf5", "a")
        # print(return_dict)
        for data_path in return_dict.keys():
            dset = file.create_group(data_path)
            video_data = return_dict[data_path]['video']
            label_data = return_dict[data_path]['label']
            video_shape = video_data.shape
            label_shape = label_data.shape
            dset.create_dataset('video', video_shape, np.uint8, video_data, chunks=video_shape)
            dset.create_dataset('label', label_shape, np.float32, label_data, chunks=label_shape)
            # dset['video'] = return_dict[data_path]['video']
            # dset['label'] = return_dict[data_path]['label']
        # for data_path in return_dict.keys():
        #   input_vid = return_dict[data_path]['video']
            # nofs = input_vid.shape[0]
        #     label = return_dict[data_path]['label'].reshape(-1, 1)
        #
        #     if init:
        #         file.create_dataset('video', data=input_vid, shape=(nofs, img_size, img_size, chunk_shape[3]),
        #                             dtype=np.uint8, chunks=chunk_shape, maxshape=(None, img_size, img_size, chunk_shape[3]))
        #         file.create_dataset('label', data=label, shape=(nofs, 1), dtype=np.float32, chunks=(
        #             chunk_shape[0], 1), maxshape=(None, 1))
        #         init = False
        #     else:
        #         file['video'].resize((file['video'].shape[0] + nofs), axis=0)
        #         file['video'][-nofs:] = input_vid
        #         file['label'].resize((file['label'].shape[0] + nofs), axis=0)
        #         file['label'][-nofs:] = label
        file.close()
        #
        del process, return_dict
        gc.collect()

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


def preprocess_dataset(path, flag, data_path, dataset_name, return_dict):

    video = file_Img[data_path]["video"]
    if dataset_name == 'MANHOB_HCI':
        label = MTTS_MANHOB_Label(path + "/ground_truth.txt", video.shape[0])
        if label is None:
            return
        else:
            video_frames = np.array(video)
            return_dict[path.split("/")[-1]] = {'video': video_frames, 'label': label}

    del video, label
    gc.collect()


if __name__ == '__main__':
    start_time = time.time()
    print("Hello")
    multiprocessing.set_start_method('forkserver')
    preprocessing()
    # print(f"the length of data_list:{len(data_list)}")
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
