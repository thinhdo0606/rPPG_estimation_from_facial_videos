import h5py
import math
import random
k = 5
dataset_name = 'MANHOB_HCI'
root_path = "Baseline_Size72_matchDao/"
refer_path = "Baseline_FiveFold(Testing_Unseen_Rand)/"

save_path = "Baseline_FiveFold_Size72(Testing_Unseen_Rand)/match_Dao/"

file = h5py.File(root_path + dataset_name + ".hdf5", "r")
Total_len = len(file.keys())
print(f"Total_len={Total_len}")

# Test
# print(f"random_videos={random_videos}")
# print("segment = {}".format(segment))
# print("len(file.keys()) = {}".format(len(file.keys())))
# print("file.keys() = {}".format(file.keys()))

# train_length = int(len(file.keys()) * cv_ratio)

#---Datasets duplicate file keys from reference dataset---#
for i in range(1, k+1):
    print('----------------------------------')
    print(f"FOLD: {i}")

    # Training dataset list
    refer_dataset_path = refer_path + dataset_name + "_train_" + str(i) + ".hdf5"
    refer_file_tr = h5py.File(refer_dataset_path, 'r')
    train_ids = list(refer_file_tr.keys())
    print(f"train_ids = {train_ids}, len={len(train_ids)}")
    # Validation dataset list
    refer_dataset_path = refer_path + dataset_name + "_test_" + str(i) + ".hdf5"
    refer_file_val = h5py.File(refer_dataset_path, 'r')
    test_ids = list(refer_file_val.keys())
    print(f"test_ids = {test_ids}, len={len(test_ids)}")

    # Datasets random sampling
    # if i==k:
    #     train_ids = random_videos[0:(i-1)*segment]
    #     print(len(train_ids), "Train_ids: ", train_ids)
    #     test_ids = random_videos[(i-1)*segment:]
    #     print(len(test_ids), "Test_ids: ", test_ids)
    # else:
    #     train_ids = random_videos[0:(i-1)*segment] + random_videos[i*segment:]
    #     print(len(train_ids), "Train_ids: ", train_ids)
    #     test_ids = random_videos[(i-1)*segment: i*segment]
    #     print(len(test_ids), "Test_ids: ", test_ids)
    # Datasets sequencial sampling
    # if i==k:
    #     train_ids = list(file.keys())[0:(i-1)*segment]
    #     print(len(train_ids), "Train_ids: ", train_ids)
    #     test_ids = list(file.keys())[(i-1)*segment:]
    #     print(len(test_ids), "Test_ids: ", test_ids)
    # else:
    #     train_ids = list(file.keys())[0:(i-1)*segment] + list(file.keys())[i*segment:]
    #     print(len(train_ids), "Train_ids: ", train_ids)
    #     test_ids = list(file.keys())[(i-1)*segment: i*segment]
    #     print(len(test_ids), "Test_ids: ", test_ids)

    # Writting data
    #------------------------------------------------------------------------------------#
    train_file = h5py.File(save_path + dataset_name + f"_train_{i}.hdf5", "w")
    for data_path in train_ids:
        try:
            file.copy(file[data_path], train_file, data_path)
        except:
            pass
    train_file.close()

    test_file = h5py.File(save_path + dataset_name + f"_test_{i}.hdf5", "w")
    for data_path in test_ids:
        try:
            file.copy(file[data_path], test_file, data_path)
        except:
            pass
    test_file.close()
    #------------------------------------------------------------------------------------#
file.close()


# for i in range(1, k+1):
#     print('----------------------------------')
#     print(f"FOLD: {i}")
#     if Total_len - (i-1)*segment < segment:
#         print("F1")
#         Interp = (i)*segment - Total_len
#         train_ids = list(file.keys())[0:(i-1)*segment-Interp]
#         print(len(train_ids), "Train_ids: ", train_ids)
#         test_ids = list(file.keys())[(i-1)*segment-Interp:]
#         print(len(test_ids), "Test_ids: ", test_ids)
#     else:
#         train_ids = list(file.keys())[0:(i-1)*segment] + list(file.keys())[i*segment:]
#         print(len(train_ids), "Train_ids: ", train_ids)
#         test_ids = list(file.keys())[(i-1)*segment: i*segment]
#         print(len(test_ids), "Test_ids: ", test_ids)
#     train_file = h5py.File(save_path + dataset_name + f"_train_{i}.hdf5", "w")
    
#     for data_path in train_ids:
#         file.copy(file[data_path], train_file, data_path)
#     train_file.close()

#     test_file = h5py.File(save_path + dataset_name + f"_test_{i}.hdf5", "w")
#     for data_path in test_ids:
#         file.copy(file[data_path], test_file, data_path)
#     test_file.close()
# file.close()