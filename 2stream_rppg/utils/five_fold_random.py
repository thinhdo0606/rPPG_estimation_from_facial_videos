import h5py
import math
import random
k = 5
dataset_name = 'PURE'
root_path = "Baseline/"
# root_path = "Baseline_Size96/"
save_path = "Baseline_FiveFold_NewSplit(Random)_test/"
# save_path = "Baseline_FiveFold_Size96(Random)/"

# root_path = "Face_Tracking_Baseline_Size72/"
# save_path = "Face_Tracking_FiveFold_Size72(Random)/"
file = h5py.File(root_path + dataset_name + ".hdf5", "r")
Total_len = len(file.keys())
print(f"Total_len={Total_len}")

random_videos = list(file.keys())
random.shuffle(random_videos)
segment = len(file.keys()) // k + 1

for i in range(1, k+1):
    print('----------------------------------')
    print(f"FOLD: {i}")
    # Datasets random sampling
    if i==k:
        train_ids = random_videos[0:(i-1)*segment]
        print(len(train_ids), "Train_ids: ", train_ids)
        test_ids = random_videos[(i-1)*segment:]
        print(len(test_ids), "Test_ids: ", test_ids)
    else:
        train_ids = random_videos[0:(i-1)*segment] + random_videos[i*segment:]
        print(len(train_ids), "Train_ids: ", train_ids)
        test_ids = random_videos[(i-1)*segment: i*segment]
        print(len(test_ids), "Test_ids: ", test_ids)

    # train_file = h5py.File(save_path + dataset_name + f"_train_{i}.hdf5", "w")
    # for data_path in train_ids:
    #     file.copy(file[data_path], train_file, data_path)
    # train_file.close()

    # test_file = h5py.File(save_path + dataset_name + f"_test_{i}.hdf5", "w")
    # for data_path in test_ids:
    #     file.copy(file[data_path], test_file, data_path)
    # test_file.close()
file.close()