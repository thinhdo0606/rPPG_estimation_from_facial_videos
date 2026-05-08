import h5py
k = 5
dataset_name = 'PURE'
# root_path = "Face_Tracking_Test/"
root_path = "Face_Tracking_Baseline/"
save_path = "Face_Tracking_Test/Five_Fold/"
file = h5py.File(root_path + dataset_name + ".hdf5", "r")
segment = len(file.keys()) // k

# Test
# print("segment = {}".format(segment))
# print("len(file.keys()) = {}".format(len(file.keys())))
# print("file.keys() = {}".format(file.keys()))

# train_length = int(len(file.keys()) * cv_ratio)
for i in range(1, k+1):
    print('----------------------------------')
    print(f"FOLD: {i}")
    train_ids = list(file.keys())[0:(i-1)*segment] + list(file.keys())[i*segment:]
    print(len(train_ids), "Train_ids: ", train_ids)
    test_ids = list(file.keys())[(i-1)*segment: i*segment]
    print(len(test_ids), "Test_ids: ", test_ids)

    train_file = h5py.File(save_path + dataset_name + f"_train_{i}.hdf5", "w")
    
    for data_path in train_ids:
        file.copy(file[data_path], train_file, data_path)
    train_file.close()

    test_file = h5py.File(save_path + dataset_name + f"_test_{i}.hdf5", "w")
    for data_path in test_ids:
        file.copy(file[data_path], test_file, data_path)
    test_file.close()
file.close()