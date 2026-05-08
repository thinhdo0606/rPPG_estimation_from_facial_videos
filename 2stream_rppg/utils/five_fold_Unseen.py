import h5py
import math
import random
import numpy as np
k = 5
dataset_name = 'MANHOB_HCI'
# root_path = "Baseline/"
# save_path = "Baseline_FiveFold(Testing_Unseen_Rand)/"
root_path = "Baseline_Size72_BPF_0.67_4/"
save_path = "Baseline_FiveFold_Size72(Testing_Unseen_Rand)/"
file = h5py.File(root_path + dataset_name + ".hdf5", "r")
print(f"file.keys() = {file.keys()}")
print(f"Total_len={len(file.keys())}")
#------------------------------------------------------------------------------------#
#---PURE Dataset---#
# videos = list(file.keys())
# extracted_numbers = []
# extracted_numbers_set = []
# # ---Extract the first two numbers from each key and add them to the list---#
# for key in videos:
#     numbers = key.split('-')[0]
#     extracted_numbers.append(''.join(numbers))
# extracted_numbers_set = list(set(extracted_numbers))      # Random
# # extracted_numbers_set = sorted(set(extracted_numbers))  # Sequential
# print(f"extracted_numbers_set = {extracted_numbers_set}")
# segment = len(extracted_numbers_set) // k
# print(f"segment = {segment}")
# # Test
# # print("len(file.keys()) = {}".format(len(file.keys())))
# # print("file.keys() = {}".format(file.keys()))
# # print(f"Extracted_numbers={extracted_numbers}")
# # print(f"Extracted_numbers set={extracted_numbers_set}")
# unseen_videos = [[] for _ in range(k)]
# for index, i in enumerate(range(0, len(extracted_numbers_set), segment)):
#     for key in videos:
#         numbers = key.split('-')[0]
#         for j in range(0,segment):
#             if numbers == extracted_numbers_set[i+j]:
#                 unseen_videos[index].append(key)
#------------------------------------------------------------------------------------#
#---MMSE Dataset---#
# videos = list(file.keys())
# extracted_numbers = []
# extracted_numbers_set = []
# #---Extract the first two numbers from each key and add them to the list---#
# for key in videos:
#     numbers = key.split('_')[0]
#     extracted_numbers.append(''.join(numbers))
# extracted_numbers_set = list(set(extracted_numbers))        # Random
# # extracted_numbers_set = sorted(set(extracted_numbers))    # Sequential
# print(f"extracted_numbers_set = {extracted_numbers_set}")
# print(f"the length of extracted_numbers_set = {len(extracted_numbers_set)}")
# segment = round(len(extracted_numbers_set) / k)
# print(f"segment = {segment}")
# unseen_videos = [[] for _ in range(k)]
# for index, i in enumerate(range(0, len(extracted_numbers_set), segment)):
#     for key in videos:
#         numbers = key.split('_')[0]
#         for j in range(0,segment):
#             if i+j < len(extracted_numbers_set) and numbers == extracted_numbers_set[i+j]:
#                 unseen_videos[index].append(key)
#------------------------------------------------------------------------------------#
#---UBFC Dataset---#
# videos = list(file.keys())
# extracted_numbers = []
# extracted_numbers_set = []
# #---Extract the first two numbers from each key and add them to the list---#
# for key in videos:
#     numbers = key.split('ubject')[1]
#     extracted_numbers.append(''.join(numbers))
# extracted_numbers_set = list(set(extracted_numbers))        # Random
# # extracted_numbers_set = sorted(set(extracted_numbers))    # Sequential
# # print(f"extracted_numbers_set = {extracted_numbers_set}")
# print(f"the length of extracted_numbers_set = {len(extracted_numbers_set)}")
# segment = round(len(extracted_numbers_set) / k)
# print(f"segment = {segment}")
# unseen_videos = [[] for _ in range(k)]
# for index, i in enumerate(range(0, len(extracted_numbers_set), segment)):
#     for key in videos:
#         numbers = key.split('ubject')[1]
#         for j in range(0,segment):
#             if i+j < len(extracted_numbers_set) and numbers == extracted_numbers_set[i+j]:
#                 unseen_videos[index].append(key)
#------------------------------------------------------------------------------------#
#---MAHNOB_HCI Dataset---#
videos = list(file.keys())
random.shuffle(videos)
#---Extract the first two numbers from each key and add them to the list---#
# extracted_numbers_set = sorted(set(extracted_numbers))    # Sequential
# print(f"videos = {videos}")
print(f"the length of extracted_numbers_set = {len(videos)}")
segment = round(len(videos) / k)
print(f"segment = {segment}")
unseen_videos = [[] for _ in range(k)]
for index, i in enumerate(range(0, len(videos), segment)):
    for key in videos:
        for j in range(0,segment):
            if i+j < len(videos) and key == videos[i+j]:
                unseen_videos[index].append(key)


# print(f"Unseen videos={unseen_videos}")


for i in range(1, k+1):
    print('----------------------------------')
    print(f"FOLD: {i}")
    if i==k:
        test_ids = unseen_videos[i-1]
        train_ids = unseen_videos[0:(i-1)]
    else:
        test_ids = unseen_videos[i-1]
        train_ids = unseen_videos[0:i-1] + unseen_videos[i:]
    train_ids_reshaped = [item for sublist in train_ids for item in sublist]
    #------------------------------------------------------------------------------------#
    # Test
    print(len(train_ids), "Train_ids: ", train_ids)
    print(len(test_ids), "Test_ids: ", test_ids)
    #------------------------------------------------------------------------------------#
    # Writing the dataset
    # train_file = h5py.File(save_path + dataset_name + f"_train_{i}.hdf5", "w")
    # for data_path in train_ids_reshaped:
    #     file.copy(file[data_path], train_file, data_path)
    # train_file.close()

    # test_file = h5py.File(save_path + dataset_name + f"_test_{i}.hdf5", "w")
    # for data_path in test_ids:
    #     file.copy(file[data_path], test_file, data_path)
    # test_file.close()
    #------------------------------------------------------------------------------------#
file.close()