import h5py
import math
import random
import numpy as np
import cv2
from eval_metrics_Hao import *

# save_path = "Baseline_FiveFold(Testing_Unseen_Rand)/"            # Database faceROI=36*36
# save_path = "Baseline_FiveFold_Size72(Testing_Unseen_Rand)/"
# save_path = "Baseline_FiveFold_Size72(Testing_Unseen_Rand)/match_36/"       # Database faceROI=72*72
# save_path = "Baseline_FiveFold_Size72(Testing_Unseen)/"
# save_path = "Baseline_FiveFold(Testing_Unseen_Rand)/redo_mp/"        # Dataset size = 36*36 mediapipe method redo
# save_path = "Baseline_FiveFold(Testing_Unseen_Rand)/redo_cv/"
# save_path = "Baseline_FiveFold_Size72(Testing_Unseen_Rand)/match_36/"
# save_path = "Baseline_FiveFold(Testing_Unseen_Rand)/redo_cv_v2/"    # redo PURE version2 dataset

save_path1 = "Baseline/"               # Original Dao dataset
# save_path1 = "Baseline_redo_cv/"       # redo PURE dataset
# save_path1 = "Baseline_redo_cv_v2/"    # redo PURE version2 dataset
# save_path2 = "Baseline_redo_cv_v3/"       # redo PURE version3 dataset
# save_path2 = "Baseline_redo_cv_v4/"       # redo PURE version3 dataset
# save_path2 = "Baseline_redo_cv_v5/"       # redo PURE version3 dataset
save_path2 = "Baseline_Size72/"

#-- print dataset video number --#
# save_path_v = "Baseline_FiveFold_Size72(Testing_Unseen_Rand)/"
# file = h5py.File(save_path_v + "PURE_test_1" + ".hdf5", "r")
# # save_path = "Baseline/"
# # save_path = "Baseline_redo_mp/"
# # file = h5py.File(save_path + "PURE" + ".hdf5", "r")
# Total_len = len(file.keys())
# print(f"Total_len={Total_len}")
# print(file.keys())

#-- show all of imageROI in dataset --#
# with h5py.File(save_path + "PURE_test_1" + ".hdf5", "r") as file:
#     # Assuming the dataset containing the image is named "image_data"
#     for key in file.keys():
#         video_data = file[key]['video']
#         print(f"keys = {key}")
#         print(f"len(image_data) = {len(video_data)}")
#         # Display the image using OpenCV
#         for img in video_data:
#             cv2.imshow('Image', img)
#             cv2.waitKey(10)  # Wait for 100 milliseconds

#-- show all of GT RPPG signal in dataset --#
# count = 0
# loss=0
# with h5py.File(save_path1 + "MANHOB_HCI" + ".hdf5", "r") as file_Dao:
#     with h5py.File(save_path2 + "MANHOB_HCI" + ".hdf5", "r") as file_Hao:
#         # Assuming the dataset containing the image is named "image_data"
#         for key in file_Dao.keys():
#             try:
#                 label_data_Dao = file_Dao[key]['label']
#                 label_data_Hao = file_Hao[key]['label']
#                 print(f"keys = {key}")
#                 print(f"len(lebel_data) = {len(label_data_Dao)}")
#                 # Display the image using OpenCV
#                 sum = 0
#                 c2 = 0
#                 for i in range(len(label_data_Dao)):
#                     if label_data_Dao[i] != label_data_Hao[i]:
#                         sum += abs(label_data_Dao[i]- label_data_Hao[i])
#                         count += 1
#                         c2+=1
#                 if c2 != 0:
#                     print(f"the different rppg signal is {c2}")
#             except:
#                 loss+=1
# print(f"the different rppg signal is {count}")
# print(f"the loss rppg signal is {loss}")
# print(sum/count)

#-- compare all of img between two database --#
# with h5py.File(save_path1 + "PURE" + ".hdf5", "r") as file_Dao:
#     with h5py.File(save_path2 + "PURE" + ".hdf5", "r")as file_redo:
#     # Assuming the dataset containing the image is named "image_data"
#         c= 0
#         for key in file_redo.keys():
#             # if key=='06-04':
#             video_data1 = file_Dao[key]['video']
#             video_data2 = file_redo[key]['video']
#             print(f"keys = {key}")
#             print(f"len(image_data) = {len(video_data1)}")
#             # Display the image using OpenCV
#             count = 0  # Counter to keep track of the number of images shown
            
#             # n = 0
#             for (img1, img2) in zip(video_data1, video_data2):
#                 img2 = cv2.resize(img2, (36, 36), interpolation = cv2.INTER_LINEAR)
#                 diff_img = cv2.absdiff(img1, img2)
#                 if diff_img.sum()!=0:
#                     count += 1
#                     # final_frame = cv2.hconcat((img1, img2))
#                     # cv2.imshow('Image', cv2.resize(final_frame,(256,128)))
#                     # cv2.imwrite(f'Debug/different image({c}-{0}.jpg',cv2.resize(final_frame,(256,128)))
#                     # c = c+1
#                     # n += 3
#                     # print(f"different image({c})")
#                 # if n > 0:
#                     # check_frame = cv2.hconcat((img1, img2))
#                     # cv2.imwrite(f'Debug/different image({c}-{3-n}.jpg',cv2.resize(check_frame,(256,128)))
#                     # n -= 1
#             print(f'the number of different frame is {count}')
#             c += count
#         print(c)
