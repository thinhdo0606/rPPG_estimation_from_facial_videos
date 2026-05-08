import h5py
import math
import random
import numpy as np
import cv2
from funcs2 import plot_graph, BPF_dict

# see_what = "five_fold_size_72/Baseline_Size72/PURE_test_2.hdf5"


# save_path1 = "/media/user/DATA/pytorch_rppgs_Hao/Baseline_FiveFold(Testing_Unseen_Rand)/"            # Database faceROI=36*36
# save_path = "Baseline_FiveFold_Size72(Testing_Unseen_Rand)/"
# save_path = "Baseline_FiveFold_Size72(Testing_Unseen_Rand)/match_36/"       # Database faceROI=72*72
# save_path = "Baseline_FiveFold_Size72(Testing_Unseen)/"
# save_path = "Baseline_FiveFold(Testing_Unseen_Rand)/redo_mp/"        # Dataset size = 36*36 mediapipe method redo
# save_path = "Baseline_FiveFold(Testing_Unseen_Rand)/redo_cv/"
# save_path = "Baseline_FiveFold_Size72(Testing_Unseen_Rand)/match_36/"
# save_path2 = "/media/user/DATA/pytorch_rppgs_Hao/redo_cv/"
save_path1 = "./Baseline/"
# save_path2 = "./Baseline_Size72/"
save_path2 = "./Baseline_Size72_matchDao/"
# save_path2 = "/media/user/9af50fc3-08ee-4b94-8c57-d5a7745c895d/MAHNOB_72_old_backup/"
#-- print dataset video number --#
# file = h5py.File(save_path + "PURE_test_5" + ".hdf5", "r")
# # save_path = "Baseline/"
# # save_path = "Baseline_redo_mp/"
# # file = h5py.File(save_path + "PURE" + ".hdf5", "r")
# Total_len = len(file.keys())
# print(f"Total_len={Total_len}")
# print(file.keys())


# #-- show all of imageROI in dataset --#
# MANHOB_HCI
# with h5py.File(save_path1 + "MANHOB_HCI" + ".hdf5", "r") as file_Dao:
#     with h5py.File(save_path2 + "MANHOB_HCI" + ".hdf5", "r")as file_redo:
#         # print(len(file_Dao),len(file_redo))
#     # Assuming the dataset containing the image is named "image_data"
#         c= 0
#         for key in file_redo.keys():
        #-- check the length of video, label or rppg signal figure --#
        #     if key=='3130':
        #     # if key=='01-01':
        #     # if key=='subject1':
        #         video_data1 = file_Dao[key]['video']
        #         video_data2 = file_redo[key]['video']
        #         print(f"keys = {key}")
        #         print(f"len(Dao_video_data) = {len(video_data1)}, len(redo_video_data) = {len(video_data2)}")
        #         if len(video_data1) != len(video_data2):
        #             print(f"the lens of rppg signal is different")
        #         label_data_Dao = file_Dao[key]['label']
        #         label_data_Hao = file_redo[key]['label']
        #         print(f"keys = {key}")
        #         print(f"len(Dao_lebel_data) = {len(label_data_Dao)}, len(redo_lebel_data) = {len(label_data_Hao)}")
        #         if len(label_data_Dao) != len(label_data_Hao):
        #             print(f"the lens of rppg signal is different")
        # # label_data_Dao = BPF_dict(label_data_Dao, 61)
        # # label_data_Hao = BPF_dict(label_data_Hao, 61)
        # plot_graph(0, 1830, label_data_Dao, label_data_Hao)
        #-- end --#
            
        #-- show face in dataset --#
            # video_data1 = file_Dao[key]['video']
            # video_data2 = file_redo[key]['video']
            # label_data1 = file_Dao[key]['label']
            # label_data2 = file_redo[key]['label']
            # print(f"keys = {key}")
            # plot_graph(0, 1830, label_data1, label_data2)
            # for (img1, img2) in zip(video_data1, video_data2):
            #     # diff_img = cv2.absdiff(img1, img2)
            #     # if diff_img.sum()!=0:
            #     img1 = cv2.resize(img1, (128,128))
            #     img2 = cv2.resize(img2, (128,128))
            #     final_frame = cv2.hconcat((img1, img2))
            #     cv2.imshow('Image', cv2.resize(final_frame,(256,128)))
            #     cv2.waitKey(1)
            # cv2.imwrite(f'/media/user/E62CA8D52CA8A1D3/pytorch_rppgs/forDebug/different image({c}).jpg',cv2.resize(final_frame,(256,128)))
            # c = c+1
                # print(f"different image({c})")
        #-- end --#
                

count = 0
with h5py.File(save_path1 + "MANHOB_HCI" + ".hdf5", "r") as file_Dao:
    with h5py.File(save_path2 + "MANHOB_HCI" + ".hdf5", "r") as file_redo:
        # Assuming the dataset containing the image is named "image_data"
        print("len=",len(file_redo.keys()))
        for key in file_redo.keys():
            label_data_Dao = file_Dao[key]['label']
            label_data_Hao = file_redo[key]['label']
            print(f"keys = {key}")
            print(f"len(Dao_lebel_data) = {len(label_data_Dao)}, len(redo_lebel_data) = {len(label_data_Hao)}")
            if len(label_data_Dao) != len(label_data_Hao):
                print(f"the lens of rppg signal is different")
            # Display the image using OpenCV
            sum = 0
            c2 = 0
            for i in range(len(label_data_Dao)):
                if label_data_Dao[i] != label_data_Hao[i]:
                    sum += abs(label_data_Dao[i]- label_data_Hao[i])
                    count += 1
                    c2+=1
            if c2 != 0:
                print(f"the different rppg signal is {c2}")
            plot_graph(0, 1830, label_data_Dao, label_data_Hao)
print(f"the different rppg signal is {count}")
print(sum/count)
