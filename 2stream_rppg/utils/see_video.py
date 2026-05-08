import h5py
import math
import random
import numpy as np
import cv2

# see_what = "five_fold_size_72/Baseline_Size72/PURE_test_2.hdf5"


save_path1 = "/media/user/E62CA8D52CA8A1D3/pytorch_rppgs/Baseline/"            # Database faceROI=36*36
# save_path = "Baseline_FiveFold_Size72(Testing_Unseen_Rand)/"
# save_path = "Baseline_FiveFold_Size72(Testing_Unseen_Rand)/match_36/"       # Database faceROI=72*72
# save_path = "Baseline_FiveFold_Size72(Testing_Unseen)/"
# save_path = "Baseline_FiveFold(Testing_Unseen_Rand)/redo_mp/"        # Dataset size = 36*36 mediapipe method redo
# save_path = "Baseline_FiveFold(Testing_Unseen_Rand)/redo_cv/"
# save_path = "Baseline_FiveFold_Size72(Testing_Unseen_Rand)/match_36/"
save_path2 = "size_72/Baseline_Size72/"
#-- print dataset video number --#
# file = h5py.File(save_path + "PURE_test_5" + ".hdf5", "r")
# # save_path = "Baseline/"
# # save_path = "Baseline_redo_mp/"
# # file = h5py.File(save_path + "PURE" + ".hdf5", "r")
# Total_len = len(file.keys())
# print(f"Total_len={Total_len}")
# print(file.keys())


#-- show all of imageROI in dataset --#
with h5py.File(save_path1 + "PURE" + ".hdf5", "r") as file36:
    with h5py.File(save_path2 + "PURE" + ".hdf5", "r")as file72:
    # Assuming the dataset containing the image is named "image_data"
        for key in file36.keys():
            if key=='06-04':
                video_data36 = file36[key]['video']
                video_data72 = file72[key]['video']
                print(f"keys = {key}")
                print(f"len(image_data) = {len(video_data36)}")
                # Display the image using OpenCV
                # count = 0  # Counter to keep track of the number of images shown
                for (img1, img2) in zip(video_data36, video_data72):
                    cv2.imshow('Image36', cv2.resize(img1,(128,128)))
                    cv2.imshow('Image72', cv2.resize(img2,(256,256)))
                    key = cv2.waitKey(10) & 0xFF
                    if key == ord('p'):  # Wait for 100 milliseconds
                        cv2.waitKey(-1)  # 等待按下任意键后继续执行循环
            # count += 1
            
            # # Check if 10 images have been shown
            # if count % 1000 == 0:
            #     cv2.destroyAllWindows()