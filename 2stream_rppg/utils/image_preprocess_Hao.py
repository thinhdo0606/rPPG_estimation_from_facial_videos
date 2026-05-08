import os

import cv2
import numpy as np
from tqdm import tqdm
from face_recognition import face_locations, face_landmarks
# import mediapipe as mp
import gc
import matplotlib.pyplot as plt
import dlib
from imutils import face_utils


def preprocess_Video_RGB_only_Hao(path, flag, vid_res): # vis_res: image size

    # cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    cap = cv2.VideoCapture(path)
    frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # cap.get() -> get the total frame of video

    raw_video = np.empty((frame_total, vid_res, vid_res, 3), dtype=np.uint8)  # empty array for all the frame

    # Hao
    tracker = cv2.TrackerMIL_create()
    
    j = 0
    # print("frame: ", frame_total)
    # print("open: ", cap.isOpened())
    with tqdm(total=frame_total, position=0, leave=True, desc=path) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            # frame_height, frame_width = frame.shape[:2]
            # frame = cv2.resize(frame, [frame_width//2, frame_height//2])

            if frame is None:
                return False, None
            if flag: # detect face

                # Dao
                # print("Hello workd")
                rst, bbox, resized_frame = faceDetection(frame)
                # print("crop: ", crop_frame)
                # Hao
                # bbox = cv2.selectROI("Select ROI",frame, False)
                ret = tracker.init(frame, bbox)
                ret, bbox = tracker.update(frame)
    
                crop_frame = resized_frame[max(0, bbox[1] - 10):min(resized_frame.shape[0], bbox[3] + 10),
                              max(0, bbox[0] - 10):min(resized_frame.shape[1], bbox[2] + 10)]
                # reference
                # crop_frame =  frame[
                #     max(0, bbox[0] - 10):min(frame.shape[0], bbox[1] + 10),
                #     max(0, bbox[2] - 10):min(frame.shape[1], bbox[3] + 10)]

                if not rst:
                    return False, None
                
            # resize to a square
            crop_frame = cv2.resize(crop_frame, dsize=(vid_res, vid_res), interpolation=cv2.INTER_AREA)
            raw_video[j] = crop_frame
            j += 1
            if j == frame_total:
                break
            pbar.update(1)
        cap.release()
    pbar.close()
    del pbar, j, flag, rst, ret, frame, frame_total
    gc.collect()

    if np.isnan(raw_video).any():
        print('Nan value detected')
        del raw_video
        return False, None
    if np.isinf(raw_video).any():
        print('Infinite value detected')
        del raw_video
        return False, None

    return True, raw_video



# input frame and found the bounding box of face and output the resized image and True
def faceDetection(frame):
    global locat


    p = "./utils/shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)

    resized_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    resized_frame_gray = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2GRAY)
    # print('resize: ', resized_frame.shape)
    # Dao
    face_location = face_locations(resized_frame, model='cnn')
    # Hao
    # face_location = detector(resized_frame_gray)

    if len(face_location) == 0:  # cant detect face
        print('cant detect face')
        return False, None
        # if len(locat[0]) != 4:  # 기존 frame
        #     # dst = resized_frame[resized_frame.shape[0] // 4: resized_frame.shape[0] // 4 * 3,
        #     #       resized_frame.shape[1] // 4:resized_frame.shape[1] // 4 * 3]
        # else:
        #     top, right, bottom, left = locat[0]
        #     dst = resized_frame[max(0, top - 10):min(resized_frame.shape[0], bottom + 10),
        #           max(0, left - 10):min(resized_frame.shape[1], right + 10)]
        # return False, dst
        #     return True, dst

    #  face bounding box
    top, right, bottom, left = face_location[0]

    # Hao
    # top = face_location[0].top()
    # right = face_location[0].right()
    # bottom = face_location[0].bottom()
    # left = face_location[0].left()
    bbox_coor = (left, top, right, bottom)
    # print(face_location[0])

    # Dao
    # dst = resized_frame[max(0, top - 10):min(resized_frame.shape[0], bottom + 10),
    #       max(0, left - 10):min(resized_frame.shape[1], right + 10)]
    
    # plt.imshow (frame)
    # # plt.show()
    # # plt.imshow(dst)
    # plt.show()
    locat = face_location
    # Dao
    # return True, bbox_coor, dst
    # Hao
    return True, bbox_coor, resized_frame





# img = cv2.imread("/home/dsp520/Documents/pytorch_rppgs/image/download.jpeg")
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



# rects = detector(gray, 0)
# for (i, rect) in enumerate(rects):
#     shape = predictor(gray, rect)
#     shape = face_utils.shape_to_np(shape)

#     for (x,y) in shape:
#             cv2.circle(img, (x,y), 2, (0, 255, 0), -1)
# cv2.imshow("test", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()