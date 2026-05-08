import cv2
from face_recognition import face_locations, face_landmarks
import dlib
from imutils import face_utils

p = "./utils/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)
img = cv2.imread("/home/dsp520/Documents/pytorch_rppgs/image/download.jpeg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


rects = detector(gray)
top = rects[0].top()
right = rects[0].right()
bottom = rects[0].bottom()
left = rects[0].left()

# for rect in rects:
#     x1 = rect.top()
#     y1 = rect.right()
#     x2 = rect.bottom()
#     y2 = rect.left()
#     print(x1, y1, x2, y2, end='  ')
# face_location = face_locations(img, model='cnn')
# top, right, bottom, left = face_location[0]
# print(top, right, bottom, left, end='  ')

# rects = detector(gray, 0)
# for (i, rect) in enumerate(rects):
#     shape = predictor(gray, rect)
#     shape = face_utils.shape_to_np(shape)

#     for (x,y) in shape:
#             cv2.circle(img, (x,y), 2, (0, 255, 0), -1)
# cv2.imshow("test", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()