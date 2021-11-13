# https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html


import cv2
import numpy as np

cap = cv2.VideoCapture("DrivingFootage1.mp4")
size = (480, 360)

ret, frame1 = cap.read()
frame1 = cv2.resize(frame1, size)
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

while(1):
    ret, frame2 = cap.read()
    next_frame = cv2.cvtColor(cv2.resize(frame2, size), cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        prvs, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    cv2.imshow('Dense Optical Flow', rgb)
    cv2.waitKey(1)
    prvs = next_frame

cap.release()
cv2.destroyAllWindows()
