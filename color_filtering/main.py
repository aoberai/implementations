# Same thing as everything else except in python

import cv2
import numpy as np

window_name = "Color Filter"
cap = cv2.VideoCapture("../../../python/blind-navigation/TurnSidewalk.mp4")  # replace this with video feed

lower_bound = [0, 0, 0]
upper_bound = [0, 0, 0]
blur = 5

def lH_change(val):
    print("Changing")
    lower_bound[0] = val


def lS_change(val):
    print("Changing")
    lower_bound[1] = val


def lV_change(val):
    print("Changing")
    lower_bound[2] = val


def hH_change(val):
    print("Changing")
    upper_bound[0] = val


def hS_change(val):
    print("Changing")
    upper_bound[1] = val


def hV_change(val):
    print("Changing")
    upper_bound[2] = val

def blur_change(val):
    global blur
    print("Changing")
    blur = val;

cv2.imshow(window_name, cap.read()[1])

cv2.createTrackbar('lH', window_name, 0, 255, lH_change)
cv2.createTrackbar('lS', window_name, 0, 255, lS_change)
cv2.createTrackbar('lV', window_name, 0, 255, lV_change)

cv2.createTrackbar('hH', window_name, 0, 255, hH_change)
cv2.createTrackbar('hS', window_name, 0, 255, hS_change)
cv2.createTrackbar('hV', window_name, 0, 255, hV_change)

cv2.createTrackbar('blur', window_name, 0, 25, blur_change)

while(1):
    _, frame = cap.read()
    if blur != 0:
        frame = cv2.blur(frame, (blur, blur,))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower_bound), np.array(upper_bound))
    res = cv2.bitwise_and(frame, frame, mask=mask)

    # cv2.imshow('frame',frame)
    # cv2.imshow('mask',mask)
    cv2.imshow(window_name, res)

    k = cv2.waitKey(5) & 0xFF

    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
