import cv2
import numpy as np
import os

l_cap = cv2.VideoCapture(0)
r_cap = cv2.VideoCapture(1)
calib_imgs = os.listdir("calib_imgs")
try:
    counter = max([int(i.replace("r_cap", "").replace("l_cap", "").replace(".png", "")) for i in calib_imgs])
except Exception as e:
    counter = 0

print("Starting at " + str(counter))
while True:
    l_img = l_cap.read()[1]
    r_img = r_cap.read()[1]
    cv2.imshow("stereo_img", cv2.hconcat([l_img, r_img]))
    if cv2.waitKey(25) & 0xFF == ord('s'):
        print("Saving {}".format(counter))
        cv2.imwrite("calib_imgs/l_cap" + str(counter) + ".png", l_cap.read()[1])
        cv2.imwrite("calib_imgs/r_cap" + str(counter) + ".png", r_cap.read()[1])
        counter += 1

