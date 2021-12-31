import cv2
import numpy as np

l_cap = cv2.VideoCapture(0)
r_cap = cv2.VideoCapture(1)
while True:
    l_img = l_cap.read()[1]
    r_img = r_cap.read()[1]
    cv2.imshow("stereo_img", cv2.hconcat([l_img, r_img]))
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.imwrite("l_cap.png", l_cap.read()[1])
        cv2.imwrite("r_cap.png", r_cap.read()[1])
        cv2.destroyAllWindows()
        break

