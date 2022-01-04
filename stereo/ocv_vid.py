import cv2
import numpy as np

captureLeft = cv2.VideoCapture(0)
captureRight = cv2.VideoCapture(1)



fourccRight = cv2.VideoWriter_fourcc(*'mp4v')
fourccLeft = cv2.VideoWriter_fourcc(*'mp4v')
videoWriterRight = cv2.VideoWriter('./right.mp4', fourccRight, 30.0, (640, 480))
videoWriterLeft = cv2.VideoWriter('./left.mp4', fourccLeft, 30.0, (640, 480))

for i in range(300):

    ret1, left = captureLeft.read()
    ret2, right = captureRight.read()

    if ret1 and ret2:
        cv2.imshow('videoRight', right)
        cv2.imshow('videoLeft', left)
        videoWriterLeft.write(left)
        videoWriterRight.write(right)

    if (cv2.waitKey(1) & 0xFF) == ord('q'):  # Hit `q` to exit
        break

captureLeft.release()
captureRight.release()
videoWriterLeft.release()
videoWriterRight.release()
cv2.destroyAllWindows()
