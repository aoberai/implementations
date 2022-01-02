import cv2
import numpy as np

imgL = cv2.imread("l_cap.png",0)
imgR = cv2.imread("r_cap.png",0)
size = (480, 360)
imgL = cv2.resize(imgL, size)
imgR = cv2.resize(imgR, size)

left_cam_file = cv2.FileStorage("left_cam.yml", cv2.FILE_STORAGE_READ)
right_cam_file = cv2.FileStorage("right_cam.yml", cv2.FILE_STORAGE_READ)
stereo_file = cv2.FileStorage("stereo_cam.yml", cv2.FILE_STORAGE_READ)

K1 = left_cam_file.getNode("K").mat()
D1 = left_cam_file.getNode("D").mat()

K2 = right_cam_file.getNode("K").mat()
D2 = right_cam_file.getNode("D").mat()

R = stereo_file.getNode("R").mat()
# R = np.eye(3, 3)
print(R)
T = stereo_file.getNode("T").mat()

R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(cameraMatrix1=K1, distCoeffs1=D1, cameraMatrix2=K2, distCoeffs2=D2, imageSize=size, R=R, T=T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=1)

leftMapX, leftMapY = cv2.initUndistortRectifyMap(K1, D1, R1, P1, size, cv2.CV_32FC1)
left_rectified = cv2.remap(imgL, leftMapX, leftMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
rightMapX, rightMapY = cv2.initUndistortRectifyMap(K2, D2, R2, P2, size, cv2.CV_32FC1)
right_rectified = cv2.remap(imgR, rightMapX, rightMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

cv2.imwrite("rectified_1.png", left_rectified)
cv2.imwrite("rectified_2.png", right_rectified)

cv2.imshow("Left", left_rectified)
cv2.imshow("Right", right_rectified)
cv2.waitKey(0)


