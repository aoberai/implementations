import cv2
import numpy as np

imgL = cv2.imread("l_cap.png",0)
imgR = cv2.imread("r_cap.png",0)
# TODO: this might be breaking things
# imgL = cv2.resize(imgL, size)
# imgR = cv2.resize(imgR, size)
size = (np.shape(imgL)[1], np.shape(imgL)[0])

# left_cam_file = cv2.FileStorage("left_cam.yml", cv2.FILE_STORAGE_READ)
# right_cam_file = cv2.FileStorage("right_cam.yml", cv2.FILE_STORAGE_READ)
cv_file = cv2.FileStorage("stereo_cam.yml", cv2.FILE_STORAGE_READ)

# TODO: I fixed by not using stereo rectify and using these matrices from calibration itself
K1 = cv_file.getNode("K1").mat()
D1 = cv_file.getNode("D1").mat()
K2 = cv_file.getNode("K2").mat()
D2 = cv_file.getNode("D2").mat()
R = cv_file.getNode("R").mat()
T = cv_file.getNode("T").mat()
E = cv_file.getNode("E").mat()
F = cv_file.getNode("F").mat()
R1 = cv_file.getNode("R1").mat()
R2 = cv_file.getNode("R2").mat()
P1 = cv_file.getNode("P1").mat()
P2 = cv_file.getNode("P2").mat()
Q = cv_file.getNode("Q").mat()

# R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(cameraMatrix1=K1, distCoeffs1=D1, cameraMatrix2=K2, distCoeffs2=D2, imageSize=size, R=R, T=T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1)

leftMapX, leftMapY = cv2.initUndistortRectifyMap(K1, D1, R1, P1, size, cv2.CV_32FC1)
left_rectified = cv2.remap(imgL, leftMapX, leftMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
rightMapX, rightMapY = cv2.initUndistortRectifyMap(K2, D2, R2, P2, size, cv2.CV_32FC1)
right_rectified = cv2.remap(imgR, rightMapX, rightMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

cv2.imwrite("rectified_1.png", left_rectified)
cv2.imwrite("rectified_2.png", right_rectified)

cv2.imshow("Left", left_rectified)
cv2.imshow("Right", right_rectified)
cv2.waitKey(0)


