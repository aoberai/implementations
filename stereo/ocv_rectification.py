import cv2
import numpy as np

imgL = cv2.imread("l_img.png",0)
imgR = cv2.imread("r_img.png",0)

cv_file = cv2.FileStorage("calibration.yml", cv2.FILE_STORAGE_READ)

camera_matrix = cv_file.getNode("K").mat()
dist_vector = cv_file.getNode("D").mat()

R = cv2.eye(shape=(3, 3))

cv2.stereoRectify(cameraMatrix1=camera_matrix, distCoeffs1=dist_vector, cameraMatrix2=camera_matrix, distCoeffs2=dist_vector, imageSize=(np.shape(imgL)[1], np.shape(imgL)[0]), R, T, alpha=1)
