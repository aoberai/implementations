import numpy as np
import cv2

# Reading the left and right images.

imgL = cv2.imread("rectified_1.png",0)
imgR = cv2.imread("rectified_2.png",0)

# Setting parameters for StereoSGBM algorithm
minDisparity = 0;
numDisparities = 64;
blockSize = 8;
disp12MaxDiff = 1;
uniquenessRatio = 10;
speckleWindowSize = 10;
speckleRange = 8;

# Creating an object of StereoSGBM algorithm
stereo = cv2.StereoSGBM_create(minDisparity = minDisparity,
                                       numDisparities = numDisparities,
                                       blockSize = blockSize,
                                       disp12MaxDiff = disp12MaxDiff,
                                       uniquenessRatio = uniquenessRatio,
                                       speckleWindowSize = speckleWindowSize,
                                       speckleRange = speckleRange
                                   )

# Calculating disparith using the StereoSGBM algorithm
disp = stereo.compute(imgL, imgR).astype(np.float32)
disp = cv2.normalize(disp,0,255,cv2.NORM_MINMAX)

# Displaying the disparity map
cv2.imshow("disparity",disp)
cv2.waitKey(0)
