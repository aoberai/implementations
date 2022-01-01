import numpy as np
import cv2

# Reading the left and right images.

imgL = cv2.imread("rectified_1.png",0)
imgR = cv2.imread("rectified_2.png",0)

# Setting parameters for StereoSGBM algorithm
minDisparity = -64;
numDisparities = 192;
blockSize = 8;
disp12MaxDiff = 10;
uniquenessRatio = 1;
speckleWindowSize = 150;
speckleRange = 2;

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
cv2.imshow("disparity",cv2.blur(cv2.blur(disp, (13, 13)), (13, 13)))
cv2.waitKey(0)
