import numpy as np
import cv2

# Reading the left and right images.

imgL = cv2.imread("rectified_1.png",0)
imgR = cv2.imread("rectified_2.png",0)

# Setting parameters for StereoSGBM algorithm
minDisparity = -1;
numDisparities = 5*16;
blockSize = 3;
disp12MaxDiff = 12;
uniquenessRatio = 10;
speckleWindowSize = 50;
speckleRange = 32;
preFilterCap=63
window_size = 3

# Creating an object of StereoSGBM algorithm
stereo = cv2.StereoSGBM_create(minDisparity = minDisparity,
                               numDisparities = numDisparities,
                               blockSize = blockSize,
                               P1=8 * 3 * window_size,
                               # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
                               P2=32 * 3 * window_size,
                               disp12MaxDiff = disp12MaxDiff,
                               uniquenessRatio = uniquenessRatio,
                               speckleWindowSize = speckleWindowSize,
                               speckleRange = speckleRange,
                               preFilterCap = preFilterCap,
                               mode = cv2.STEREO_SGBM_MODE_SGBM_3WAY
                               )

# Calculating disparith using the StereoSGBM algorithm
disp = stereo.compute(imgL, imgR).astype(np.float32)
disp = cv2.normalize(disp,0,255,cv2.NORM_MINMAX)

# Displaying the disparity map
cv2.imshow("disparity", cv2.resize(disp, (480, 360)))
cv2.waitKey(0)
