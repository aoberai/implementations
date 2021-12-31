import numpy as np
import cv2
from matplotlib import pyplot as plt

l_img = cv2.imread('l_cap.png', cv2.IMREAD_GRAYSCALE)
r_img = cv2.imread('r_cap.png', cv2.IMREAD_GRAYSCALE)
# cv2.imshow("stereo_img", cv2.hconcat([l_img, r_img]))
# cv2.waitKey(0)


# Initiate ORB detector
sift = cv2.SIFT_create()
# find the keypoints and descriptors with ORB
kp1, des1 = sift.detectAndCompute(l_img,None)
kp2, des2 = sift.detectAndCompute(r_img,None)



# Match keypoints in both images
# Based on: https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# Keep good matches: calculate distinctive image features
# Lowe, D.G. Distinctive Image Features from Scale-Invariant Keypoints. International Journal of Computer Vision 60, 91–110 (2004). https://doi.org/10.1023/B:VISI.0000029664.99615.94
# https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
matchesMask = [[0, 0] for i in range(len(matches))]
good = []
pts1 = []
pts2 = []

for i, (m, n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        # Keep this keypoint pair
        matchesMask[i] = [1, 0]
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)


# Draw the keypoint matches between both pictures
# Still based on: https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
draw_params = dict(matchColor=(0, 255, 0),
                                      singlePointColor=(255, 0, 0),
                                      matchesMask=matchesMask[300:500],
                                      flags=cv2.DrawMatchesFlags_DEFAULT)

keypoint_matches = cv2.drawMatchesKnn(
        l_img, kp1, r_img, kp2, matches[300:500], None, **draw_params)
cv2.imshow("Keypoint matches", keypoint_matches)
cv2.waitKey(0)

# # create BFMatcher object
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# # Match descriptors.
# matches = bf.match(des1,des2)
# # Sort them in the order of their distance.
# matches = sorted(matches, key = lambda x:x.distance)
# # Draw first 10 matches.
# img3 = cv2.drawMatches(l_img,kp1, r_img,kp2,matches[:30],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# plt.imshow(img3),plt.show()
#

# stereo = cv2.StereoSGBM_create(numDisparities=16, blockSize=15)
# disparity = stereo.compute(l_img,r_img)
# plt.imshow(disparity,'gray')
# plt.show()
