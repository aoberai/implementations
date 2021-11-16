'''
Sources:
https://muthu.co/harris-corner-detector-implementation-in-python/
feature://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_features_harris/py_features_harris.html
https://sbme-tutorials.github.io/2018/cv/notes/6_week6.html
'''

import cv2
import numpy as np
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
import time
img_size = (480, 360)
# orig_img = cv2.resize(cv2.imread("dogo.jpg"), tuple(reversed(img_size)))
orig_img = cv2.resize(cv2.imread("grid.jpg"), tuple(reversed(img_size)))
# grayscale
gray_img = np.zeros(img_size)
for i in range(len(orig_img)):
    for j in range(len(orig_img[i])):
        gray_img[i][j] = (orig_img[i][j][0] * 0.114 + orig_img[i][j][1] * 0.587 + orig_img[i][j][2] * 0.299)/255 # tis how cv2 does
# apply gaussian filter (blur)
blur_kernel = (5, 5)
img = cv2.GaussianBlur(gray_img, blur_kernel, 0)
# apply sobel operator for x, y gradient
sobel_x_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y_kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
gradient = np.array([convolve(img, sobel_x_kernel), convolve(img, sobel_y_kernel)])
edges = np.sqrt(np.square(gradient[0]) + np.square(gradient[1]))
hessian = np.array([[convolve(gradient[0], sobel_x_kernel), convolve(gradient[1], sobel_x_kernel)],
                   [convolve(gradient[0], sobel_y_kernel), convolve(gradient[1], sobel_y_kernel)]])

# harris = det(H) - a x trace(H)
hessian_det = hessian[0][0]*hessian[1][1] - hessian[0][1]*hessian[1][0]
hessian_trace = hessian[0][0] + hessian[1][1]
a = 0.1 # sensitivity factor to separate corners from edges; larger -> less found
harris = hessian_det - a * np.square(hessian_trace) # TODO: research derivation more

# edge: harris (r) < 0; corner: r > 0; flat: r = 0

# only for perfect image
features = np.zeros(tuple(np.shape(harris)[0:2]) + (3,))
for i in range(0, len(harris)):
    for j in range(0, len(harris[i])):
        if harris[i][j] > 0: # corner
            features[i][j] = [harris[i][j]*255]*3
        elif harris[i][j] < 0: # edge
            features[i][j] = [0, 0, 0]

harris_maxs = np.zeros(np.shape(harris))
# O(N^2) rel min 
for i in range(1, len(harris) - 1):
    for j in range(1, len(harris[i]) - 1):
        harris_maxs[i][j] = 1 if harris[i+1][j] <= harris[i][j] and harris[i-1][j] <= harris[i][j] and harris[i][j+1] <= harris[i][j] and harris[i][j-1] <= harris[i][j] and harris[i][j] > 0 else 0
        # harris_maxs[i][j] = 0
# Dilate (for da eyes)
dilate_kernel = np.ones((3, 3), dtype="uint8")
fin_pts = harris_maxs.copy()
iterations = 1
for i in range(iterations):
    fin_pts = convolve(fin_pts, dilate_kernel)

'''
plt.imshow(255*harris, cmap='Greys', interpolation='nearest')
plt.show()
time.sleep(5)
plt.close()
'''
while True:
    cv2.imshow("win", fin_pts)
    cv2.waitKey(1)

