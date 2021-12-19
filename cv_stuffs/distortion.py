import cv2
import numpy as np

img = cv2.imread("assets/dogo.jpg")
img_size = (480, 480)
# orig_img = cv2.resize(cv2.imread("dogo.jpg"), tuple(reversed(img_size)))
img = cv2.resize(img, tuple(reversed(img_size)))

tmp_img = np.zeros(img_size)

print(np.shape(tmp_img))
print(np.shape(img))

# grayscale
for i in range(len(img)):
    for j in range(len(img[i])):
        tmp_img[i][j] = (img[i][j][0] * 0.114 + img[i][j][1] * 0.587 + img[i][j][2] * 0.299)/255 # tis how cv2 does

img = tmp_img

# cv2.imshow("win", np.multiply(img, np.eye(len(img))))
cv2.imshow("win", img.T)
cv2.waitKey(0)

# TODO play around with stuff


