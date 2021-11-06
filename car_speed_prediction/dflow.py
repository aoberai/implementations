# https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html

import constants
import cv2
import numpy as np


class DenseOpFlow:
    def __init__(self, init_frame):
        # self.prvs = init_frame
        self.prvs = cv2.cvtColor(init_frame, cv2.COLOR_BGR2GRAY)
        self.hsv = np.zeros_like(init_frame)
        self.hsv[..., 1] = 255

    def get(self, image):
        # new_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            self.prvs, image, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        self.hsv[..., 0] = ang * 180 / np.pi / 2
        self.hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(self.hsv, cv2.COLOR_HSV2BGR)
        self.prvs = image
        return rgb
