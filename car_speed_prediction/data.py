import numpy as np
import cv2
import sys
import constants
from dense_opflow import DenseOpFlow

np.set_printoptions(threshold=sys.maxsize)

train_vid = cv2.VideoCapture("./data/train.mp4")
test_vid = cv2.VideoCapture("data/test.mp4")
train_speed = open("data/train.txt")
op_flow = DenseOpFlow(cv2.resize(train_vid.read()[1], constants.image_size[0:2]))

train_X = []
train_Y = []

test_X = []

trainset_size = 20400
testset_size = 10798

def preprocess(original_frame):
    original_frame = cv2.resize(original_frame, constants.image_size[0:2])
    opflow_frame = op_flow.get(original_frame)
    cv2.imshow("Frame", np.hstack([original_frame, opflow_frame]))
    cv2.waitKey(1)
    merged = np.concatenate((original_frame / 255, opflow_frame / 255, ), 2)
    return merged

counter = 0
while True:
    ret, frame = train_vid.read()
    if not ret:
        break
    frame = preprocess(frame)
    speed = float(train_speed.readline())
    train_X.append(frame)
    train_Y.append(speed)
    counter+=1
    print(counter/trainset_size)

np.save("frames", np.array(train_X))
np.save("speeds", np.array(train_Y))

