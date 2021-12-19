import numpy as np
import cv2
import sys
import constants as ct
from dflow import DenseOpFlow

np.set_printoptions(threshold=sys.maxsize)

train_vid = cv2.VideoCapture("./data/train.mp4")
test_vid = cv2.VideoCapture("data/test.mp4")
train_speed = open("data/train.txt")
op_flow = None

train_X_img = []
train_X_flow = []
train_Y = []

test_X = []

trainset_size = train_vid.get(cv2.CAP_PROP_FRAME_COUNT)
testset_size = test_vid.get(cv2.CAP_PROP_FRAME_COUNT)

start_frame = 0*train_vid.get(cv2.CAP_PROP_FRAME_COUNT)
train_vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

def preprocess(frame):
    global op_flow
    frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[260:260+90, 180:180+260] , ct.IMG_SIZE[0:2])
    # roi = np.transpose(cv2.resize(cv2.cvtColor(cv2.imread("roi_mask.jpg"), cv2.COLOR_BGR2GRAY), np.shape(frame)[0:2]))
    # print(np.shape(frame), np.shape(roi))
    # frame = cv2.bitwise_and(frame, roi)
    if op_flow is None:
        op_flow = DenseOpFlow(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)) # TODO: this scuffed
    opflow_frame = cv2.cvtColor(op_flow.get(frame), cv2.COLOR_BGR2GRAY)
    '''
    Visualization
    '''
    cv2.imshow("Frame", cv2.resize(np.hstack([frame, opflow_frame]), tuple(
        [scale_factor := 7 * i for i in ct.IMG_SIZE[0:2]])))
    cv2.waitKey(1)

    frame = frame / 255
    opflow_frame = opflow_frame / 255
    return (frame, opflow_frame)


counter = 0
while True:
    ret, frame = train_vid.read()
    if not ret:
        break
    data = preprocess(frame)
    speed = float(train_speed.readline())
    train_X_img.append(data[0])
    train_X_flow.append(data[1])
    train_Y.append(speed)
    print("Speed:", speed)
    counter += 1
    print(counter / trainset_size)

np.save("frames", np.array(train_X_flow))
np.save("flow", np.array(train_X_flow))
np.save("speeds", np.array(train_Y))
