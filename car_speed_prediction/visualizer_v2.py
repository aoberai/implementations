import numpy as np
import cv2
import sys
import constants as ct
from dflow import DenseOpFlow
import tensorflow as tf
import matplotlib.pyplot as plt

np.set_printoptions(threshold=sys.maxsize)

train_vid = cv2.VideoCapture("./data/train.mp4")
test_vid = cv2.VideoCapture("data/test.mp4")
train_speed = open("data/train.txt")
speeds = train_speed.read().split()
op_flow = None

train_X_img = []
train_X_flow = []
train_Y = []

test_X = []

trainset_size = train_vid.get(cv2.CAP_PROP_FRAME_COUNT)
testset_size = test_vid.get(cv2.CAP_PROP_FRAME_COUNT)

frame_index = int(0.6*train_vid.get(cv2.CAP_PROP_FRAME_COUNT))
train_vid.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

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
    print(max(opflow_frame[0])/255.0)

    frame = frame / 255.0
    opflow_frame = opflow_frame / 255.0
    return (frame, opflow_frame)

speed_predictor = tf.keras.models.load_model(ct.MODEL_PATH)
model_frames = []
predicted_speeds = []

while True:
    ret, frame = train_vid.read()
    if not ret:
        break
    data = preprocess(frame)
    train_X_img.append(data[0])
    train_X_flow.append(data[1])
    speed = speeds[frame_index]

    model_frames.append(data[1])

    if len(model_frames) > ct.TIMESTEPS:
        del model_frames[0]
        print(np.array(model_frames).dtype)
        frames_np = np.expand_dims(np.array(model_frames).astype("float32"), 0)
        predicted_speeds.append(predicted_speed := speed_predictor.predict(frames_np))
        # cv2.imshow("Flow", cv2.cvtColor(train_X_flow[-1].astype("float32"), cv2.COLOR_GRAY2BGR))
        # cv2.waitKey(1)
        print("Predicted Speed:",
              predicted_speed[0][0],
              "Real Speed:", speed)
    frame_index+=1

series = []
series.append(plt.scatter(
    np.array(predicted_speeds), np.array([i for i in range(len(predicted_speeds))]), 5))
plt.legend(series, ["prediction"], fontsize=8, loc='upper left')
plt.show()


