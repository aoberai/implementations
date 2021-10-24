import numpy as np
import tensorflow as tf
import constants
import cv2
from dense_opflow import DenseOpFlow

vid_type = "test"
if vid_type == "train":
    vid = cv2.VideoCapture("./data/train.mp4")
    position_video = 0.9
    position = round(position_video * vid.get(cv2.CAP_PROP_FRAME_COUNT))
    vid.set(cv2.CAP_PROP_POS_FRAMES, position)
    train_speeds = open("./data/train.txt")
    for i in range(position):
        train_speeds.readline()
elif vid_type == "test":
    vid = cv2.VideoCapture("./data/test.mp4")

op_flow = DenseOpFlow(cv2.resize(vid.read()[1], constants.image_size[0:2]))

speed_predictor = tf.keras.models.load_model(constants.model_path)

model_frames = []

while True:
    ret, frame = vid.read()
    if not ret:
        break
    model_frame = cv2.resize(frame, (constants.image_size[0], constants.image_size[1],))

    opflow_frame = op_flow.get(model_frame)
    merged = np.concatenate((model_frame / 255, opflow_frame / 255, ), 2)
    model_frames.append(merged)
    if vid_type == "train":
        real_speed = train_speeds.readline()
    else:
        real_speed = "Unknown"
    if len(model_frames) > constants.frame_window_size:
        del model_frames[0]
        frames_np = np.expand_dims(np.array(model_frames), 0)
        predicted_speed = speed_predictor.predict(frames_np)
        cv2.imshow("Speed Predictor", cv2.resize(frame, (constants.image_size[0]*3, constants.image_size[1]*3)))
        cv2.waitKey(1)
        print("Predicted Speed:", predicted_speed[0][0], "Real Speed:", real_speed)

if vid_type == "train":
    train_speeds.close()
