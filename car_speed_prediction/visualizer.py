import numpy as np
import tensorflow as tf
import constants
import cv2

test_vid = cv2.VideoCapture("./data/test.mp4")

speed_predictor = tf.keras.models.load_model(constants.model_path)

model_frames = []

while True:
    ret, frame = test_vid.read()
    if not ret:
        break
    model_frame = cv2.resize(frame, (constants.image_size[0], constants.image_size[1],))
    model_frames.append(model_frame)
    if len(model_frames) > constants.frame_window_size:
        del model_frames[0]
        frames_np = np.expand_dims(np.array(model_frames), 0)
        predicted_speed = speed_predictor.predict(frames_np)
        cv2.imshow("Speed Predictor", cv2.resize(frame, (constants.image_size[0]*3, constants.image_size[1]*3)))
        cv2.waitKey(1)
        print("Speed:", predicted_speed[0])
