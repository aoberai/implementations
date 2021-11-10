import numpy as np
import tensorflow as tf
import constants
import cv2
from dflow import DenseOpFlow
import argparse

def initialize_video(vid_type="train"):
    if vid_type == "train":
        train_speeds = []
        vid = cv2.VideoCapture("./data/train.mp4")
        with open("./data/train.txt") as f:
            for _ in range(int(vid.get(cv2.CAP_PROP_FRAME_COUNT))):
                train_speeds.append(f.readline())
        return (vid, train_speeds)
    else:
        vid = cv2.VideoCapture("./data/test.mp4")
        return (vid, None)


# Create the parser
cli_parser = argparse.ArgumentParser(description='')

# Add the arguments
cli_parser.add_argument('--model',
                        metavar='model',
                        type=str,
                        default=constants.model_path,
                        help='Model to run inference on')

cli_parser.add_argument('--pos',
                        metavar='pos',
                        type=float,
                        default=0.0,
                        help='Start position for video (given as proportion)')

args = cli_parser.parse_args()

if __name__ == "__main__":
    vid, train_speeds = initialize_video()
    position = round(args.pos * vid.get(cv2.CAP_PROP_FRAME_COUNT))
    vid.set(cv2.CAP_PROP_POS_FRAMES, position)

    op_flow = DenseOpFlow(cv2.resize(vid.read()[1], constants.image_size[0:2]))
    speed_predictor = tf.keras.models.load_model(args.model)
    model_frames = []
    counter = position

    while True:
        ret, frame = vid.read()
        if not ret:
            break
        model_frame = cv2.resize(
            frame, (constants.image_size[0], constants.image_size[1],))

        opflow_frame = op_flow.get(model_frame)
        merged = np.concatenate((model_frame / 255, opflow_frame / 255, ), 2)
        model_frames.append(merged)
        if len(model_frames) > constants.frame_window_size:
            del model_frames[0]
            frames_np = np.expand_dims(np.array(model_frames), 0)
            predicted_speed = speed_predictor.predict(frames_np)
            cv2.imshow(
                "Speed Predictor",
                cv2.resize(
                    frame,
                    (constants.image_size[0] *
                     3,
                     constants.image_size[1] *
                     3)))
            cv2.waitKey(1)
            print("Predicted Speed:",
                  predicted_speed[0][0],
                  "Real Speed:",
                  "None" if train_speeds is None else train_speeds[(counter := counter + 1)])
