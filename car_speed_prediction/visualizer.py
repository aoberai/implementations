import numpy as np
import tensorflow as tf
import constants as ct
import cv2
from dflow import DenseOpFlow
import argparse

op_flow = None

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
    # cv2.imshow("Frame", cv2.resize(np.hstack([frame, opflow_frame]), tuple(
    #     [scale_factor := 7 * i for i in ct.IMG_SIZE[0:2]])))
    # cv2.waitKey(1)

    frame = frame / 255
    opflow_frame = opflow_frame / 255
    # merged = np.stack((frame / 255, opflow_frame / 255), axis=2)
    return (frame, opflow_frame)


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
                        default=ct.MODEL_PATH,
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

    # op_flow = DenseOpFlow(cv2.resize(vid.read()[1], ct.IMG_SIZE[0:2]))
    speed_predictor = tf.keras.models.load_model(args.model)
    model_frames = []
    counter = position

    while True:
        ret, frame = vid.read()
        if not ret:
            break
        model_frames.append(preprocess(frame)[1])

        if len(model_frames) > ct.TIMESTEPS:
            del model_frames[0]
            frames_np = np.expand_dims(np.array(model_frames), 0)
            predicted_speed = speed_predictor.predict(frames_np)
            cv2.imshow(
                "Speed Predictor",
                cv2.resize(
                    frame,
                    (ct.IMG_SIZE[0] *
                     3,
                     ct.IMG_SIZE[1] *
                     3)))
            cv2.waitKey(1)
            print("Predicted Speed:",
                  predicted_speed[0][0],
                  "Real Speed:",
                  "None" if train_speeds is None else train_speeds[(counter := counter + 1)])
