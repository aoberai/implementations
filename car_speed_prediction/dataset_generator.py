import numpy as np
import cv2
import sys
import constants

np.set_printoptions(threshold=sys.maxsize)

train_vid = cv2.VideoCapture("./data/train.mp4")
test_vid = cv2.VideoCapture("data/test.mp4")
train_speed = open("data/train.txt")

train_X = []
train_Y = []

test_X = []

trainset_size = 20400
testset_size = 10798

def preprocess_frame(original_frame):
    # original_frame = original_frame.astype(np.float32)
    original_frame = cv2.resize(original_frame, (constants.image_size[0], constants.image_size[1]))
    cv2.imshow("Frame", original_frame)
    cv2.waitKey(1)

    original_frame = original_frame / 255.

    return original_frame

counter = 0
while True:
    ret, frame = train_vid.read()
    if not ret:
        break
    frame = preprocess_frame(frame)
    speed = float(train_speed.readline())
    train_X.append(frame)
    train_Y.append(speed)
    counter+=1
    print(counter/trainset_size)

np.save("frames", np.array(train_X))
np.save("speeds", np.array(train_Y))

# counter = 0
# while True:
#     ret, frame = test_vid.read()
#     if not ret:
#         break
#     frame = preprocess_frame(frame)
#     test_X.append(frame)
#     counter += 1
#     print(counter/testset_size)
#
# np.save("frames_test", np.array(test_X))
