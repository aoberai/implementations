import cv2
import pygame
import numpy as np
import tensorflow as tf
import argparse
import time

# Create the parser
cli_parser = argparse.ArgumentParser(description='List the content of a folder')

# Add the arguments
cli_parser.add_argument('--video',
                        metavar='video',
                        type=str,
                        default="./DrivingFootage.mp4",
                        help='Path to video to run on')

cli_parser.add_argument('--model',
                        metavar='model',
                        type=str,
                        default="./models/v4/GeneratorEpoch53.h5",
                        help='Model to run inference on')

cli_parser.add_argument('--pos',
                        metavar='pos',
                        type=float,
                        default=0.0,
                        help='Start position for video (given as proportion)')

cli_parser.add_argument('--size_multiplier',
                        metavar='size_multiplier',
                        type=float,
                        default=3.0,
                        help='Display Size: (size_multiplier * 256, size_multiplier * 256)')

args = cli_parser.parse_args()
assert args.pos <=1 and args.pos >= 0

generator_model = tf.keras.models.load_model(args.model) # Model 53 looking better than Model 63
orig_img_shape = (256, 256, 3)

display_res = tuple(int(val * args.size_multiplier) for val in orig_img_shape[:2])

# Visualize on video
cap = cv2.VideoCapture(args.video)
# Sets vid starting position
start_frame = args.pos*cap.get(cv2.CAP_PROP_FRAME_COUNT)
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

# TODO: fix bottom bar
bottom_bar_size = 20
full_display_res = display_res
# full_display_res = list(display_res)
# full_display_res[1] += bottom_bar_size
# full_display_res = tuple(full_display_res)

# Init Pygame
wn = pygame.display.set_mode(full_display_res)
clock = pygame.time.Clock()

result = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), cap.get(cv2.CAP_PROP_FPS), full_display_res)

def normalize(img):
    return img / 127.5 - 1

def denormalize(img):
    return (img + 1) * 127.5

gen_img_transparency = 0.5
delta_overlay_interval = 2
counter = delta_overlay_interval
current_frame_count = start_frame
skip_frame_count = 100
print("Press Up or Down arrows to change transparancy of segmentation mask")
while True:
    counter += 1
    current_frame_count += 1
    run, img = cap.read()
    img = cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
    if not run:
        break
    img = normalize(cv2.resize(img, orig_img_shape[:2]))
    gen_img = denormalize((generator_model.predict(np.expand_dims(img, 0))[0])).astype(np.uint8)
    img = denormalize(img).astype(np.uint8)
    overlay_img = cv2.resize(cv2.addWeighted(img, 1 - gen_img_transparency, gen_img, gen_img_transparency, 0), display_res)
    overlay_img = cv2.putText(overlay_img,'Purple: Your Car, Red: Lane Lines, Dark Brown: Road, Light Brown: Undrivable, Green: Other Vehicles', (bottom_bar_size, full_display_res[0] - bottom_bar_size), cv2.FONT_HERSHEY_DUPLEX, 0.7 * (display_res[0] * display_res[1])/(1000*1000), (255,255,255), 1)
    # padding = np.full((full_display_res[1], bottom_bar_size, 3), [255, 255, 255], dtype=np.uint8)
    # overlay_img = np.hstack((overlay_img, padding))
    # print(np.shape(overlay_img))
    # print(full_display_res)
    wn.blit(pygame.image.frombuffer(overlay_img.tobytes(), full_display_res, "BGR"), (0, 0))
    result.write(overlay_img)
    pygame.display.update()
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.KEYDOWN and counter >= delta_overlay_interval:
            # TODO: make arrow key skip x sec forward or backward
            if (event.key == pygame.K_DOWN):
                gen_img_transparency = max(gen_img_transparency - 0.1, 0)
                counter = 0
            elif (event.key == pygame.K_UP):
                gen_img_transparency = min(gen_img_transparency + 0.1, 1)
                counter = 0
            if event.key == pygame.K_LEFT:
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_count:=current_frame_count-skip_frame_count)
                counter = 0
            elif event.key == pygame.K_RIGHT:
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_count:=current_frame_count+skip_frame_count)
                counter = 0

            if event.key == pygame.K_SPACE:
                time.sleep(1)
                while True:
                    events = pygame.event.get()
                    if event.type == pygame.KEYDOWN and any(event.key == pygame.K_SPACE for event in pygame.event.get()):
                        break
                # print("++ Gen Img Transparency")
    # assert np.max(gen_img) <= 255 and np.min(gen_img) >= 0

cap.release()
result.release()

'''
# Visualize with test images
test_x_path = "/home/aoberai/programming/ml-datasets/comma10k/imgs/"
test_x_img_paths = [os.path.join(test_x_path, img_name) for img_name in os.listdir(test_x_path)]

test_y_path = "/home/aoberai/programming/ml-datasets/comma10k/masks/"
test_y_img_paths = [os.path.join(test_y_path, img_name) for img_name in os.listdir(test_y_path)]

# shuffle
seed = np.random.randint(100)
np.random.seed(seed)
np.random.shuffle(test_x_img_paths)
np.random.seed(seed)
np.random.shuffle(test_y_img_paths)

for i in range(len(test_x_img_paths)):
    x_img = cv2.imread(test_x_img_paths[i])
    x_scaling_factor = image_shape[0]/np.shape(x_img)[0]
    x_img = tf.image.random_crop(value=cv2.resize(x_img, None, fx=x_scaling_factor, fy=x_scaling_factor), size=image_shape).numpy()
    gen_img = (generator_model.predict(np.expand_dims(x_img, 0))[0]).astype(np.uint8)
    print(gen_img)
    print(np.max(gen_img))
    print(np.min(gen_img))
    cv2.imshow("Orig", x_img)
    cv2.imshow("Gen", gen_img)
    cv2.waitKey()

'''
