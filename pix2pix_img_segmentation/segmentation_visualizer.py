import cv2
import pygame
import numpy as np
import tensorflow as tf

generator_model = tf.keras.models.load_model("models/v4/GeneratorEpoch53.h5") # Model 53 looking better than Model 63
orig_img_shape = (256, 256, 3)

# Does this ':=' upset you :)
display_res = tuple(val * (size_multiplier:=3) for val in orig_img_shape[:2])

# Visualize on video
cap = cv2.VideoCapture("DrivingFootage.mp4")
# Sets vid starting position
cap.set(cv2.CAP_PROP_POS_FRAMES, (position_video:=0.3*cap.get(cv2.CAP_PROP_FRAME_COUNT)))
# Init Pygame
wn = pygame.display.set_mode(display_res)
clock = pygame.time.Clock()

def normalize(img):
    return img / 127.5 - 1

def denormalize(img):
    return (img + 1) * 127.5

gen_img_transparency = 0.5
delta_overlay_interval = 2
counter = delta_overlay_interval
print("Press Up or Down arrows to change transparancy of segmentation mask")
while True:
    counter += 1
    run, img = cap.read()
    if not run:
        break
    img = normalize(cv2.resize(img, orig_img_shape[:2]))
    gen_img = denormalize((generator_model.predict(np.expand_dims(img, 0))[0])).astype(np.uint8)
    img = denormalize(img).astype(np.uint8)
    overlay_img = cv2.resize(cv2.addWeighted(img, 1 - gen_img_transparency, gen_img, gen_img_transparency, 0), tuple(val * size_multiplier for val in orig_img_shape[:2]))
    wn.blit(pygame.image.frombuffer(overlay_img.tobytes(), display_res, "BGR"), (0, 0))
    pygame.display.update()
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.KEYDOWN:
            if (event.key == pygame.K_LEFT or event.key == pygame.K_DOWN) and counter >= delta_overlay_interval:
                gen_img_transparency = max(gen_img_transparency - 0.1, 0)
                counter = 0
                # print("-- Gen Img Transparency")
            elif (event.key == pygame.K_UP or event.key == pygame.K_RIGHT) and counter >= delta_overlay_interval:
                gen_img_transparency = min(gen_img_transparency + 0.1, 1)
                counter = 0
                # print("++ Gen Img Transparency")
    # assert np.max(gen_img) <= 255 and np.min(gen_img) >= 0

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
