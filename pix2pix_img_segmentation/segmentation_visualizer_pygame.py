import numpy as np
import cv2
import pygame
import tensorflow as tf

generator_model = tf.keras.models.load_model("models/v2/GeneratorEpoch9.h5")
image_shape = (256, 256, 3)


# Visualize on video

cap = cv2.VideoCapture("DrivingFootage.mp4")
success = True
wn = pygame.display.set_mode(image_shape[:2])
clock = pygame.time.Clock()

while success:
    # clock.tick(60)
    img = cv2.resize(cap.read()[1], image_shape[:2])
    # for event in pygame.event.get():
    gen_img = (generator_model.predict(np.expand_dims(img, 0))[0]).astype(np.uint8)
    wn.blit(pygame.image.frombuffer(gen_img.tobytes(), image_shape[:2], "BGR"), (0, 0))
    pygame.display.update()
    # pygame.quit()
