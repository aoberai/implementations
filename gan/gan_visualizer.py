
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, Input, Sequential, layers
import cv2
import numpy as np
import time


generator = tf.keras.models.load_model("generator.h5", compile=False)

noise_dim = 100

counter = 0
while True:
    noise = np.expand_dims(tf.random.normal([noise_dim]).numpy(), 0)

    generated_image = generator.predict(noise)

    # print(generated_image)

    # print(type(generated_image))
    # print(np.shape(generated_image))
    # exit(0)

    cv2.imshow("Original Image", generated_image[0,:,:,:])
    cv2.waitKey(1)

    if counter != 0:
        input() # wait till enter pressed to load next image
    counter += 1



