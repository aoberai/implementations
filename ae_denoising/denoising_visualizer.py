
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, Input, Sequential, layers
import cv2
import numpy as np
import time

x_test = np.load("dataset/mnist_noise_test.npy")
y_test = np.load("dataset/mnist_test.npy")

encoder = tf.keras.models.load_model("models/encoder.h5")
decoder = tf.keras.models.load_model("models/decoder.h5")

display_image_size = (360, 360)

for i in range(0, len(x_test)):
    cv2.imshow("Goal Image", cv2.resize(y_test[i], display_image_size))
    cv2.imshow("Noisy Image", cv2.resize(x_test[i], display_image_size))
    encoded_image = encoder.predict(np.expand_dims(x_test[i], 0))
    print("Image squeezed to encoded shape: ", np.shape(encoded_image))
    generated_image = cv2.resize(decoder.predict(encoded_image)[0,:,:,:], display_image_size)
    cv2.imshow("Generated Image", generated_image)
    cv2.waitKey(1)

    if i != 0:
        input() # wait till enter pressed to load next image

    # time.sleep(3)


