
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, Input, Sequential, layers
import cv2
import numpy as np
import time

(_ , _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')
x_test = x_test / 255.

encoder = tf.keras.models.load_model("encoder.h5")
decoder = tf.keras.models.load_model("decoder.h5")

mean_vector, std_vector = encoder.predict(x_test)

# Sampling from gaussian latent space
z_vector = mean_vector + tf.multiply(std_vector, tf.random.normal(tf.shape(mean_vector), 0, 1, tf.float32))

generated_images = decoder.predict(z_vector)

for i in range(0, len(generated_images)):
    cv2.imshow("Original", cv2.resize(x_test[i], (360, 360)))
    cv2.waitKey(1)

    cv2.imshow("Generated", cv2.resize(generated_images[i], (360, 360)))
    cv2.waitKey(1)

    if i != 0:
        input() # wait till enter pressed to load next image


