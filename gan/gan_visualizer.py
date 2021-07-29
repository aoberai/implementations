
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, Input, Sequential, layers
import cv2
import numpy as np
import time


encoder = tf.keras.models.load_model("generator.h5")


noise = tf.random.normal([num_examples_to_generate, noise_dim])
for i in range(0, len(x_test)):
    image = x_test[i]
    print("Original Image Shape", np.shape(image))
    cv2.imshow("Original Image", cv2.resize(image, (360, 360)))
    encoded_image = encoder.predict(np.expand_dims(image, 0))
    print("Image squeezed to encoded shape: ", np.shape(encoded_image))
    # print(encoded_image)
    generated_image = cv2.resize(decoder.predict(encoded_image)[0,:,:,:], (360, 360))
    # print(np.shape(generated_image))
    cv2.imshow("Generated Image", generated_image)
    cv2.waitKey(1)

    if i != 0:
        input() # wait till enter pressed to load next image

    # time.sleep(3)


