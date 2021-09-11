
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, Input, Sequential, layers
import cv2
import numpy as np
import time


encoder = tf.keras.models.load_model("models/encoder.h5")
decoder = tf.keras.models.load_model("models/decoder.h5")

# encoder = tf.keras.models.load_model("encoder.h5")
# decoder = tf.keras.models.load_model("decoder.h5")

# encoder = tf.keras.models.load_model("encoder1epoch.h5")
# decoder = tf.keras.models.load_model("decoder1epoch.h5")

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


