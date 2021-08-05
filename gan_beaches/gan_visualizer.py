
import tensorflow as tf
import cv2
import numpy as np
import constants


generator = tf.keras.models.load_model("best_generator1.h5", compile=False)
generator = tf.keras.models.load_model("best_generator2.h5", compile=False)

noise_dim = 100

counter = 0
while True:
    noise = np.expand_dims(tf.random.normal([noise_dim]).numpy(), 0)

    generated_image = generator.predict(noise)[0,:,:,:] * 255

    # print(generated_image)

    # print(type(generated_image))
    # print(np.shape(generated_image))
    # exit(0)

    # generated_image = cv2.resize(generated_image, (constants.image_shape[0]*2, constants.image_shape[1]*2)) * 255


    cv2.imshow("Original Image", generated_image)
    cv2.waitKey(1)

    if counter != 0:
        input() # wait till enter pressed to load next image
    counter += 1



