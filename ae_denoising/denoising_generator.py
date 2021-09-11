import tensorflow as tf
from tensorflow import keras
import numpy as np
import random

image_size = (28, 28)
(raw_images, _), (raw_images_test, _) = keras.datasets.mnist.load_data()
raw_images = raw_images.reshape(raw_images.shape[0], image_size[0], image_size[1], 1).astype('float32')
raw_images_test = raw_images_test.reshape(raw_images_test.shape[0], image_size[0], image_size[1], 1).astype('float32')
raw_images = raw_images / 255.
raw_images_test = raw_images_test / 255

crunch_size = (20, 20)
upscale_size = (360, 240)

noisy_images = np.zeros(np.shape(raw_images))
noisy_images_test = np.zeros(np.shape(raw_images_test))

random_seed=random.randint(0, 100)

noise_frequency = 0.2

# Adds some noise to the data
for image in range(len(raw_images)):
    print("Finished : ", int(image * 100 /np.shape(noisy_images)[0]), "%", end='\r')
    for row_pixel in range(len(raw_images[image])):
        for column_pixel in range(len(raw_images[image][row_pixel])):
            add_noise = (random.randint(1, 10) > 10*(1 - noise_frequency))
            noisy_images[image][row_pixel][column_pixel] = raw_images[image][row_pixel][column_pixel] if not add_noise else max(raw_images[image][row_pixel][column_pixel] + random.randint(1, 10) / 10, 1)

np.random.seed(random_seed)
np.random.shuffle(noisy_images)

np.random.seed(random_seed)
np.random.shuffle(raw_images)

np.save("mnist_train", raw_images)
np.save("mnist_noise_train", noisy_images)

print("Done adding noise to original data")

for image in range(len(raw_images_test)):
    print("Finished : ", int(image * 100 /np.shape(noisy_images_test)[0]), "%", end='\r')
    for row_pixel in range(len(raw_images_test[image])):
        for column_pixel in range(len(raw_images_test[image][row_pixel])):
            add_noise = (random.randint(1, 10) > 10*(1 - noise_frequency))
            noisy_images_test[image][row_pixel][column_pixel] = raw_images_test[image][row_pixel][column_pixel] if not add_noise else max(raw_images_test[image][row_pixel][column_pixel] + random.randint(1, 10) / 10, 1)


np.random.seed(random_seed)
np.random.shuffle(noisy_images_test)
np.random.seed(random_seed)
np.random.shuffle(raw_images_test)

np.save("mnist_test", raw_images_test)
np.save("mnist_noise_test", noisy_images_test)

print("Done adding noise to test data")
