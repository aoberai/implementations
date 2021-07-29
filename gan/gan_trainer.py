import tensorflow as tf
import cv2
import random



(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

image_shape = (360, 360, 1)
x_train = x_train.reshape(x_train.shape[0], image_shape[0], image_shape[1], image_shape[2]).astype('float32')
x_test = x_test.reshape(x_test.shape[0], image_shape[0], image_shape[1], image_shape[2]).astype('float32')
x_train = x_train / 255.
x_test = x_test / 255.

noise = tf.random.normal([1, 100])


def generator(noise_shape, output_shape):
    inputs = tf.keras.layers.Input(input_shape = noise_shape)
    x = tf.keras.layers.Dense(output_shape[0] * output_shape[1])(inputs)
    x = tf.keras.layers.Reshape((output_shape[0], output_shape[1], 1), input_shape=(output_shape[0] * output_shape[1],))(x)
    x = tf.keras.layers.Conv2DTranspose(128, kernel_size=5, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Conv2DTranspose(64, kernel_size=5, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Conv2DTranspose(32, kernel_size=5, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    outputs = tf.keras.layers.Conv2DTranspose(1, kernel_size=5, padding='same')(x)

    generator = tf.keras.models.Model(inputs, outputs)

    assert generator.output_shape == (None) + output_shape

    return generator

def discriminator(input_shape, output_shape):
    inputs = tf.keras.layers.Input(input_shape = input_shape)
    x = tf.keras.layers.Conv2D(32, kernel_size=5, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    
    x = tf.keras.layers.Conv2D(64, kernel_size=5, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)


    x = tf.keras.layers.Conv2D(128, kernel_size=5, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(32)(x)
    outputs = tf.keras.layers.Dense(2)(x)

    discriminator = tf.keras.models.Model(inputs, outputs)

    return discriminator

    



