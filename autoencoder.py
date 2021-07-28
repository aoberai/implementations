
# An implementation of an autoencoder in Tensorflow
# https://learnopencv.com/autoencoder-in-tensorflow-2-beginners-guide/#intro-auto

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, Input, Sequential, layers
import matplotlib.pyplot as plt
import cv2

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255.
x_test = x_test / 255.

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(60000).batch(128)

latent_space_dims = 200

def encoder(input_shape, output_latent_space_dims):
  input = Input(shape=input_shape)
  x = layers.Conv2D(32, 3, activation='relu', padding='same')(input)
  x = layers.BatchNormalization()(x)
  x = layers.LeakyReLU()(x)
  x = layers.Flatten()(x)
  # Bottleneck
  output = layers.Dense(output_latent_space_dims)(x)

  encoder = Model(input, output, name="Encoder")

  print(encoder.summary())

  return encoder

encoder_model = encoder((28, 28, 1), latent_space_dims)

def decoder(output_shape, input_latent_space_dims):
  decoder = Sequential()
  input = Input(shape=input_latent_space_dims)
  x = layers.Dense(output_shape[0] * output_shape[1])(input)
  x = tf.keras.layers.Reshape((28, 28, 1), input_shape=(784,))(x)
  x = layers.Conv2DTranspose(32, kernel_size=3, activation='relu', padding='same')(x)
  x = layers.BatchNormalization()(x)
  x = layers.LeakyReLU()(x)
  output = layers.Conv2DTranspose(1, kernel_size=3, activation='relu', padding='same')(x)

  decoder = Model(input, output, name="Decoder")
  print(decoder.summary())


decoder_model = decoder((28, 28, 1), latent_space_dims)

