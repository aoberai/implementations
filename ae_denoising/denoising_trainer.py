# Uses plain autoencoder to upsample image quality

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, Input, Sequential, layers
import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow.keras.backend as K
import time

tf.config.run_functions_eagerly(True)

mnist_train = np.load("dataset/mnist_train.npy")
mnist_noise_train = np.load("dataset/mnist_noise_train.npy")

upscale_size = (480, 480)

print("\n\n\n Press Control-C to start Training")
for i in range(len(mnist_train)):
    try:
        cv2.imshow("Raw Image", cv2.resize(mnist_train[i], upscale_size))
        cv2.imshow("Noisy Image", cv2.resize(mnist_noise_train[i], upscale_size))
        cv2.waitKey(1)
        if i > 1:
            time.sleep(3)
    except KeyboardInterrupt:
        break

print("\n\n\n\nStarting Training Process\n\n\n")

# Batch the data and place in tf.Dataset
BATCH_SIZE = 128
train_dataset_x = tf.data.Dataset.from_tensor_slices(mnist_noise_train).batch(BATCH_SIZE)
train_dataset_y = tf.data.Dataset.from_tensor_slices(mnist_train).batch(BATCH_SIZE)

test_dataset_x = np.load("dataset/mnist_noise_test.npy")
test_dataset_y = np.load("dataset/mnist_test.npy")

print(np.shape(mnist_noise_train), np.shape(mnist_train), np.shape(test_dataset_x), np.shape(test_dataset_y))

bottleneck_depth = 10

image_dims = (28, 28, 1)

def encoder(image_shape, bottleneck):
  inputs = Input(shape=image_shape)
  x = layers.Conv2D(64, 3, padding='same')(inputs)
  x = layers.BatchNormalization()(x)
  x = layers.LeakyReLU()(x)

  x = layers.Conv2D(32, 3, padding='same')(x)
  x = layers.BatchNormalization()(x)
  x = layers.LeakyReLU()(x)

  x = layers.Conv2D(8, 3, padding='same')(inputs)
  x = layers.BatchNormalization()(x)
  x = layers.LeakyReLU()(x)
  x = layers.Flatten()(x)

  outputs = layers.Dense(bottleneck)(x)
  encoder = Model(inputs, outputs, name="Encoder")
  print(encoder.summary)
  return encoder

encoder_model = encoder(image_dims, bottleneck_depth)

def decoder(bottleneck, reconstructed_image_shape):
  inputs = Input(shape=bottleneck)

  x = layers.Dense(reconstructed_image_shape[0] * reconstructed_image_shape[1])(inputs)
  x = tf.keras.layers.Reshape(reconstructed_image_shape)(x)
  x = layers.Conv2DTranspose(16, kernel_size=3, padding='same')(x)
  x = layers.BatchNormalization()(x)
  x = layers.LeakyReLU()(x)

  x = layers.Conv2D(8, kernel_size=3, padding='same')(x)
  x = layers.BatchNormalization()(x)
  x = layers.LeakyReLU()(x)

  outputs = layers.Conv2D(reconstructed_image_shape[2], kernel_size=3, padding='same')(x)

  decoder = Model(inputs, outputs, name="Decoder")
  print(decoder.summary())

  return decoder

decoder_model = decoder(bottleneck_depth, image_dims)

optimizer = tf.keras.optimizers.Adam(learning_rate = 10e-3)

def autoencoder_loss(y_true, y_pred):
    loss = K.mean(K.square(y_true - y_pred), axis = [1,2,3])
    return loss

# This annotation causes the function to be "compiled".
@tf.function
def train_step(noisy_images, orig_images):
    with tf.GradientTape() as encoder, tf.GradientTape() as decoder:
      latent = encoder_model(noisy_images, training=True)
      generated_images = decoder_model(latent, training=True)
      loss = autoencoder_loss(orig_images, generated_images) # calculates MSE difference between original image and generated image

    # Computes gradient through backpropping through operations specified in gradient tape section
    encoder_gradients = encoder.gradient(loss, encoder_model.trainable_variables)
    # print(encoder_model.trainable_variables)
    decoder_gradients = decoder.gradient(loss, decoder_model.trainable_variables)
    # print(decoder_model.trainable_variables)

    # Applies negative of derivative of cost function with respect to weights for both encoder and decoder
    optimizer.apply_gradients(zip(encoder_gradients, encoder_model.trainable_variables))
    optimizer.apply_gradients(zip(decoder_gradients, decoder_model.trainable_variables))

    return loss

def train(noisy_images, orig_images, noisy_images_test, orig_images_test, epochs=10):
    validation_loss = []
    training_loss = []
    try:
        for epoch in range(epochs):
            start_time = time.time()
            loss_average = []
            counter = 0
            for noisy_images_batch, orig_images_batch in zip(noisy_images, orig_images):
              loss_average.append(np.mean(train_step(noisy_images_batch, orig_images_batch).numpy()))
              counter+=1
              print('Epoch Completed: {0:3f}'.format(counter / noisy_images.__len__().numpy()), end='\r')

            validation_loss.append(np.square(orig_images_test - decoder_model.predict(encoder_model.predict(noisy_images_test))).mean())
            training_loss.append(np.mean(np.array(loss_average)))

            print('\nTime for epoch {} is {} sec; Training loss : {} Validation Loss: {} Counter : {}'.format(epoch + 1, time.time()-start_time, training_loss[-1], validation_loss[-1], counter))
            print("\n\n\nPress Control C to stop training")

    except KeyboardInterrupt:
            encoder_model.save("encoder.h5")
            decoder_model.save("decoder.h5")

train(train_dataset_x, train_dataset_y, test_dataset_x, test_dataset_y, epochs=50)


