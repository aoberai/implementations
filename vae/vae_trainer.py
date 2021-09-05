
# An implementation of an autoencoder in Tensorflow

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, Input, Sequential, layers
import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow.keras.backend as K
import time
import sys

tf.config.run_functions_eagerly(True)

(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')
x_train = x_train / 255.
x_test = x_test / 255.


print("\n\n\n Press Control-C to start Training")
while True:
    try:
        cv2.imshow("Dataset Image", x_train[0])
        cv2.waitKey(1)
    except KeyboardInterrupt:
        break

print("\n\n\n\nStarting Training Process\n\n\n")

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(60000).batch(32)

latent_space_dims = 5

def encoder(input_shape, output_latent_space_dims):
  inputs = Input(shape=input_shape)
  x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
  x = layers.BatchNormalization()(x)
  x = layers.LeakyReLU()(x)
  x = layers.Flatten()(x)

  # Bottleneck
  mean_vector = layers.Dense(output_latent_space_dims)(x)
  std_vector = layers.Dense(output_latent_space_dims)(x)

  encoder = Model(inputs, (mean_vector, std_vector), name="Encoder")
  print(encoder.summary())
  return encoder

encoder_model = encoder((28, 28, 1), latent_space_dims)

def decoder(output_shape, input_latent_space_dims):
  decoder = Sequential()
  inputs = Input(shape=(input_latent_space_dims))

  x = layers.Dense(output_shape[0] * output_shape[1])(inputs)
  x = tf.keras.layers.Reshape((28, 28, 1), input_shape=(784,))(x)

  x = layers.Conv2DTranspose(32, kernel_size=3, activation='relu', padding='same')(x)
  x = layers.BatchNormalization()(x)
  x = layers.LeakyReLU()(x)

  output = layers.Conv2DTranspose(1, kernel_size=3, activation='sigmoid', padding='same')(x)

  decoder = Model(inputs, output, name="Decoder")
  print(decoder.summary())

  return decoder

decoder_model = decoder((28, 28, 1), latent_space_dims)

optimizer = tf.keras.optimizers.Adam(learning_rate = 10e-3)

def reconstruction_loss(y_true, y_pred):
    loss = K.mean(K.square(y_true - y_pred), axis = [1,2,3])
    return loss

# Kullback–Leibler divergence
def kl_divergence_loss(mean_vector, std_vector):
    return -0.5 * K.sum(1 + K.log(1e-8 + K.square(std_vector)) - K.square(mean_vector) - K.square(std_vector), axis=1)

# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    with tf.GradientTape() as encoder, tf.GradientTape() as decoder:
      mean_vector, std_vector = encoder_model(images, training=True)

      # Sampling from gaussian latent space
      z_vector = mean_vector + tf.multiply(std_vector, tf.random.normal(tf.shape(mean_vector), 0, 1, tf.float32))

      generated_images = decoder_model(z_vector, training=True)

      loss = reconstruction_loss(images, generated_images) # reconstruction: calculates MSE difference between original and generated image
      # loss += kl_divergence_loss(mean_vector, std_vector) # regularization: calculates kullback leibler divergence between normal distribution and distribution for each latent space variable

    # Computes gradient through backpropping through operations specified in gradient tape section
    encoder_gradients = encoder.gradient(loss, encoder_model.trainable_variables)
    decoder_gradients = decoder.gradient(loss, decoder_model.trainable_variables)

    # Applies negative of derivative of cost function with respect to weights for both encoder and decoder
    optimizer.apply_gradients(zip(encoder_gradients, encoder_model.trainable_variables))
    optimizer.apply_gradients(zip(decoder_gradients, decoder_model.trainable_variables))

    return loss

def train(dataset, epochs=10):
    validation_loss = []
    training_loss = []

    mean_vector, std_vector = encoder_model.predict(x_test)

    # Sampling from gaussian latent space
    z_vector = mean_vector + tf.multiply(std_vector, tf.random.normal(tf.shape(mean_vector), 0, 1, tf.float32))

    generated_images = decoder_model.predict(z_vector)

    print("Epoch 0 Validation Loss:", np.mean(reconstruction_loss(x_test, generated_images))) # reconstruction: calculates MSE difference between original and generated image

    try:
        for epoch in range(epochs):
            start_time = time.time()
            loss_average = []
            counter = 0

            for image_batch in dataset:
              loss_average.append(np.mean(train_step(image_batch).numpy()))
              counter+=1
              print('Epoch Completed: {0:3f}'.format(counter / dataset.__len__().numpy()), end='\r')

            # validation_loss.append(np.square(x_test - decoder_model.predict(encoder_model.predict(x_test))).mean())

            mean_vector, std_vector = encoder_model.predict(x_test)

            # Sampling from gaussian latent space
            z_vector = mean_vector + tf.multiply(std_vector, tf.random.normal(tf.shape(mean_vector), 0, 1, tf.float32))

            generated_images = decoder_model.predict(z_vector)

            validation_loss.append(np.mean(reconstruction_loss(x_test, generated_images))) # reconstruction: calculates MSE difference between original and generated image
            # loss += kl_divergence_loss(mean_vector, std_vector) # regularization: calculates kullback leibler divergence between normal distribution and distribution for each latent space variable

            training_loss.append(np.mean(np.array(loss_average)))
            print('\nTime for epoch {} is {} sec; Training loss : {} Validation Loss: {} Counter : {}'.format(epoch + 1, time.time()-start_time, training_loss[-1], validation_loss[-1], counter))
            print("\n\n\nPress Control C to stop training")

    except KeyboardInterrupt:
        input()
        save_model = input("Would you like to save model?")

        if save_model == 'yes':
            print("Saving model")

            encoder_model.save("encoder.h5")
            decoder_model.save("decoder.h5")

train(train_dataset, epochs=50)


