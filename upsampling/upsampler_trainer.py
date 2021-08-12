# Uses plain autoencoder to upsample image quality

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, Input, Sequential, layers
import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow.keras.backend as K
import time

image_size = (32, 32)
(raw_images, _), (raw_images_test, _) = keras.datasets.cifar100.load_data()
raw_images = raw_images.reshape(raw_images.shape[0], image_size[0], image_size[1], 3).astype('float32')
raw_images_test = raw_images_test.reshape(raw_images_test.shape[0], image_size[0], image_size[1], 3).astype('float32')
raw_images = raw_images / 255.
raw_images_test = raw_images_test / 255

crunch_size = (10, 10)
upscale_size = (360, 240)

low_res_images = [cv2.resize(cv2.resize(image, crunch_size), image_size) for image in raw_images]
low_res_test_images = [cv2.resize(cv2.resize(image, crunch_size), image_size) for image in raw_images_test]


# shuffle data
random_seed=100
np.random.seed(random_seed)
np.random.shuffle(low_res_images)
np.random.seed(random_seed)
np.random.shuffle(raw_images)

print("\n\n\n Press Control-C to start Training")
for i in range(len(raw_images)):
    try:
        cv2.imshow("Raw Image", cv2.resize(raw_images[i], upscale_size))
        cv2.imshow("Low Res Image", cv2.resize(low_res_images[i], upscale_size))
        cv2.waitKey(1)
        time.sleep(3)
    except KeyboardInterrupt:
        break

print("\n\n\n\nStarting Training Process\n\n\n")

BATCH_SIZE = 128
# Batch the data and place in tf.Dataset
train_dataset_x = tf.data.Dataset.from_tensor_slices(low_res_images).batch(BATCH_SIZE)
train_dataset_y = tf.data.Dataset.from_tensor_slices(raw_images).batch(BATCH_SIZE)

bottleneck_depth = 32

def encoder(input_shape, output_depth):
  inputs = Input(shape=input_shape)
  x = layers.Conv2D(128, 3, activation='relu', padding='same')(inputs)
  x = layers.BatchNormalization()(x)
  x = layers.LeakyReLU()(x)

  x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
  x = layers.BatchNormalization()(x)
  x = layers.LeakyReLU()(x)

  x = layers.Conv2D(output_depth, 3, activation='relu', padding='same')(x)
  x = layers.BatchNormalization()(x)
  outputs = layers.LeakyReLU()(x)

  encoder = Model(inputs, outputs, name="Encoder")

  print(encoder.summary)

  return encoder

encoder_model = encoder(image_size + (3,), bottleneck_depth)

def decoder(input_depth):
  decoder = Sequential()
  print(image_size + (input_depth,))

  inputs = Input(shape=image_size + (input_depth,))

  x = layers.Conv2D(16, kernel_size=3, activation='relu', padding='same')(inputs)
  x = layers.BatchNormalization()(x)
  x = layers.LeakyReLU()(x)

  x = layers.Conv2D(8, kernel_size=3, activation='relu', padding='same')(x)
  x = layers.BatchNormalization()(x)
  x = layers.LeakyReLU()(x)

  outputs = layers.Conv2D(3, kernel_size=3, activation='sigmoid', padding='same')(x)

  decoder = Model(inputs, outputs, name="Decoder")
  print(decoder.summary())

  return decoder

decoder_model = decoder(image_size)

optimizer = tf.keras.optimizers.Adam(learning_rate = 10e-3)

def autoencoder_loss(y_true, y_pred):
    loss = K.mean(K.square(y_true - y_pred), axis = [1,2,3])
    temp_loss = K.mean(K.square(y_true - y_pred))
    assert loss == temp_loss # TODO: delete delete
    exit(0)
    return loss

# This annotation causes the function to be "compiled".
@tf.function
def train_step(low_res_images, raw_images):
    with tf.GradientTape() as encoder, tf.GradientTape() as decoder:
      latent = encoder_model(low_res_images, training=True)
      generated_images = decoder_model(latent, training=True)
      loss = autoencoder_loss(raw_images, generated_images) # calculates MSE difference between original image and generated image

    # Computes gradient through backpropping through operations specified in gradient tape section
    encoder_gradients = encoder.gradient(loss, encoder_model.trainable_variables)
    # print(encoder_model.trainable_variables)
    decoder_gradients = decoder.gradient(loss, decoder_model.trainable_variables)
    # print(decoder_model.trainable_variables)

    # Applies negative of derivative of cost function with respect to weights for both encoder and decoder
    optimizer.apply_gradients(zip(encoder_gradients, encoder_model.trainable_variables))
    optimizer.apply_gradients(zip(decoder_gradients, decoder_model.trainable_variables))

    return loss

def train(low_res_images_dataset, raw_images_dataset, low_res_test_images_dataset, raw_test_images_dataset, epochs=10):
    validation_loss = []
    training_loss = []
    try:
        for epoch in range(epochs):
            start_time = time.time()
            loss_average = []
            counter = 0

            for low_res_images_batch, raw_images_batch in low_res_images_dataset, raw_images_dataset:
              loss_average.append(np.mean(train_step(low_res_images_batch, raw_images_batch).numpy()))
              counter+=1
              print('Epoch Completed: {0:3f}'.format(counter / low_res_images_dataset.__len__().numpy()), end='\r')

            validation_loss.append(np.square(raw_test_images_dataset - decoder_model.predict(encoder_model.predict(low_res_test_images_dataset))).mean())
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

train(train_dataset_x, train_dataset_y, low_res_test_images, raw_images_test, epochs=50)


