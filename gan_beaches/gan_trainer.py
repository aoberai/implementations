import tensorflow as tf
import cv2
import numpy as np
import time
import time
import constants

# Configuring gpus for memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

dataset_images_mmap = np.load('beaches.npy', mmap_mode = 'r')

BATCH_SIZE = 8

def data_generator():
    for i in range(0, len(dataset_images_mmap)):
        yield dataset_images_mmap[i]

total_dataset = tf.data.Dataset.from_generator(generator=data_generator, output_signature=(tf.TensorSpec(shape=(constants.image_shape[1], constants.image_shape[0], 3,), dtype=np.float32)))

validation_set_size = int(len(dataset_images_mmap) * 0.2)

val_dataset = total_dataset.take(validation_set_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
train_dataset = total_dataset.skip(validation_set_size).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


def generator(noise_shape, output_shape):
    inputs = tf.keras.layers.Input(shape = noise_shape)
    x = tf.keras.layers.Dense(output_shape[0] * output_shape[1] * 4)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Reshape((output_shape[0]//4, output_shape[1]//4, 64,))(x)

    x = tf.keras.layers.Conv2DTranspose(32, kernel_size=5, strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Conv2DTranspose(16, kernel_size=5, strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    outputs = tf.keras.layers.Conv2DTranspose(output_shape[2], kernel_size=5, padding='same', activation='tanh')(x)

    generator = tf.keras.models.Model(inputs, outputs)

    assert generator.output_shape == (None,) + output_shape

    return generator

def discriminator(input_shape, output_size):
    inputs = tf.keras.layers.Input(shape = input_shape)
    x = tf.keras.layers.Conv2D(32, kernel_size=5, strides=(2, 2), padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Conv2D(64, kernel_size=5, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(0.3)(x)


    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(output_size)(x)

    discriminator = tf.keras.models.Model(inputs, outputs)

    return discriminator



noise_dim = 100

# print("\n\n\n\nSize of dataset", len(dataset_images))

generator_model = generator((noise_dim,), (constants.image_shape[1], constants.image_shape[0], 3,))
discriminator_model = discriminator((constants.image_shape[1], constants.image_shape[0], 3,), 1)

print("Created Model")

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss_function(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss_function(fake_output):
    # generator wants discriminator to think generated images are real
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(5e-5)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator_model(noise, training=True)

        real_output = discriminator_model(images, training=True)
        fake_output = discriminator_model(generated_images, training=True)

        generator_loss = generator_loss_function(fake_output)
        discriminator_loss = discriminator_loss_function(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(generator_loss, generator_model.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(discriminator_loss, discriminator_model.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator_model.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator_model.trainable_variables))

    return (generator_loss, discriminator_loss)


def train(epochs=10, epoch_save_checkpoint=10):
    try:
        dataset_len = None
        for epoch in range(epochs):
            generator_loss = []
            discriminator_loss = []
            start_time = time.time()
            counter = 0
            iterator = train_dataset.as_numpy_iterator()
            while True:
              try:
                image_batch = next(iterator)
                loss = train_step(image_batch)
                generator_loss.append(loss[0])
                discriminator_loss.append(loss[1])
                counter+=1
                print('Epoch Completed: %0.3f Generator Loss: %0.5f Discriminator Loss: %0.5f' % (counter / 1 if dataset_len is None else counter / dataset_len, np.array(generator_loss).mean(), np.array(discriminator_loss).mean()), end='\r')
              except StopIteration:
                  break

            dataset_len = counter

            print('\nTime for epoch {} is {} sec; Generator Training loss : {}; Discriminator Training loss : {}; Counter : {}'.format(epoch + 1, time.time()-start_time, np.array(generator_loss).mean(), np.array(discriminator_loss).mean(), counter))
            print("\n\n\nPress Control C to stop training")

            if epoch % epoch_save_checkpoint == 0:
                generator_model.save("generator_epoch%d.h5" % epoch)

    except KeyboardInterrupt:
        input()
        save_model = input("Would you like to save model?")

        if save_model == 'yes':
            print("Saving model")

            generator_model.save("generator_final.h5")
            discriminator_model.save("discriminator_final.h5")


epoch_save_checkpoint = 10 # save model every 10 epochs
print("\n\n\n\n Starting Training ... \n\n\n")
train(epochs=50, epoch_save_checkpoint=epoch_save_checkpoint)


