import tensorflow as tf
import cv2
import numpy as np
import time
import time
import constants


dataset_images = np.load('beaches.npy', mmap_mode = 'r')

BATCH_SIZE = 128

print("Beaches dataset size:", len(dataset_images))

def data_generator():
    for i in range(0, len(dataset_images)):
        yield dataset_images[i]

# train_dataset = tf.data.Dataset.from_tensor_slices(dataset_images).batch(BATCH_SIZE)
total_dataset = tf.data.Dataset.from_generator(generator=data_generator, output_signature=(tf.TensorSpec(shape=constants.image_shape, dtype=np.float32)))
total_dataset.batch(BATCH_SIZE).cache()

validation_set_size = int(len(dataset_images) * 0.2)

val_dataset = total_dataset.take(validation_set_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
train_dataset = total_dataset.skip(validation_set_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

def generator(noise_shape, output_shape):
    inputs = tf.keras.layers.Input(shape = noise_shape)
    x = tf.keras.layers.Dense(output_shape[0] * output_shape[1])(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

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

    outputs = tf.keras.layers.Conv2DTranspose(output_shape[2], kernel_size=5, padding='same')(x)

    generator = tf.keras.models.Model(inputs, outputs)

    assert generator.output_shape == (None,) + output_shape

    return generator

def discriminator(input_shape, output_size):
    inputs = tf.keras.layers.Input(shape = input_shape)
    x = tf.keras.layers.Conv2D(32, kernel_size=5, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    x = tf.keras.layers.Conv2D(64, kernel_size=5, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Conv2D(128, kernel_size=5, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)


    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(32)(x)
    outputs = tf.keras.layers.Dense(output_size)(x)

    discriminator = tf.keras.models.Model(inputs, outputs)

    return discriminator



noise_dim = 100

print("\n\n\n\nSize of dataset", len(dataset_images))

generator_model = generator((noise_dim,), constants.image_shape)
discriminator_model = discriminator(constants.image_shape, 1)

print("Created Model")

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss_function(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss_function(fake_output):
    # generator wants discriminator to think generated images are real
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
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

    return generator_loss
    # return {"generator_loss": generator_loss.numpy(), "discriminator_loss": discriminator_loss.numpy()}


def train(dataset, epochs=10):
    try:
        training_loss = []
        for epoch in range(epochs):
            start_time = time.time()
            loss_average = []
            counter = 0

            for image_batch in dataset:
              loss_average.append(train_step(image_batch))
              counter+=1
              print('Epoch Completed: %0.3f Loss: %0.5f' % (counter / dataset.__len__().numpy(), np.mean(np.array(loss_average))), end='\r')

            training_loss.append(np.mean(np.array(loss_average)))
            print('\nTime for epoch {} is {} sec; Training loss : {} Counter : {}'.format(epoch + 1, time.time()-start_time, training_loss[-1], counter))
            print("\n\n\nPress Control C to stop training")
            # ymax = 5 * (validation_loss[-1])
            # plt.scatter(epoch, validation_loss[-1])
            # plt.ylim(0, ymax)
            # plt.pause(0.05)
        # plt.show()

    except KeyboardInterrupt:
        input()
        save_model = input("Would you like to save model?")

        if save_model == 'yes':
            print("Saving model")

            generator_model.save("generator.h5")
            discriminator_model.save("discriminator.h5")


print("\n\n\n\n Starting Training ... \n\n\n")
train(train_dataset, epochs=50)


