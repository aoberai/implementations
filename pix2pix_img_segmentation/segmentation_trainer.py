import tensorflow as tf
import numpy as np
import time
import constants
import cv2


'''
Based off of this paper: https://arxiv.org/abs/1611.07004v3
Uses GAN model to do image segmentation of car driving data

Summary of Paper:
Image to Image Translation : [Image of scene -> Segmentation mask]

Using Gan Approach ->
	- learns loss function to train mapping so no need to specify as with basic ConvNet
	- better than MSE loss which causes blurring due to "averaging all plausible outputs"
        - Setting high level goal of making indistinguishable from reality can make easier and better loss function; "can, in theory, penalize any possible structure that differs between output and target"
	- Reduces blurry images which are common with generative VAE's since discriminator learns that blurry images are obv fake


Existing Solutions


"Image to image translation problems are often formulated as per-pixel classification or regression." These individually take the loss for each pixel independently, considering the output space to be "unstructured" whereas something like a GAN calculates loss based on the "joint configuration of the output" i.e. the whole image produced.

How this solution works:

"Discriminator learns to classify betweeen fake (synthesized by generator) and real {input image, output image} tuples

Difference between normal gan architecture is that in a pix2pix GAN, both the generator and discriminator see the input image; the generator, instead of taking in only random noise(which is optional), takes in the input image which is to be translated to a different format.

We can mix discriminator loss with simple traditiona L1 or L2 loss (L1 encourages less blurring and prevents mode collapse (where generated samples are very similar)) -> groundtruths result for early iterations when discriminator not working well

For a lil bit of variation, you can apply dropout on the generator at both training and inference time instead of gaussian noise z (latent vector) which is often times just ignored by network.

Recommended modules are in the form of convolution -> batchnorm -> relu

Generator:
    U net is basically encoder decoder setup with "skip connections"
        Skip connections allow low level information to be shuttled without passing through bottleneck.
            U net adds connections between each layer i and layer n - i

Discriminator:
    Merging loss with L1 "motivates restricting ... to only model high-frequency structure, relying on a L1 term to force low-frequency correctness"
    Since only looking at high-freq struc, then discriminator only penalizes at the scale of patches. Classifies if NxN patch is real or fake. Runs this convolution filter across the whole image, averaging all responses for final discriminator loss - side benefit of this is that patchgan "can be applied to arbitrarily large images."
    Markovian in nature since patchgan assumes "independence between pixels separated by more than a patch diameter"

Optimization of network:

Alternate between one gradient descent step on D, then on G
Train G with loss func of log(D(x, G(x, z))) instead of log(1-D(x, G(x,z))), divide this by 2 while optimizing D, "slowing down the rate at which D learns relative to G"

Recs: lr: 2e-4; momentum=0.5
batch_size between 1 to 10
70x70 patchgans


'''

debug = True

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
    generator = tf.keras.models.Model(inputs, outputs)
    return generator

def discriminator(input_shape, output_size):
    discriminator = tf.keras.models.Model(inputs, outputs)
    return discriminator


noise_dim = 100

# print("\n\n\n\nSize of dataset", len(dataset_images))

generator_model = generator()
discriminator_model = discriminator()
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

    return (generator_loss, discriminator_loss)


def train(epochs=10, epoch_save_checkpoint=10):
    try:
        example_noise = tf.random.normal([noise_dim]).numpy()
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
                if debug and counter % 20 == 0: # displays every 5 train steps
                    example_image = generator_model.predict(np.expand_dims(example_noise, 0))[0]
                    # example_image = cv2.resize(example_image, (360*2, 240*2), interpolation = cv2.INTER_AREA)
                    cv2.imshow("Example", example_image)
                    cv2.waitKey(1)

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


