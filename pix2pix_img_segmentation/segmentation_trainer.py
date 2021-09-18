
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

We can mix discriminator loss with simple traditional L1 or L2 loss (L1 encourages less blurring compared to l2) and prevents mode collapse (where generated samples are very similar)) -> groundtruths result for early iterations when discriminator not working well

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

Trained on comma10k dataset
'''


import tensorflow as tf
from tensorflow.keras import Input, Model, layers
import tensorflow.keras.backend as K
import numpy as np
import os
import cv2
import time

image_shape = (256, 256, 3)

def downsample(inputs, filter_count, conv_kernel_size=(3,3,), dropout_rate=0, downsample_num=0):
    x = layers.Conv2D(filter_count, conv_kernel_size, strides=2 if downsample_num > 0 else 1, padding='same')(inputs)
    for _ in range(downsample_num - 1):
        x = layers.MaxPooling2D((2, 2,))(x)
    x = layers.BatchNormalization()(x)
    if dropout_rate != 0:
        x = layers.Dropout(dropout_rate)(x)
    outputs = layers.LeakyReLU()(x)
    print(outputs)
    print("\n\n\n")
    return outputs

# TODO: no batch norm on output data in final layers?
def upsample(decode_inputs, skip_inputs, filter_count, conv_kernel_size=(3, 3,), dropout_rate=0, upsample_num=0, apply_batchnorm=False):
    inputs = tf.keras.layers.Concatenate()([decode_inputs, skip_inputs])
    x = layers.Conv2DTranspose(filter_count, conv_kernel_size, strides=2 if upsample_num > 0 else 1, padding='same')(inputs)
    for _ in range(upsample_num - 1):
        x = layers.UpSampling2D((2, 2,))(x)
    if apply_batchnorm:
        x = layers.BatchNormalization()(x)
    if dropout_rate != 0:
        x = layers.Dropout(dropout_rate)(x)
    outputs = layers.ReLU()(x)
    print(outputs)
    print("\n\n\n")
    return outputs

# Generator
def generator(image_shape):

    inputs = Input(image_shape)
    up_stacks_conv = [32, 64, 128]
    down_stacks_conv = [64, 32, 3]
    up_stacks_scale = [("DownSample",1), ("DownSample",1), ("DownSample",2)]
    down_stacks_scale = [("UpSample",2), ("UpSample",1), ("UpSample",1)]
    down_stacks_batchnorm = [False, False, False]
    up_stacks_dropout = [0, 0, 0]
    down_stacks_dropout = [0.2, 0.2, 0.2]

    assert len(up_stacks_conv) == len(up_stacks_dropout) == len(up_stacks_scale)
    assert len(down_stacks_conv) == len(down_stacks_dropout) == len(down_stacks_scale)
    print("\n\n\n\n")
    s = inputs
    skips = []

    for i in range(len(up_stacks_conv)):
        # downsample; encoder
        s = downsample(s, up_stacks_conv[i], (3, 3,), up_stacks_dropout[i], up_stacks_scale[i][1] if up_stacks_scale[i][0]=="DownSample" else 0)
        skips.append(s)
    for i in range(len(down_stacks_conv)):
        # upsample; decoder
        s = upsample(s, skips[-1], down_stacks_conv[i], (3, 3), down_stacks_dropout[i], down_stacks_scale[i][1] if down_stacks_scale[i][0]=="UpSample" else 0, apply_batchnorm=down_stacks_batchnorm[i])
        del skips[-1]

    print("\n\n\n\n")
    return Model(inputs, s)

# Discriminator

'''
def discriminator_patchgan(image_shape, receptive_field=(70, 70)):
    # run convolution across entire image and average the result
    # TODO: Finish
    pass
'''

def discriminator(image_shape):
    input_img = layers.Input(image_shape, name='input img')
    target_img = layers.Input(image_shape, name='target img')
    x = layers.concatenate([input_img, target_img])
    x = downsample(x, 128, downsample_num=1)
    x = downsample(x, 64, downsample_num=1)
    x = downsample(x, 32, downsample_num=1)
    x = downsample(x, 8, downsample_num=1)
    x = layers.Flatten()(x)
    outputs = layers.Dense(1)(x)
    return Model([input_img, target_img], outputs)


generator_model = generator(image_shape)
generator_model.summary()

discriminator_model = discriminator(image_shape)
discriminator_model.summary()

tf.keras.utils.plot_model(
    generator_model,
    to_file="model_generator.png",
    show_shapes=True,
    expand_nested=True
)
tf.keras.utils.plot_model(
    discriminator_model,
    to_file="model_discriminator.png",
    show_shapes=True,
    expand_nested=True
)

# L1 Loss
def l1_loss(orig_imgs, gen_imgs):
    l1_losses = [tf.reduce_mean(tf.abs(tf.subtract(orig_imgs[img_i], gen_imgs[img_i]))) for img_i in range(len(orig_imgs))]
    return tf.reduce_mean(l1_losses)


binary_cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def disc_loss_func(real_output, fake_output):
    # discriminator loss: real images as real and fake as fake
    disc_loss_val = binary_cross_entropy(tf.ones_like(real_output), real_output) + binary_cross_entropy(tf.zeros_like(fake_output), fake_output)
    return disc_loss_val

def gen_l1_loss_func(disc_fake_output, target_imgs, gen_imgs, l1_weight_lambda=100):
    # l1 loss to prevent mode collapse and groundtruth
    l1_loss_val = l1_loss(target_imgs, gen_imgs)
    # generator wants discriminator to think generated images are real
    disc_loss_val = binary_cross_entropy(tf.ones_like(disc_fake_output), disc_fake_output)
    weighted_loss = disc_loss_val + l1_weight_lambda * l1_loss_val
    return weighted_loss, disc_loss_val, l1_loss_val

gen_optim = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
disc_optim = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

@tf.function
def train_step(input_imgs, target_imgs):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_imgs = generator_model(input_imgs, training=True)
        disc_fake_output = discriminator_model([input_imgs, gen_imgs], training=True)
        disc_real_output = discriminator_model([input_imgs, target_imgs], training=True)

        gen_loss = gen_l1_loss_func(disc_fake_output, target_imgs, gen_imgs, l1_weight_lambda=150)[0]
        disc_loss = disc_loss_func(disc_real_output, disc_fake_output)

        gen_grads = gen_tape.gradient(gen_loss, generator_model.trainable_variables)
        disc_grads = disc_tape.gradient(disc_loss, discriminator_model.trainable_variables)

        gen_optim.apply_gradients(zip(gen_grads,
                                      generator_model.trainable_variables))
        disc_optim.apply_gradients(zip(disc_grads,
                                       discriminator_model.trainable_variables))


def fit(epochs=10, batch_size=64):
    for epoch in range(epochs):
        steps = 0
        print("\n\nEpoch:", epoch, "\n\n")
        train_x_path = "/home/aoberai/programming/ml-datasets/comma10k/imgs/"
        train_x_img_paths = [os.path.join(train_x_path, img_name) for img_name in os.listdir(train_x_path)]
        trainset_size = len(train_x_img_paths)

        train_y_path = "/home/aoberai/programming/ml-datasets/comma10k/masks/"
        train_y_img_paths = [os.path.join(train_y_path, img_name) for img_name in os.listdir(train_y_path)]

        # shuffle
        seed = np.random.randint(100)
        np.random.seed(seed)
        np.random.shuffle(train_x_img_paths)
        np.random.seed(seed)
        np.random.shuffle(train_y_img_paths)

        
        while True:
            img_batch = []
            x_img_path_batch = train_x_img_paths[:batch_size]
            y_img_path_batch = train_y_img_paths[:batch_size]
            if len(x_img_path_batch) < batch_size:
                break
            del train_x_img_paths[:batch_size]
            del train_y_img_paths[:batch_size]
            for (x_img_path, y_img_path) in zip(x_img_path_batch, y_img_path_batch):
                rd_seed = np.random.randint(100)

                # TODO put in preprocessing block
                x_img = cv2.imread(x_img_path)
                x_scaling_factor = image_shape[0]/np.shape(x_img)[0]
                x_img = cv2.resize(x_img, None, fx=x_scaling_factor, fy=x_scaling_factor)
                y_img = cv2.imread(y_img_path)
                y_scaling_factor = image_shape[0]/np.shape(y_img)[0]
                y_img = cv2.resize(y_img, None, fx=y_scaling_factor, fy=y_scaling_factor)
                img_batch.append(tf.image.random_crop(value=np.stack((x_img, y_img), axis=0), size=(2,) + image_shape, seed=rd_seed).numpy())

                '''
                Img Visualization
                cv2.imshow("X", x_img)
                cv2.imshow("X", img_batch[-1][0])
                cv2.waitKey(1)
                cv2.imshow("Y", y_img)
                cv2.imshow("Y", img_batch[-1][1])
                cv2.waitKey(1)
                time.sleep(5)
                '''

            # img_batch = np.swapaxes(np.array(np.divide(img_batch, 255), dtype="float32"), 0, 1)
            img_batch = np.swapaxes(np.array(img_batch, dtype="float32"), 0, 1) # TODO: why float 32
            train_step(img_batch[0], img_batch[1])
            steps+=1
            print("Steps:%d -- Progress:%0.4f" % (steps, steps / (trainset_size // batch_size)), end="\r")

            try:
                # Debug Gan Status
                x_img = cv2.imread(train_x_img_paths[-1])
                y_img = cv2.imread(train_y_img_paths[-1])

                x_scaling_factor = image_shape[0]/np.shape(x_img)[0]
                x_img = tf.image.random_crop(value=cv2.resize(x_img, None, fx=x_scaling_factor, fy=x_scaling_factor), size=image_shape).numpy()
                gen_img = np.round(generator_model.predict(np.expand_dims(x_img, 0))[0]).astype(np.uint8) # TODO: np round is because if channel vals are floats, then it displays rgb from 0 - 1 where we want image to be generated from 0 - 255
                # gen_img = generator_model.predict(np.expand_dims(x_img, 0))[0] 
                # stacked_img = np.concatenate([x_img, gen_img, cv2.resize(y_img, image_shape[:2])], axis=1)
                # cv2.imshow("Visualizer", stacked_img)
                cv2.imshow("Original", x_img)
                cv2.imshow("Generated", gen_img)
                cv2.imshow("Target", cv2.resize(y_img, image_shape[:2]))
                if steps % 100 == 0:
                    print(gen_img)
                cv2.waitKey(1)
            except Exception as e:
                pass

        generator_model.save("GeneratorEpoch%s.h5" % epoch)


fit(epochs=100, batch_size=16)

'''
# testing L1 Loss
orig_imgs = [[3, 4, 5,
  6, 7, 8,
  9, 6, 7],[3, 4, 5,
  6, 7, 8,
  9, 6, 7]]
gen_imgs = [[1, 2, 3,
  4, 5, 6,
  7, 8, 9], [1, 2, 3,
  4, 5, 6,
  7, 8, 9]]

print(l1_loss(orig_imgs, gen_imgs).numpy())

exit(0)
'''

