
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

Trained on comma10k dataset
'''


import tensorflow as tf
from tensorflow.keras import Input, Model, layers
import numpy as np
import cv2

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

def upsample(decode_inputs, skip_inputs, filter_count, conv_kernel_size=(3, 3,), dropout_rate=0, upsample_num=0):
    inputs = tf.keras.layers.Concatenate()([decode_inputs, skip_inputs])
    x = layers.Conv2DTranspose(filter_count, conv_kernel_size, strides=2 if upsample_num > 0 else 1, padding='same')(inputs)
    for _ in range(upsample_num - 1):
        x = layers.UpSampling2D((2, 2,))(x)
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
        s = upsample(s, skips[-1], down_stacks_conv[i], (3, 3), down_stacks_dropout[i], down_stacks_scale[i][1] if down_stacks_scale[i][0]=="UpSample" else 0)
        del skips[-1]

    print("\n\n\n\n")
    return Model(inputs, s)

# Discriminator
# def discriminator_patchgan(image_shape, receptive_field=(70, 70)):
    # run convolution across entire image and average the result
    # TODO: Finish
    # pass
def discriminator(image_shape):
    inputs = layers.Input((2,) + image_shape)
    x = downsample(inputs, 128, downsample_num=1)
    x = downsample(x, 64, downsample_num=1)
    x = downsample(x, 32, downsample_num=1)
    x = downsample(x, 8, downsample_num=1)
    x = layers.Flatten()(x)
    outputs = layers.Dense(1)(x)
    return Model(inputs, outputs)


generator_model = generator(image_shape)
generator_model.summary()

discriminator_model = discriminator(image_shape)
discriminator_model.summary()


# This may be garbage
# visualkeras.layered_view(generator_model, legend=True, to_file='model.png')

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
# from tensorflow.keras.datasets import mnist
#
# (mnist_train, _), (mnist_test, _) = mnist.load_data()
# mnist_train = mnist_train[:3000]
# mnist_test = mnist_test[:10]
# x_train = np.zeros((len(mnist_train),) + image_shape)
# x_test = np.zeros((len(mnist_test),) + image_shape)
#
# for i in range(len(mnist_train)):
#     x_train[i] = cv2.cvtColor(cv2.resize(mnist_train[i], (image_shape[0], image_shape[1],)), cv2.COLOR_GRAY2RGB)/255.0
#
# for i in range(len(mnist_test)):
#     x_test[i] = cv2.cvtColor(cv2.resize(mnist_test[i], (image_shape[0], image_shape[1],)), cv2.COLOR_GRAY2RGB)/255.0
#
# generator_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4), loss='mean_squared_error', metrics=['MSE'])
#
# generator_model.fit(x_train, x_train, batch_size=128, epochs=10, shuffle=1, validation_data=(x_test, x_test))
#

# L1 Loss

# Train








