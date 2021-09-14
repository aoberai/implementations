
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
import visualkeras
from PIL import ImageFont

image_shape = (256, 256, 3)

# Generator
# TODO: fix the weird thing with conv2d not showing up with consecutive same filter size
def generator(image_shape):

    inputs = Input(image_shape)
    stacks_conv_filter = [64, 64, 128, 128, 256, 256, 512, 512, 256, 256, 128, 128, 64, 64, 3]
    stacks_scale = [None, "MaxPool", None, "MaxPool", None, "MaxPool", None, None, "UpSample", "UpSample", "UpSample", "UpSample", "UpSample", None, None]
    stacks_dropout_rate = [0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0]

    print(len(stacks_conv_filter), len(stacks_scale), len(stacks_dropout_rate))
    assert len(stacks_conv_filter) == len(stacks_dropout_rate) == len(stacks_scale)

    def downsample(inputs, filter_count, conv_kernel_size=(3,3,), dropout_rate=0, maxpool=False):
        x = layers.Conv2D(filter_count, conv_kernel_size, padding='same')(inputs)
        if maxpool:
            x = layers.MaxPool2D(pool_size=(2, 2))(x)
        x = layers.BatchNormalization()(x)
        if dropout_rate != 0:
            x = layers.Dropout(dropout_rate)(x)
        outputs = layers.LeakyReLU()(x)
        return outputs

    def upsample(inputs, filter_count, conv_kernel_size=(3, 3,), dropout_rate=0, upsample=False):
        print(2 if upsample else 1)
        x = layers.Conv2DTranspose(filter_count, conv_kernel_size, strides=2 if upsample else 1, padding='same')(inputs)
        # x = layers.Conv2DTranspose(filter_count, conv_kernel_size, strides=2, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        if dropout_rate != 0:
            x = layers.Dropout(dropout_rate)(x)
        outputs = layers.ReLU()(x)
        return outputs

    s = inputs
    for i in range(0, len(stacks_conv_filter)):
        if stacks_conv_filter[i-1]  <= stacks_conv_filter[i]:
            # downsample; encoder
            s = downsample(s, stacks_conv_filter[i], (3, 3,), stacks_dropout_rate[i], True if stacks_scale[i]=="MaxPool" else False)
            print(s)
        else:
            # upsample; decoder
            s = upsample(s, stacks_conv_filter[i], (3, 3), stacks_dropout_rate[i], True if stacks_scale[i]=="UpSample" else False)
    return Model(inputs, s)


generator_model = generator(image_shape)
generator_model.summary()
# tf.keras.utils.plot_model(
#     generator_model,
#     to_file="model.png",
#     show_shapes=True, expand_nested=False)
# font=ImageFont.truetype("arial.ttf", 32)
visualkeras.layered_view(generator_model, legend=True, to_file='model.png')

# Discriminator

# L1 Loss

# Train








