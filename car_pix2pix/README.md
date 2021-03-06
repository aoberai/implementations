# Car State: Pix 2 Pix Image Segmentation 

https://user-images.githubusercontent.com/55261018/134445600-a3c9be78-3d52-4476-b14a-f4ab244ed777.mp4

<sub> Bigger model with longer training could make this a lot better; this approach has a ton of potentional. Unfortunately, I only have access to a small laptop which prevents my ability to train for long periods of time + I don't have enough RAM. </sub>


<br>

Tried to implement this paper from scratch with minimal outside sources to see if I could. Applied GAN pix2pix idea to "traditional" semantic image segmentation which was not done in the paper. This generative adversarial architecture makes things easier since one does not have to hand code a loss function for the generator (discriminator model much more sophisticated compared to anything any individual could probably write). My dataset was a set of 10k segmented images of cars driving in a variety of settings, made opensource by commaai. 

<br>

Paper: https://arxiv.org/abs/1611.07004v3
Dataset: https://github.com/commaai/comma10k


<!-- #### Status as of Sep 19, 2021 -->
<!-- <br> -->
<!-- **Pink**: Your Car  -->
<!-- <br> -->
<!-- **Green**: Other Car -->
<!-- <br> -->
<!-- **Dark Brown**: Drivable Road -->
<!-- <br> -->
<!-- **Light Brown**: Non-Drivable Area -->
<!-- <br> -->
<!-- **Red**: Lane Lines -->
<!-- <br> -->
# Other Details
**Generator**: Unet Architecture as implemented in [paper](https://arxiv.org/abs/1505.04597)
![](model_generator.png)

**Discriminator**: Need to implement PatchGan discriminator. Right now just uses simple conv net.

![](model_discriminator.png)

