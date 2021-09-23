# Car State: Pix 2 Pix Image Segmentation 

[![Watch the video](assets/DemoThumbnail.png)](assets/demo.avi)
<br>

Tried to implement this paper from scratch with minimal outside sources to see if I could. Applied GAN pix2pix idea to "traditional" semantic image segmentation which was not done in the paper. This generative adversarial architecture makes things easier since one does not have to hand code a loss function for the generator (discriminator model much more sophisticated compared to anything any individual could probably write). <br>

Paper: https://arxiv.org/abs/1611.07004v3
Dataset: https://github.com/commaai/comma10k

## Car on Road

Tried to see if this technique could work well for image segmentation tasks such as this one - <!-- #### Status as of Sep 19, 2021 -->
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

