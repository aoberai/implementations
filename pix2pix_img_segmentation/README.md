# Pix 2 Pix Image Segmentation 

## Car on Road

Tried to see if this technique could work well for image segmentation tasks such as this one - making things easier since one does not have to hand code a loss function for the generator. <br>

#### Status as of Sep 19, 2021
<br>

![](V2DemoImage.png)
<br>
**Pink**: Your Car 
<br>
**Green**: Other Car
<br>
**Dark Brown**: Drivable Road
<br>
**Light Brown**: Non-Drivable Area
<br>
**Red**: Lane Lines - Not working yet
<br>

# Other Details
**Generator**: Unet Architecture as implemented in [paper](https://arxiv.org/abs/1505.04597)
![](model_generator.png)

**Discriminator**: Need to implement PatchGan discriminator. Right now just uses simple conv net.

![](model_discriminator.png)

