# Some NN projects


## GAN Beaches

Given dataset of over a 1000 beaches from across the globe, this gan network can generate images of new beaches, none of which previously ever existed. 

Here is a sample image:

![](gan_beaches/generated_gan_beach.png)

Notice how the model learned a number of important characteristics of tropical beaches:

- Further you go away from the coast, the darker the color of water becomes. Vice versa is also true where the water near the beach is the lightest shade of blue.

- Sand generated around the coast, proportionally sound relative to water

- Coast shows some green, learning from the dataset that beaches are commonly surrounded by trees and other greenery.


## Stock State Prediction

Uses lstm to predict whether should buy, sell, or hold stock based on predicted state of stock in next x minutes.

It can be used to test if common day trading patterns exist broadly over a range of tech stocks which can successfully prove to serve as a long-term profitable trading strategy. If there is a high enough model accuracy, we are given evidence against the random walk hypothesis.

At the moment, the accuracy is not great on the model which could be attributable to the random walk hypothesis. 
