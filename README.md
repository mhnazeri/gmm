# SAS-GAN: Self-Attention Social GAN

To assure the safety of humans in an environment shared with machines, it is a necessity for machines to first understand human behavior and then based on this understanding, prevent fatal accidents by predicting future movements of humans.
SAS-GAN is an attempt to tackle the challenge of understanding the human driving behaviors.
The aim of this post is to gather all information about sasgan in one place.
The sasgan architecture consists of two separate networks, the first one is responsible for transforming all the features in to a unified latent feature space. Exerting the first network, the second network is responsible to generate plausible trajectories.

TODO list:
- [x] Data loader
- [ ] Visualization
- [ ] Network module
- [ ] Self-attention module
- [ ] ADE and FDE
- [ ] Train loop
- [ ] Test loop


## Tutorial on how to use the logger class: 
I have to recall that this class works with tensorboard and is independant of the python logger package.

### list of the functions

1. 
 
