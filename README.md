# CycleGAN
In this project, we attempt to modify the Loss function, and the architecture and analyze the results.
The code used for the project is added here.

The Blog can be accessed using the link below:

https://hackmd.io/byPmGZaBQqWTHlzFVKRFjA

To Create the Directory, clone this repository:

!git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

This will download the baseline cyclegan directory.
To review the Modified Loss Functions or Embedded CycleGAN, replace the files networks.py and cycle_gan_model.py in the baseline cyclegan directory named "models" with the files in the folders in this repository.

Notes:

For Modified Loss Functions, review the backward_G function in line 153 of models/cycle_gan_model.py and read the comments to enable to correct loss function.

For Embedded CycleGAN, review the forward function in line 375 of models/networks.py and read comments to enable the appropriate model for training and testing.

