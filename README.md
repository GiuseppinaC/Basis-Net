# BaSIS-Net
**BaSIS-Net: From Point Estimate to Predictive Distribution in Neural Networks - A Bayesian Sequential Importance Sampling Framework**

Data-driven Deep Learning (DL) models have revolutionized autonomous systems, but ensuring their safety and reliability necessitates the assessment of predictive confidence or uncertainty. Bayesian DL provides a principled approach to quantify uncertainty via probability density functions defined over model parameters. However, the exact solution is intractable for most DL models, and the approximation methods, often based on heuristics, suffer from scalability issues and stringent distribution assumptions and may lack theoretical guarantees. This work develops a Sequential Importance Sampling framework that approximates the posterior probability density function through weighted samples (or particles), which can be used to find the mean, variance, or higher-order moments of the posterior distribution. We demonstrate that propagating particles, which capture information about the higher-order moments, through the layers of the DL model results in increased robustness to natural and malicious noise (adversarial attacks). The variance computed from these particles effectively quantifies the model's decision uncertainty, demonstrating well-calibrated and accurate predictive confidence.

This repository contains the implementation of BaSIS-Net as described in the paper. To be published in Transactions on Machine Learning Research, 2024.
The project is structured into two folders (one for MNIST and one for CIFAR-10):

## Project structure
```sh
├── README.md

├── Main_CIFAR.py

├── Main_MNIST.py

├── model_functions.py

├── other_functions.py

├── Requirements.txt
```


**Scripts**

- **Main_MNIST.py**: The main script to run training and testing for the MNIST dataset.
- **Main_CIFAR.py**: The main script to run training and testing for the CIFAR-10 dataset.
- **model_functions.py**: Contains functions related to the model, such as generating particles, computing importance ratios, and likelihood.
- **other_functions.py**: Contains other utility functions, such as generating plots (e.g., histograms with predictive distributions) and computing metrics.

**Arguments**
Here is a list of all the arguments that can be used with **Main_MNIST.py** and **Main_CIFAR.py**:

- **--Training**: Flag to indicate if training should be performed (True or False).
- **--continue_training**: Flag to indicate if training should continue from a checkpoint (True or False).
- **--saved_model_epochs**: Number of epochs after which the model should be saved.
- **--Testing**: Flag to indicate if testing should be performed (True or False).
- **--weights**: Path to the weights to be used in the model.
- **--Random_noise**: Flag to indicate if random noise should be added (True or False).
- **--gaussian_noise_var**: Variance of the Gaussian noise to be added.
- **--Adversarial_noise**: Flag to indicate if (FGSM) adversarial noise should be added (True or False).
- **--epsilon**: Epsilon value for adversarial noise.
- **--adversary_target_cls**: Target class for adversarial noise.
- **--Targeted**: Flag to indicate if the adversarial attack is targeted (True or False).
- **--histogram**: Flag to indicate if histograms should be generated (True or False).
- **--num_kernels**: Number of kernels for convolutional layers.
- **--kernels_size**: Kernel size for convolutional layers.
- **--maxpooling_size**: Max pooling size.
- **--maxpooling_stride**: Stride for max pooling.
- **--maxpooling_pad**: Padding for max pooling.
- **--class_num**: Number of classes.
- **--batch_size**: Batch size for training.
- **--epochs**: Number of epochs for training.
- **--lr**: Learning rate.
- **--lr_end**: Ending learning rate.
- **--reg_factor**: Regularization factor.
- **--N**: Number of particles.
- **--init_sigma**: Initial sigma for particles.
- **--folder**: Folder to save model and logs.

In addition, **Main_MNIST.py** includes:

- **--sigma_particles_conv**: Sigma for particles in the convolutional layer.
- **--sigma_particles_fc**: Sigma for particles in the fully connected layer.

While **Main_CIFAR.py**:

- **--sigma_particles**: Sigma for particles in both convolutional as well as fully connected layers.
- **--PGD_Adversarial_noise**: Flag to indicate if PGD adversarial noise should be added (True or False).
- **--maxAdvStep**: Number of steps for the PGD attack.
- **--stepSize**: Step size for the PGD attack.
  
## Requirements

Install all required python dependencies:

```sh
pip install -r requirements.txt ```








