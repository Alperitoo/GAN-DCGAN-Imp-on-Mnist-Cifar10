
# Gan and DcGan on Mnist and Cifar10
**This repository has been prepared for GAN and DCGAN implementation on [CIFAR10](#cifar10) and [MNIST](#mnist) datasets with using [Pytorch](https://pytorch.org/) and [TensorFlow-Keras](https://www.tensorflow.org/?hl=tr) **

## Requirements

To install requirements:
```setup
pip install -r requirements.txt
```

Folder structure:

```console
GAN-DCGAN-Imp-on-Mnist-Cifar10
├── CIFAR10
|   |__DCGAN-
|   |__GAN
├── MNIST
    |__DCGAN
    |__GAN

```
## MNIST
The MNIST dataset is a widely used benchmark in the field of machine learning and computer vision. It consists of a collection of handwritten digits from 0 to 9, each represented as a 28x28 grayscale image. MNIST has been extensively utilized for developing and evaluating algorithms in image classification, digit recognition, and deep learning models.

[MNIST](https://paperswithcode.com/dataset/mnist)


<p align="center">
    <img src="https://production-media.paperswithcode.com/datasets/MNIST-0000000001-2e09631a_09liOmx.jpg" width="640"\>
</p>

<b>See also:</b> [PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN)

## CIFAR10
The CIFAR-10 dataset is a popular dataset used for image classification tasks in machine learning and computer vision. It comprises 60,000 color images in 10 different classes, with each rgb image having a resolution of 32x32 pixels. The classes include common objects such as airplanes, automobiles, birds, cats, dogs, and more.
[CIFAR10](https://paperswithcode.com/dataset/cifar-10)


<p align="center">
    <img src="https://production-media.paperswithcode.com/datasets/4fdf2b82-2bc3-4f97-ba51-400322b228b1.png" width="640"\>
</p>

## IMPLEMENTATIONS
* DCGAN
  To train DCGAN models and save the model for both [mnist](#mnist) and [cifar10](#cifar10) datasets , 
  You need to run the 
  [Train-Code](MNIST/DCGAN/MNISTDCGAN.ipynb) , [Train-Code](CIFAR-10/DCGAN/DCGAN-CIFAR10.ipynb)
  The evaluations are inside the notebook for CIFAR10 dataset implementations.
  
  Evaluation and FID calculation for MNIST;
   [Eval-Code](MNIST/DCGAN/eval_fid.ipynb) 
* GAN
  To train GAN models and save the model for both [mnist](#mnist) and [cifar10](#cifar10) datasets , 
  You need to run the 
  [Train-Code](MNIST/GAN/GAN-MNIST.ipynb) , [Train-Code](CIFAR-10/GAN/GAN-CIFAR10.ipynb)
  
  The evaluations are inside the notebook for CIFAR10 dataset implementations.
  
  Evaluation and FID calculation for MNIST;
   [Eval-Code](MNIST/GAN/eval_fid.ipynb)

## Models 
  Models are saved for further use.
  ```console
    CIFAR-10
    ├── DCGAN
    |   |__model
    |   
    ├── GAN
    |   |__model
    |
    MNIST
    ├──DCGAN
    |    |__model
    |        |__gen
    |        |__gan
    |        |__disc
    |    
    ├──GAN
    |    |__model
    |        |__gen
    |        |__gan
    |        |__disc
```



## Installation
    $ git clone https://github.com/Alperitoo/DLGITHUB
    $ cd Keras-GAN/
    $ sudo pip3 install -r requirements.txt

## Results

Our model achieves the following performance on [MNIST](#mnist):

| Model name         | FID             |  
| ------------------ |---------------- | 
| GAN                |     123         |
|------------------- |---------------- |
| DCGAN              |    55.6         |
|------------------- |---------------- |

Our model achieves the following performance on [CIFAR-10](#cifar10)

| Model name         | FID             | Accuracy       |   
| ------------------ |---------------- |--------------- |
| GAN                |     159         |    0.67        |
|------------------- |---------------- |--------------- |
| DCGAN              |     126         |    0.985       |
|------------------- |---------------- |--------------- |


## References
* https://github.com/eriklindernoren/Keras-GAN
* https://nealjean.com/ml/frechet-inception-distance/
* https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
* https://github.com/soumith/ganhacks
* https://github.com/mseitzer/pytorch-fid
* DCGAN original [paper](https://arxiv.org/abs/1511.06434)
* GAN original [paper](https://arxiv.org/abs/1406.2661)

