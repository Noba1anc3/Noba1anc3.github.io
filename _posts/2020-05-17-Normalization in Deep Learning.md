---
layout: post
---

{: class="table-of-content"}
* TOC
{:toc}

Deep Neural Networks (DNNs) are notorious for requiring less feature engineering than Machine Learning algorithms. For example convolutional networks learn by themselves the right convolution kernels to apply on an image. No need of carefully handcrafted kernels.

However a common point to all kinds of neural networks is the need of normalization. Normalizing is often done on the input, but it can also take place inside the network. In this article I’ll try to describe what the literature is saying about this.

This article is not exhaustive but it tries to cover the major algorithms. If you feel I missed something important, tell me!

## Normalizing the input
It is extremely common to normalize the input [(lecun-98b)](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf), especially for computer vision tasks. Three normalization schemes are often seen:
1. Normalizing the pixel values between 0 and 1:

```
img /= 255.
```

2. Normalizing the pixel values between -1 and 1 (as [Tensorflow does](https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py#L42-L45)):

```
img /= 127.5
img -= 1.
```

3. Normalizing according to the dataset mean & standard deviation (as [Torch does](https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py#L47-L50)):

```
img /= 255.
mean = [0.485, 0.456, 0.406] # Here it's ImageNet statistics
std = [0.229, 0.224, 0.225]

for i in range(3): # Considering an ordering NCHW (batch, channel, height, width)
    img[i, :, :] -= mean[i]
    img[i, :, :] /= std[i]
```

Why is it recommended? Let’s take a neuron, where:  
$$y = w \cdot x$$

The partial derivative of $$y$$ for $$w$$ that we use during backpropagation is:  
$$\frac{\partial y}{\partial w} = X^T$$

The scale of the data has an effect on the magnitude of the gradient for the weights. If the gradient is big, you should reduce the learning rate. However you usually have different gradient magnitudes in a same batch. Normalizing the image to smaller pixel values is a cheap price to pay while making easier to tune an optimal learning rate for input images.

## Batch Normalization

We’ve seen previously how to normalize the input, now let’s see a normalization inside the network.

([Ioffe & Szegedy, 2015](https://arxiv.org/abs/1502.03167)) declared that DNN training was suffering from the internal covariate shift.

The authors describe it as:

```
[…] the distribution of each layer’s inputs changes during training,
 as the parameters of the previous layers change.
```

Their answer to this problem was to apply to the pre-activation a Batch Normalization (BN):
$$BN(x) = \gamma \frac{x - \mu_B}{\sigma_B} + \beta$$
$$\mu_B$$ and $$\sigma_B$$ are the mean and the standard deviation of the batch. $$\gamma$$ and $$\beta$$ are learned parameters.

The batch statistics are computed for a whole channel:
![](https://arthurdouillard.com/figures/batch_norm.png)
*Statistics are computed for a whole batch, channel per channel.*

$$\gamma$$ and $$\beta$$ are essential because they enable the BN to represent the identity transform if needed. If it couldn’t, the resulting BN’s transformation (with a mean of 0 and a variance of 1) fed to a sigmoid non-linearity would be constrained to its linear regime.

While during training the mean and standard deviation are computed on the batch, during test time BN uses the whole dataset statistics using a moving average/std.

Batch Normalization has showed a considerable training acceleration to existing architectures and is now an almost de facto layer. It has however for weakness to use the batch statistics at training time: With small batches or with a dataset non [i.i.d](https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables) it shows weak performance. In addition to that, the difference between training and test time of the mean and the std can be important, this can lead to a difference of performance between the two modes.

## Batch Renormalization
([Ioffe, 2017](https://arxiv.org/abs/1702.03275))’s Batch Renormalization (BR) introduces an improvement over Batch Normalization.

BN uses the statistics $$(\mu_B & \sigma_B)$$ of the batch. BR introduces two new parameters r & d aiming to constrain the mean and std of BN, reducing the extreme difference when the batch size is small.

Ideally the normalization should be done with the instance’s statistic:

$$\hat{x} = \frac{x - \mu}{\sigma}$$
By choosing $$r = \frac{\sigma_B}{\sigma} and d = \frac{\mu_B - \mu}{\sigma}$$:

$$\hat{x} = \frac{x - \mu}{\sigma} = \frac{x - \mu_B}{\sigma_B} \cdot r + d$$
The authors advise to constrain the maximum absolute values of r and d. At first to 1 and 0, behaving like BN, then to relax gradually those bounds.

## Internal Covariate Shift?

---
Cited as:
```
@article{
  title   = "CNN, DPM and Overfeat",
  author  = "Zhang, Xuanrui",
  journal = "noba1anc3.github.io",
  year    = "2020",
  url     = "https://noba1anc3.github.io/2020/04/22/CNN,-DPM-and-Overfeat.html"
}
```

## Reference

[1] Vincent Dumoulin and Francesco Visin. ["A guide to convolution arithmetic for deep learning."](https://arxiv.org/pdf/1603.07285.pdf) arXiv preprint arXiv:1603.07285 (2016).

[2] Haohan Wang, Bhiksha Raj, and Eric P. Xing. ["On the Origin of Deep Learning."](https://arxiv.org/pdf/1702.07800.pdf) arXiv preprint arXiv:1702.07800 (2017).

[3] Pedro F. Felzenszwalb, Ross B. Girshick, David McAllester, and Deva Ramanan. ["Object detection with discriminatively trained part-based models."](http://people.cs.uchicago.edu/~pff/papers/lsvm-pami.pdf) IEEE transactions on pattern analysis and machine intelligence 32, no. 9 (2010): 1627-1645.

[4] Ross B. Girshick, Forrest Iandola, Trevor Darrell, and Jitendra Malik. ["Deformable part models are convolutional neural networks."](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Girshick_Deformable_Part_Models_2015_CVPR_paper.pdf
) In Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), pp. 437-446. 2015.

[5] Sermanet, Pierre, David Eigen, Xiang Zhang, Michaël Mathieu, Rob Fergus, and Yann LeCun. ["OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks"](https://pdfs.semanticscholar.org/f2c2/fbc35d0541571f54790851de9fcd1adde085.pdf) arXiv preprint arXiv:1312.6229 (2013).
