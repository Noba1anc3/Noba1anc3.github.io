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
