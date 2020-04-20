---
layout: post
tags: object-detection object-recognition
---

> In This article, we would examine four object detection models: R-CNN, Fast R-CNN, Faster R-CNN, and Mask R-CNN. These models are highly related and the new versions show great speed improvement compared to the older ones.

{: class="table-of-content"}
* TOC
{:toc}

Here is a list of papers covered in this post ;)

| **Model**    | **Goal**           | **Resources**  |
| R-CNN        | Object recognition | [[paper](https://arxiv.org/abs/1311.2524)][[code](https://github.com/rbgirshick/rcnn)]   |
| Fast R-CNN   | Object recognition | [[paper](https://arxiv.org/abs/1504.08083)][[code](https://github.com/rbgirshick/fast-rcnn)]   |
| Faster R-CNN | Object recognition | [[paper](https://arxiv.org/abs/1506.01497)][[code](https://github.com/rbgirshick/py-faster-rcnn)]  |
| Mask R-CNN   | Image segmentation | [[paper](https://arxiv.org/abs/1703.06870)][[code](https://github.com/CharlesShang/FastMaskRCNN)] |
{:.info}


## R-CNN

R-CNN ([Girshick et al., 2014](https://arxiv.org/abs/1311.2524)) is short for "Region-based Convolutional Neural Networks". The main idea is composed of two steps. First, using [selective search](https://lilianweng.github.io/lil-log/2017/10/29/object-recognition-for-dummies-part-1.html#selective-search), it identifies a manageable number of bounding-box object region candidates ("region of interest" or "RoI"). And then it extracts CNN features from each region independently for classification.

<img src="/assets/images/RCNN.png" width="1333" height="354" align="middle" />
<img src="http://m.qpic.cn/psc?/fef49446-40e0-48c4-adcc-654c5015022c/90yfO.8bOadXEE4MiHsPnzJk4F9qtYmNuFFHSM0Ec3PipxLmElYUeQv8OJiNiq77tGxfeV.qPkrYdtB*e4XoUg!!/b&bo=WwaCA1sGggMRCT4!&rf=viewer_4"  width="1224" height="618" />

*Fig. 1. The architecture of R-CNN. (Image source: [Girshick et al., 2014](https://arxiv.org/abs/1311.2524))*


### Model Workflow

How R-CNN works can be summarized as follows:

1. **Pre-train** a CNN network on image classification tasks; for example, VGG or ResNet trained on [ImageNet](http://image-net.org/index) dataset. The classification task involves N classes. 
<br />
> NOTE: You can find a pre-trained [AlexNet](https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet) in Caffe Model [Zoo](https://github.com/caffe2/caffe2/wiki/Model-Zoo). I don’t think you can [find it](https://github.com/tensorflow/models/issues/1394) in Tensorflow, but Tensorflow-slim model [library](https://github.com/tensorflow/models/tree/master/research/slim) provides pre-trained ResNet, VGG, and others.
2. Propose category-independent regions of interest by selective search (~2k candidates per image). Those regions may contain target objects and they are of different sizes.
3. Region candidates are **warped** to have a fixed size as required by CNN. In the R-CNN paper, the size is set to 227*227.
> NOTE: Here are three different methods to warp the image:   
> 1.Resize width and height by different times, whether the image is distorted or not.   
> 2.Resize width and height by same times, fill the gray part by original pixel.  
> 3.Resize width and height by same times, don't fill the gray part.
4. Continue fine-tuning the CNN on warped proposal regions for K + 1 classes; The additional one class refers to the background (no object of interest). In the fine-tuning stage, we should use a much smaller learning rate and the mini-batch oversamples the positive cases because most proposed regions are just background.
> NOTE: In PASCAL VOC, K = 20.
> NOTE: The positive samples are proposed regions with IoU (intersection over union) overlap threshold 0.5, and negative samples are the others.
5. Given every image region, one forward propagation through the CNN generates a feature vector. This feature vector is then consumed by a **binary SVM** trained for **each class** independently. 
> Note: The positive samples are proposed regions with IoU (intersection over union) overlap threshold >= 0.3, and negative samples are irrelevant others.
6. To reduce the localization errors, a regression model is trained to correct the predicted detection window on bounding box correction offset using CNN features.
> Note: The positive sample are the proposed region with max IoU (intersection over union) overlap, and it should greater than 0.6.

### Bounding Box Regression

Given a predicted bounding box coordinate $$\mathbf{p} = (p_x, p_y, p_w, p_h)$$ (center coordinate, width, height) and its corresponding ground truth box coordinates $$\mathbf{g} = (g_x, g_y, g_w, g_h)$$ , the regressor is configured to learn scale-invariant transformation between two centers and log-scale transformation between widths and heights. All the transformation functions take $$\mathbf{p}$$ as input.

$$
\begin{aligned}
\hat{g}_x &= p_w d_x(\mathbf{p}) + p_x \\
\hat{g}_y &= p_h d_y(\mathbf{p}) + p_y \\
\hat{g}_w &= p_w \exp({d_w(\mathbf{p})}) \\
\hat{g}_h &= p_h \exp({d_h(\mathbf{p})})
\end{aligned}
$$

![bbox regression]({{ '/assets/images/RCNN-bbox-regression.png' }})
{: style="width: 60%;" class="center"}

*Fig. 2. Illustration of transformation between predicted and ground truth bounding boxes.*

An obvious benefit of applying such transformation is that all the bounding box correction functions, $$d_i(\mathbf{p})$$ where $$i \in \{ x, y, w, h \}$$, can take any value between [-∞, +∞]. The targets for them to learn are:

$$
\begin{aligned}
t_x &= (g_x - p_x) / p_w \\
t_y &= (g_y - p_y) / p_h \\
t_w &= \log(g_w/p_w) \\
t_h &= \log(g_h/p_h)
\end{aligned}
$$

A standard regression model can solve the problem by minimizing the SSE loss with regularization: 

$$
\mathcal{L}_\text{reg} = \sum_{i \in \{x, y, w, h\}} (t_i - d_i(\mathbf{p}))^2 + \lambda \|\mathbf{w}\|^2
$$

The regularization term is critical here and RCNN paper picked the best λ by cross validation. It is also noteworthy that not all the predicted bounding boxes have corresponding ground truth boxes. For example, if there is no overlap, it does not make sense to run bbox regression. Here, only a predicted box with a nearby ground truth box with at least 0.6 IoU is kept for training the bbox regression model.


### Common Tricks

Several tricks are commonly used in RCNN and other detection models.

**Non-Maximum Suppression**

Likely the model is able to find multiple bounding boxes for the same object. Non-max suppression helps avoid repeated detection of the same instance. After we get a set of matched bounding boxes for the same object category:
Sort all the bounding boxes by confidence score.
Discard boxes with low confidence scores.
*While* there is any remaining bounding box, repeat the following:
Greedily select the one with the highest score.
Skip the remaining boxes with high IoU (i.e. > 0.5) with previously selected one.

<img src="/assets/images/non-max-suppression.png" width="1307" height="428" align="middle" />

*Fig. 3. Multiple bounding boxes detect the car in the image. After non-maximum suppression, only the best remains and the rest are ignored as they have large overlaps with the selected one. (Image source: [DPM paper](http://lear.inrialpes.fr/~oneata/reading_group/dpm.pdf))*


**Hard Negative Mining**

We consider bounding boxes without objects as negative examples. Not all the negative examples are equally hard to be identified. For example, if it holds pure empty background, it is likely an “*easy negative*”; but if the box contains weird noisy texture or partial object, it could be hard to be recognized and these are “*hard negative*”. 

The hard negative examples are easily misclassified, and the recall rate is reduced. We can explicitly find those false positive samples with an IoU greater than a threshold during the training loops and include them in the training data so as to improve the classifier.


### Speed Bottleneck

Looking through the R-CNN learning steps, you could easily find out that training an R-CNN model is expensive and slow, as the following steps involve a lot of work:
- Running selective search to propose 2000 region candidates for every image;
- Generating the CNN feature vector for every image region (N images * 2000).
- The whole process involves three models separately without much shared computation: the convolutional neural network for image classification and feature extraction; the top SVM classifier for identifying target objects; and the regression model for tightening region bounding boxes.



## Fast R-CNN

To make R-CNN faster, Girshick ([2015](https://arxiv.org/pdf/1504.08083.pdf)) improved the training procedure by unifying three independent models into one jointly trained framework and increasing shared computation results, named **Fast R-CNN**. Instead of extracting CNN feature vectors independently for each region proposal, this model aggregates them into one CNN forward pass over the entire image and the region proposals share this feature matrix. Then the same feature matrix is branched out to be used for learning the object classifier and the bounding-box regressor. In conclusion, computation sharing speeds up R-CNN.

![Fast R-CNN]({{ '/assets/images/fast-RCNN.png' }})
{: style="width: 540px;" class="center"}
*Fig. 4. The architecture of Fast R-CNN. (Image source: [Girshick, 2015](https://arxiv.org/pdf/1504.08083.pdf))*


### RoI Pooling

It is a type of max pooling to convert features in the projected region of the image of any size, h x w, into a small fixed window, H x W. The input region is divided into H x W grids, approximately every subwindow of size h/H x w/W. Then apply max-pooling in each grid.


![RoI pooling]({{ '/assets/images/roi-pooling.png' }})
{: style="width: 540px;" class="center"}
*Fig. 5. RoI pooling (Image source: [Stanford CS231n slides](http://cs231n.stanford.edu/slides/2016/winter1516_lecture8.pdf).)*


### Model Workflow

How Fast R-CNN works is summarized as follows; many steps are same as in R-CNN: 
1. First, pre-train a convolutional neural network on image classification tasks.
2. Propose regions by selective search (~2k candidates per image).
3. Alter the pre-trained CNN:
	- Replace the last max pooling layer of the pre-trained CNN with a [RoI pooling](#roi-pooling) layer. The RoI pooling layer outputs fixed-length feature vectors of region proposals. Sharing the CNN computation makes a lot of sense, as many region proposals of the same images are highly overlapped.
	- Replace the last fully connected layer and the last softmax layer (K classes) with a fully connected layer and softmax over K + 1 classes.
4. Finally the model branches into two output layers:
	- A softmax estimator of K + 1 classes (same as in R-CNN, +1 is the "background" class), outputting a discrete probability distribution per RoI.
	- A bounding-box regression model which predicts offsets relative to the original RoI for each of K classes.


### Loss Function

The model is optimized for a loss combining two tasks (classification + localization):

| **Symbol** | **Explanation** |
| $$u$$ | True class label, $$ u \in 0, 1, \dots, K$$; by convention, the catch-all background class has $$u = 0$$. |
| $$p$$ | Discrete probability distribution (per RoI) over K + 1 classes: $$p = (p_0, \dots, p_K)$$, computed by a softmax over the K + 1 outputs of a fully connected layer. |
| $$v$$ | True bounding box $$ v = (v_x, v_y, v_w, v_h) $$. |
| $$t^u$$ | Predicted bounding box correction, $$t^u = (t^u_x, t^u_y, t^u_w, t^u_h)$$. See [above](#bounding-box-regression). |
{:.info}


The loss function sums up the cost of classification and bounding box prediction: $$\mathcal{L} = \mathcal{L}_\text{cls} + \mathcal{L}_\text{box}$$. For "background" RoI, $$\mathcal{L}_\text{box}$$ is ignored by the indicator function $$\mathbb{1} [u \geq 1]$$, defined as:

$$
\mathbb{1} [u >= 1] = \begin{cases}
    1  & \text{if } u \geq 1\\
    0  & \text{otherwise}
\end{cases}
$$

The overall loss function is:

$$
\begin{align*}
\mathcal{L}(p, u, t^u, v) &= \mathcal{L}_\text{cls} (p, u) + \mathbb{1} [u \geq 1] \mathcal{L}_\text{box}(t^u, v) \\
\mathcal{L}_\text{cls}(p, u) &= -\log p_u \\
\mathcal{L}_\text{box}(t^u, v) &= \sum_{i \in \{x, y, w, h\}} L_1^\text{smooth} (t^u_i - v_i)
\end{align*}
$$

The bounding box loss $$\mathcal{L}_{box}$$ should measure the difference between $$t^u_i$$ and $$v_i$$ using a **robust** loss function. The [smooth L1 loss](https://github.com/rbgirshick/py-faster-rcnn/files/764206/SmoothL1Loss.1.pdf) is adopted here and it is claimed to be less sensitive to outliers.

$$
L_1^\text{smooth}(x) = \begin{cases}
    0.5 x^2             & \text{if } \vert x \vert < 1\\
    \vert x \vert - 0.5 & \text{otherwise}
\end{cases}
$$

![Smooth L1 loss]({{ '/assets/images/l1-smooth.png' }})
{: style="width: 240px;" class="center"}
*Fig. 6. The plot of smooth L1 loss, $$y = L_1^\text{smooth}(x)$$. (Image source: [link](https://github.com/rbgirshick/py-faster-rcnn/files/764206/SmoothL1Loss.1.pdf))*

### Improvement

1. Only do convolution once on the whole image, refrain from repetitive computation.
2. Use RoI Pooling to transform all features into the same size, refrained from image distortion.
3. Use softmax layer for classification.
4. Integrate bbox regression into the network, and use a unified loss function.

### Speed Bottleneck

Fast R-CNN is much faster in both training and testing time. However, the improvement is not dramatic because the region proposals are generated separately by another model and that is very expensive.


## Faster R-CNN

An intuitive speedup solution is to integrate the region proposal algorithm into the CNN model. **Faster R-CNN** ([Ren et al., 2016](https://arxiv.org/pdf/1506.01497.pdf)) is doing exactly this: construct a single, unified model composed of RPN (region proposal network) and fast R-CNN with shared convolutional feature layers.

<img src="/assets/images/faster-RCNN.png" width="1260" height="600" align="middle" />

*Fig. 7. An illustration of Faster R-CNN model. (Image source: [Ren et al., 2016](https://arxiv.org/pdf/1506.01497.pdf))*

### Model Parts
#### Abstract
1. Conv Layers: It uses a set of conv + relu + pooling layers to extract feature maps of the image.
2. Region Proposal Networks: RPN is used for generating region proposals. It classify an anchor is positive or negative by softmax, and use bounding box regression to correct the position.
3. RoI Pooling: Its input is the feature maps and RPs, and extract proposal feature maps out for classification.
4. Classification: calculate the class of the proposal by proposal feature maps, use bbox regression to get the precise location.

Here is an architecture of VGG-16 based Faster R-CNN. The input size is P * Q, and it is resized to M * N. There are 13 conv layers, 13 ReLU layers and 4 Pooling layers in the Conv layers. RPN first calculate a 3 * 3 conv, then generate proposals comprised of positive anchors and its corresponding bbox offset. RoI Pooling layer uses proposals to extract proposal feature from feature maps. Then proposal features is sent into fc layer and softmax for classification.

<img src="https://pic4.zhimg.com/80/v2-e64a99b38f411c337f538eb5f093bdf3_720w.jpg" />

*Fig. 8. The Architecture of VGG-16 based Faster R-CNN model. (Image source: pascal_voc/VGG16/faster_rcnn_alt_opt/faster_rcnn_test.pt)*

#### Conv Layers
In VGG-16, there are 13 conv layers, 13 ReLU layers and 4 Pooling layers.
- All the conv layers with kernel size = 3, padding = 1, stride = 1
- All the pooling layers with kernel size = 2, padding = 0, stride = 2

After the padding process, the size is changed to (M+2) * (N+2), and after the conv process, the size is returned to M * N. So, the matrix size don't change during all the convolution process.

![](https://pic2.zhimg.com/80/v2-3c772e9ed555eb86a97ef9c08bf563c9_720w.jpg)

*Fig. 9. Convolution Process of VGG-16*

After the pooling layer, the M * N matrix is transformed into (M/2) * (N/2).  
So, after the Conv layers of Faster R-CNN, the size is changed to (M/16) * (N/16).

#### Region Proposal Network
Classical methods for generating bboxes are time-consuming, like sliding-window + paramid in adaboost, or selective search in R-CNN. In Faster R-CNN, it generate bboes by RPN directly. This is the huge advantage of generating bboxes in a fast manner.

![](https://pic3.zhimg.com/80/v2-1908feeaba591d28bee3c4a754cca282_720w.jpg)

*Fig. 10. The Architecture of RPN in Faster R-CNN*

There are two process line in RPN, the upper one decides whether an anchor is positive or negative by softmax, the lower one calculates the offset for positive anchors by bbox regression. It will discard small proposals and over-bounding proposals.

#### Multi-Channel Convolution with Muiti-Kernel
![](https://pic1.zhimg.com/80/v2-8d72777321cbf1336b79d839b6c7f9fc_720w.jpg)

*Fig. 11. Calculation Process of Multi-Channel Convolution with Multi-Kernel*

The input is a 3 channel image, and there are two conv kernels. Each kernel convolution on 3 channels, and add them as the output.  
For every convolution layer, whether there are how many channels in the input, the num of output channel always equals to the num of kernels.  
When doing 1 * 1 convolution on multi-channel image, it means to add up all the channel by a conv parameter of the image. In other words, to mix all the independent channels up.

### Model Workflow
1. Pre-train a CNN network on image classification tasks.
2. Fine-tune the RPN (region proposal network) end-to-end for the region proposal task, which is initialized by the pre-train image classifier. Positive samples have IoU (intersection-over-union) > 0.7, while negative samples have IoU < 0.3.
	- Slide a small n x n spatial window over the conv feature map of the entire image.
	- At the center of each sliding window, we predict multiple regions of various scales and ratios simultaneously. An anchor is a combination of (sliding window center, scale, ratio). For example, 3 scales + 3 ratios => k=9 anchors at each sliding position.
3. Train a Fast R-CNN object detection model using the proposals generated by the current RPN
4. Then use the Fast R-CNN network to initialize RPN training. While keeping the shared convolutional layers, only fine-tune the RPN-specific layers. At this stage, RPN and the detection network have shared convolutional layers!
5. Finally fine-tune the unique layers of Fast R-CNN
6. Step 4-5 can be repeated to train RPN and Fast R-CNN alternatively if needed.


### Loss Function

Faster R-CNN is optimized for a multi-task loss function, similar to fast R-CNN.

| **Symbol**  | **Explanation** |
| $$p_i$$     | Predicted probability of anchor i being an object. |
| $$p^*_i$$   | Ground truth label (binary) of whether anchor i is an object. |
| $$t_i$$     | Predicted four parameterized coordinates. |
| $$t^*_i$$   | Ground truth coordinates. |
| $$N_\text{cls}$$ | Normalization term, set to be mini-batch size (~256) in the paper. |
| $$N_\text{box}$$ | Normalization term, set to the number of anchor locations (~2400) in the paper. |
| $$\lambda$$ | A balancing parameter, set to be ~10 in the paper (so that both $$\mathcal{L}_\text{cls}$$ and $$\mathcal{L}_\text{box}$$ terms are roughly equally weighted). |
{:.info}

The multi-task loss function combines the losses of classification and bounding box regression:

$$
\begin{align*}
\mathcal{L} &= \mathcal{L}_\text{cls} + \mathcal{L}_\text{box} \\
\mathcal{L}(\{p_i\}, \{t_i\}) &= \frac{1}{N_\text{cls}} \sum_i \mathcal{L}_\text{cls} (p_i, p^*_i) + \frac{\lambda}{N_\text{box}} \sum_i p^*_i \cdot L_1^\text{smooth}(t_i - t^*_i) \\
\end{align*}
$$

where $$\mathcal{L}_\text{cls}$$ is the log loss function over two classes, as we can easily translate a multi-class classification into a binary classification by predicting a sample being a target object versus not. $$L_1^\text{smooth}$$ is the smooth L1 loss.

$$
\mathcal{L}_\text{cls} (p_i, p^*_i) = - p^*_i \log p_i - (1 - p^*_i) \log (1 - p_i)
$$



## Mask R-CNN

Mask R-CNN ([He et al., 2017](https://arxiv.org/pdf/1703.06870.pdf)) extends Faster R-CNN to pixel-level image segmentation. The key point is to decouple the classification and the pixel-level mask prediction tasks. Based on the framework of [Faster R-CNN](#faster-r-cnn), it added a third branch for predicting an object mask in parallel with the existing branches for classification and localization. The mask branch is a small fully-connected network applied to each RoI, predicting a segmentation mask in a pixel-to-pixel manner.

<img src="/assets/images/mask-rcnn.png" width="682" height="335" align="middle" />

*Fig. 9. Mask R-CNN is Faster R-CNN model with image segmentation. (Image source: [He et al., 2017](https://arxiv.org/pdf/1703.06870.pdf))*

Because pixel-level segmentation requires much more fine-grained alignment than bounding boxes, mask R-CNN improves the RoI pooling layer (named "RoIAlign layer") so that RoI can be better and more precisely mapped to the regions of the original image.


![Mask R-CNN Examples]({{ '/assets/images/mask-rcnn-examples.png' }})
{: style="width: 100%;" class="center"}
*Fig. 9. Predictions by Mask R-CNN on COCO test set. (Image source: [He et al., 2017](https://arxiv.org/pdf/1703.06870.pdf))*


### RoIAlign

The RoIAlign layer is designed to fix the location misalignment caused by quantization in the RoI pooling. RoIAlign removes the hash quantization, for example, by using x/16 instead of [x/16], so that the extracted features can be properly aligned with the input pixels. [Bilinear interpolation](https://en.wikipedia.org/wiki/Bilinear_interpolation) is used for computing the floating-point location values in the input.


![RoI Align]({{ '/assets/images/roi-align.png' }})
{: style="width: 640px;" class="center"}
*Fig. 10. A region of interest is mapped **accurately** from the original image onto the feature map without rounding up to integers. (Image source: [link](https://blog.athelas.com/a-brief-history-of-cnns-in-image-segmentation-from-r-cnn-to-mask-r-cnn-34ea83205de4))*


### Loss Function

The multi-task loss function of Mask R-CNN combines the loss of classification, localization and segmentation mask: $$ \mathcal{L} = \mathcal{L}_\text{cls} + \mathcal{L}_\text{box} + \mathcal{L}_\text{mask}$$, where $$\mathcal{L}_\text{cls}$$ and $$\mathcal{L}_\text{box}$$ are same as in Faster R-CNN.


The mask branch generates a mask of dimension m x m for each RoI and each class; K classes in total. Thus, the total output is of size $$K \cdot m^2$$. Because the model is trying to learn a mask for each class, there is no competition among classes for generating masks.

$$\mathcal{L}_\text{mask}$$ is defined as the average binary cross-entropy loss, only including k-th mask if the region is associated with the ground truth class k.

$$
\mathcal{L}_\text{mask} = - \frac{1}{m^2} \sum_{1 \leq i, j \leq m} \big[ y_{ij} \log \hat{y}^k_{ij} + (1-y_{ij}) \log (1- \hat{y}^k_{ij}) \big]
$$

where $$y_{ij}$$ is the label of a cell (i, j) in the true mask for the region of size m x m; $$\hat{y}_{ij}^k$$ is the predicted value of the same cell in the mask learned for the ground-truth class k.

## Summary of Models in the R-CNN family

Here I illustrate model designs of R-CNN, Fast R-CNN, Faster R-CNN and Mask R-CNN. You can track how one model evolves to the next version by comparing the small differences.

<img src="/assets/images/rcnn-family-summary.png" width="1308" height="516" align="middle" />


---
Cited as:
```
@article{
  title   = "R-CNN Family",
  author  = "Zhang, Xuanrui",
  journal = "noba1anc3.github.io/",
  year    = "2020",
  url     = "https://noba1anc3.github.io/2020/04/15/R-CNN-Family.html"
}
```

## Reference

[1] Ross Girshick, Jeff Donahue, Trevor Darrell, and Jitendra Malik. ["Rich feature hierarchies for accurate object detection and semantic segmentation."](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.pdf) In Proc. IEEE Conf. on computer vision and pattern recognition (CVPR), pp. 580-587. 2014.

[2] Ross Girshick. ["Fast R-CNN."](https://arxiv.org/pdf/1504.08083.pdf) In Proc. IEEE Intl. Conf. on computer vision, pp. 1440-1448. 2015.

[3] Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. ["Faster R-CNN: Towards real-time object detection with region proposal networks."](http://papers.nips.cc/paper/5638-faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks.pdf) In Advances in neural information processing systems (NIPS), pp. 91-99. 2015.

[4] Kaiming He, Georgia Gkioxari, Piotr Dollár, and Ross Girshick. ["Mask R-CNN."](https://arxiv.org/pdf/1703.06870.pdf) arXiv preprint arXiv:1703.06870, 2017.

[5] Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi. ["You only look once: Unified, real-time object detection."](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf) In Proc. IEEE Conf. on computer vision and pattern recognition (CVPR), pp. 779-788. 2016.

[6] ["A Brief History of CNNs in Image Segmentation: From R-CNN to Mask R-CNN"](https://blog.athelas.com/a-brief-history-of-cnns-in-image-segmentation-from-r-cnn-to-mask-r-cnn-34ea83205de4) by Athelas.

[7] Smooth L1 Loss: [https://github.com/rbgirshick/py-faster-rcnn/files/764206/SmoothL1Loss.1.pdf](https://github.com/rbgirshick/py-faster-rcnn/files/764206/SmoothL1Loss.1.pdf)

