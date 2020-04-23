---
layout: post
---

This article focuses on one-stage models for fast detection, including SSD, RetinaNet, and models in the YOLO family. These models skip the explicit region proposal stage but apply the detection directly on dense sampled areas.

In [Part 3](https://noba1anc3.github.io/2020/04/15/R-CNN-Family.html), we have reviewed models in the R-CNN family. All of them are region-based object detection algorithms. They can achieve high accuracy but could be too slow for certain applications such as autonomous driving. In Part 4, we only focus on fast object detection models, including SSD, RetinaNet, and models in the YOLO family.

Links to all the posts in the series: 
[[Part 1]()]
[[Part 2](https://noba1anc3.github.io/2020/04/22/CNN,-DPM-and-Overfeat.html)]
[[Part 3](https://noba1anc3.github.io/2020/04/15/R-CNN-Family.html)]
[[Part 4](https://noba1anc3.github.io/2020/04/23/Fast-Detection-Models.html)].

{: class="table-of-content"}
* TOC
{:toc}


## Two-stage vs One-stage Detectors

Models in the R-CNN family are all region-based. The detection happens in two stages: (1) First, the model proposes a set of regions of interests by select search or regional proposal network. The proposed regions are sparse as the potential bounding box candidates can be infinite. (2) Then a classifier only processes the region candidates.

The other different approach skips the region proposal stage and runs detection directly over a dense sampling of possible locations. This is how a one-stage object detection algorithm works. This is faster and simpler, but might potentially drag down the performance a bit.

All the models introduced in this post are one-stage detectors.


## YOLO: You Only Look Once

The **YOLO** model (**"You Only Look Once"**; [Redmon et al., 2016](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf)) is the very first attempt at building a fast real-time object detector. Because YOLO does not undergo the region proposal step and only predicts over a limited number of bounding boxes, it is able to do inference super fast.


### Workflow

1. **Pre-train** a CNN network on image classification task.

2. Split an image into $$S \times S$$ cells. If an object's center falls into a cell, that cell is "responsible" for detecting the existence of that object. Each cell predicts (a) the location of $$B$$ bounding boxes, (b) a confidence score, and (c) a probability of object class conditioned on the existence of an object in the bounding box.
<br/>
<br/>
- The **coordinates** of bounding box are defined by a tuple of 4 values, (center x-coord, center y-coord, width, height) --- $$(x, y, w, h)$$, where $$x$$ and $$y$$ are set to be offset of a cell location. Moreover, $$x$$, $$y$$, $$w$$ and $$h$$ are normalized by the image width and height, and thus all between (0, 1].
- A **confidence score** indicates the likelihood that the cell contains an object: `Pr(containing an object) x IoU(pred, truth)`; where `Pr` = probability and `IoU` = interaction under union.
- If the cell contains an object, it predicts a **probability** of this object belonging to every class $$C_i, i=1, \dots, K$$: `Pr(the object belongs to the class C_i | containing an object)`. At this stage, the model only predicts one set of class probabilities per cell, regardless of the number of bounding boxes, $$B$$.
- In total, one image contains $$S \times S \times B$$ bounding boxes, each box corresponding to 4 location predictions, 1 confidence score, and K conditional probabilities for object classification. The total prediction values for one image is $$S \times S \times (5B + K)$$, which is the tensor shape of the final conv layer of the model.
<br/>
<br/>
3. The final layer of the pre-trained CNN is modified to output a prediction tensor of size $$S \times S \times (5B + K)$$.


![YOLO workflow]({{ '/assets/images/yolo.png' }})
{: class="center"}
*Fig. 1. The workflow of YOLO model. (Image source: [original paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf))*


### Network Architecture

The base model is similar to [GoogLeNet](https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf) with inception module replaced by 1x1 and 3x3 conv layers. The final prediction of shape $$S \times S \times (5B + K)$$ is produced by two fully connected layers over the whole conv feature map.

![YOLO architecture]({{ '/assets/images/yolo-network-architecture.png' }})
{: class="center"}
*Fig. 2. The network architecture of YOLO.*

