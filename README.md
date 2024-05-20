# FreeGaze: Resource-efficient Gaze Estimation via Frequency-domain Contrastive Learning

This repository contains the introductions and the codes for EWSN 2023 paper [FreeGaze: Resource-efficient Gaze Estimation via Frequency-domain Contrastive Learning](https://arxiv.org/abs/2209.06692) by [Lingyu Du](https://github.com/LingyuDu) and [Guohao Lan](https://guohao.netlify.app/). If you have any questions, please send an email to Lingyu.Du AT tudelft.nl.

## Description

Gaze estimation is of great importance to many scientific fields and daily applications, ranging from fundamental research in cognitive psychology to attention-aware mobile systems. While recent advancements in deep learning have yielded remarkable successes in building highly accurate gaze estimation systems, the associated high computational cost and the reliance on large-scale labeled gaze data for supervised learning place challenges on the practical use of existing solutions. To move beyond these limitations, we present FreeGaze, a resource-efficient framework for unsupervised gaze representation learning. FreeGaze incorporates the frequency domain gaze estimation and the contrastive gaze representation learning in its design. The former significantly alleviates the computational burden in both system calibration and gaze estimation, and dramatically reduces the system latency; while the latter overcomes the data labeling hurdle of existing supervised learning-based counterparts, and ensures efficient gaze representation learning in the absence of gaze label. Our evaluation on two gaze estimation datasets shows that FreeGaze can achieve comparable gaze estimation accuracy with existing supervised learning-based approach, while enabling up to 6.81 and 1.67 times speedup in system calibration and gaze estimation, respectively.

## Getting Started

### Dependencies

* Tensorflow-gpu 2.9.0, jpeg2dct
* ex. Ubuntu 20.04

### Dataset preparation
In our implementation, to efficiently obtain DCT coefficents of the original RGB images, we proprocess the dataset by saving all the images in the format of .jpg. We then use jpeg2dct to directly read DCT coefficients from a jpg image in the training and testing stages.


## System Overview
The overview design of FreeGaze is shown in the following figure, which includes three stages: the self-supervised pre-training stage, the supervised calibration stage, and the deployment stage. In the self-supervised pre-training stage, FreeGaze takes unlabeled facial images as inputs to pre-train a gaze embedding network for unsupervised gaze representation learning. Next, the pre-trained gaze embedding network is transferred to the supervised calibration stage, and serves as a feature extractor for the downstream gaze estimation task. Specifically, taking a small number of labeled facial images from the targeted subject as input, we first leverage the frequency-domain image processing module to obtain the DCT coefficients of the original RGB images. Then, we fine-tune the pre-trained gaze embedding network and the gaze estimator for subject-specific gaze estimation. Note that subject-specific calibration is essential to ensure good gaze-estimation accuracy, even for high-end eye-tracking systems. Finally, in the deployment stage, the fine-tuned gaze embedding network and the gaze estimator are used for run-time gaze estimation. 

<img src="https://github.com/FreeGaze/FreeGaze-Source/blob/main/figures/overview.png" alt="My Image" width="500"/>


## Frequency-domain Gaze Estimation

To reduce the latency for gaze estimation system in both calibration and inference stages, we devise the frequency-domain gaze estimation. It leverages the feature extraction capability of the discrete cosine transform (DCT) and takes the frequency-domain DCT coefficients of the original RGB image as inputs for gaze estimation. Moreover, motivated by the fact that the critical content-defining information of the image is concentrated in the low end of the frequency spectrum, whereas signals in the high-frequency endare mostly trivial and are associated with noise, we further exploit the spectral compaction property of DCT toaggressively compact the essential perceptual information inthe RGB image into a few DCT coefficients. The pipeline of frequency-domain image processing is shown as below:

<img src="https://github.com/FreeGaze/FreeGaze-Source/blob/main/figures/dctProcessing.png" alt="My Image" width="500"/>

## Frequency-domain Contrastive Gaze Representation Learning

To overcome the data labeling hurdle of existing supervised gaze estimation systems, we propose a contrastive learning (CL)-based framework that leverages unlabeled facial images for gaze representation learning. The conventional CL are ill-suited for gaze estimation, as they focus on learning general representations that are more related to the appearance and the identity of the subjects. To resolve this challenge, we introduce a set of optimizations to enable contrastive gaze representation learning. Specifically, we devise the subject-specific negative pair sampling strategy to encourage the learning of gaze-related features and design the gaze-specific data augmentation to ensure the gaze consistency during the contrastive learning process. The two techniques lead to significant improvements in gaze estimation when compared to the conventional unsupervised method. The pipeline of the proposed frequency-domain contrastive gaze representation learning framework is shown as below:

<img src="https://github.com/FreeGaze/FreeGaze-Source/blob/main/figures/frequencyCL.png" alt="My Image" width="500"/>

## Citation

If you use this code in your research, please cite our paper:

<div style="border: 1px solid #ccc; padding: 10px; background-color: #f9f9f9; border-radius: 5px;">
<pre>
@article{du2022freegaze,
  title={Freegaze: Resource-efficient gaze estimation via frequency domain contrastive learning},
  author={Du, Lingyu and Lan, Guohao},
  journal={arXiv preprint arXiv:2209.06692},
  year={2022}
}
</pre>
</div>

