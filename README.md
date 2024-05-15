# FreeGaze: Resource-efficient Gaze Estimation via Frequency-domain Contrastive Learning

This repository contains the introductions and the codes for EWSN 2023 paper [FreeGaze: Resource-efficient Gaze Estimation via Frequency-domain Contrastive Learning](https://arxiv.org/abs/2209.06692) by [Lingyu Du](https://github.com/LingyuDu) and [Guohao Lan](https://guohao.netlify.app/). If you have any questions, please send an email to Lingyu.Du AT tudelft.nl.

## Description

Gaze estimation is of great importance to many scientific fields and daily applications, ranging from fundamental research in cognitive psychology to attention-aware mobile systems. While recent advancements in deep learning have yielded remarkable successes in building highly accurate gaze estimation systems, the associated high computational cost and the reliance on large-scale labeled gaze data for supervised learning place challenges on the practical use of existing solutions. To move beyond these limitations, we present FreeGaze, a resource-efficient framework for unsupervised gaze representation learning. FreeGaze incorporates the frequency domain gaze estimation and the contrastive gaze representation learning in its design. The former significantly alleviates the computational burden in both system calibration and gaze estimation, and dramatically reduces the system latency; while the latter overcomes the data labeling hurdle of existing supervised learning-based counterparts, and ensures efficient gaze representation learning in the absence of gaze label. Our evaluation on two gaze estimation datasets shows that FreeGaze can achieve comparable gaze estimation accuracy with existing supervised learning-based approach, while enabling up to 6.81 and 1.67 times speedup in system calibration and gaze estimation, respectively.

## Getting Started

### Dependencies

* Tensorflow-gpu 2.9.0, jpeg2dct
* ex. Ubuntu 20.04

