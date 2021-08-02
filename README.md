# ARM3D:  Attention-based Relation Module for Indoor 3D Object Detection


## Introduction

<div align=center>
<img src="./example.png" width="400" height="" />
</div>

ARM3D is a plug-and-play module which can be conveniently applied to different 3D object detectors. It provides precise and useful relation context to help 3D detectors locate and classify objects more accurately and robustly. We implement two applications on VoteNet and MLCVNet. We e valuate its improved performance on ScanNetV2 and SUN RGB-D dataset.

## Usage
ARM3D can be widely applicable on different 3D detection frameworks. For further information about our implemention on two frameworks, please refer to the sub-directories: [MLCVNet+ARM3D](./MLCVNet-ARM3D) and [VoteNet+ARM3D](./VoteNet-ARM3D).


## Acknowledgemets
This code largely benefits from excellent works [PointCNN](https://github.com/yangyanli/PointCNN) and [VoteNet](https://github.com/facebookresearch/votenet) and repositories, please also consider cite [MLCVNet](https://github.com/NUAAXQ/MLCVNet) and [VoteNet](https://arxiv.org/pdf/1904.09664.pdf) if you use this code.