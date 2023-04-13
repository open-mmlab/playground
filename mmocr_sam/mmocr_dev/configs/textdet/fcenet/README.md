# FCENet

> [Fourier Contour Embedding for Arbitrary-Shaped Text Detection](https://arxiv.org/abs/2104.10442)

<!-- [ALGORITHM] -->

## Abstract

One of the main challenges for arbitrary-shaped text detection is to design a good text instance representation that allows networks to learn diverse text geometry variances. Most of existing methods model text instances in image spatial domain via masks or contour point sequences in the Cartesian or the polar coordinate system. However, the mask representation might lead to expensive post-processing, while the point sequence one may have limited capability to model texts with highly-curved shapes. To tackle these problems, we model text instances in the Fourier domain and propose one novel Fourier Contour Embedding (FCE) method to represent arbitrary shaped text contours as compact signatures. We further construct FCENet with a backbone, feature pyramid networks (FPN) and a simple post-processing with the Inverse Fourier Transformation (IFT) and Non-Maximum Suppression (NMS). Different from previous methods, FCENet first predicts compact Fourier signatures of text instances, and then reconstructs text contours via IFT and NMS during test. Extensive experiments demonstrate that FCE is accurate and robust to fit contours of scene texts even with highly-curved shapes, and also validate the effectiveness and the good generalization of FCENet for arbitrary-shaped text detection. Furthermore, experimental results show that our FCENet is superior to the state-of-the-art (SOTA) methods on CTW1500 and Total-Text, especially on challenging highly-curved text subset.

<div align=center>
<img src="https://user-images.githubusercontent.com/22607038/142791859-1b0ebde4-b151-4c25-ba1b-f354bd8ddc8c.png"/>
</div>

## Results and models

### CTW1500

|                 Method                 |                 Backbone                  | Pretrained Model | Training set  |   Test set   | #epochs |  Test size  | Precision | Recall | Hmean  |                 Download                  |
| :------------------------------------: | :---------------------------------------: | :--------------: | :-----------: | :----------: | :-----: | :---------: | :-------: | :----: | :----: | :---------------------------------------: |
| [FCENet_r50dcn](/configs/textdet/fcenet/fcenet_resnet50-dcnv2_fpn_1500e_ctw1500.py) |             ResNet50 + DCNv2              |        -         | CTW1500 Train | CTW1500 Test |  1500   | (736, 1080) |  0.8689   | 0.8296 | 0.8488 | [model](https://download.openmmlab.com/mmocr/textdet/fcenet/fcenet_resnet50-dcnv2_fpn_1500e_ctw1500/fcenet_resnet50-dcnv2_fpn_1500e_ctw1500_20220825_221510-4d705392.pth) \| [log](https://download.openmmlab.com/mmocr/textdet/fcenet/fcenet_resnet50-dcnv2_fpn_1500e_ctw1500/20220825_221510.log) |
| [FCENet_r50-oclip](/configs/textdet/fcenet/fcenet_resnet50-oclip-dcnv2_fpn_1500e_ctw1500.py) | [ResNet50-oCLIP](https://download.openmmlab.com/mmocr/backbone/resnet50-oclip-7ba0c533.pth) |        -         | CTW1500 Train | CTW1500 Test |  1500   | (736, 1080) |  0.8383   | 0.801  | 0.8192 | [model](https://download.openmmlab.com/mmocr/textdet/fcenet/fcenet_resnet50-oclip_fpn_1500e_ctw1500/fcenet_resnet50-oclip_fpn_1500e_ctw1500_20221102_121909-101df7e6.pth) \| [log](https://download.openmmlab.com/mmocr/textdet/fcenet/fcenet_resnet50-oclip_fpn_1500e_ctw1500/20221102_121909.log) |

### ICDAR2015

|                        Method                         |    Backbone    | Pretrained Model | Training set | Test set  | #epochs |  Test size   | Precision | Recall | Hmean  |                         Download                         |
| :---------------------------------------------------: | :------------: | :--------------: | :----------: | :-------: | :-----: | :----------: | :-------: | :----: | :----: | :------------------------------------------------------: |
| [FCENet_r50](/configs/textdet/fcenet/fcenet_resnet50_fpn_1500e_icdar2015.py) |    ResNet50    |        -         |  IC15 Train  | IC15 Test |  1500   | (2260, 2260) |  0.8243   | 0.8834 | 0.8528 | [model](https://download.openmmlab.com/mmocr/textdet/fcenet/fcenet_resnet50_fpn_1500e_icdar2015/fcenet_resnet50_fpn_1500e_icdar2015_20220826_140941-167d9042.pth) \| [log](https://download.openmmlab.com/mmocr/textdet/fcenet/fcenet_resnet50_fpn_1500e_icdar2015/20220826_140941.log) |
| [FCENet_r50-oclip](/configs/textdet/fcenet/fcenet_resnet50-oclip_fpn_1500e_icdar2015.py) | ResNet50-oCLIP |        -         |  IC15 Train  | IC15 Test |  1500   | (2260, 2260) |  0.9176   | 0.8098 | 0.8604 | [model](https://download.openmmlab.com/mmocr/textdet/fcenet/fcenet_resnet50-oclip_fpn_1500e_icdar2015/fcenet_resnet50-oclip_fpn_1500e_icdar2015_20221101_150145-5a6fc412.pth) \| [log](https://download.openmmlab.com/mmocr/textdet/fcenet/fcenet_resnet50-oclip_fpn_1500e_icdar2015/20221101_150145.log) |

### Total Text

|                        Method                         | Backbone | Pretrained Model |  Training set   |    Test set    | #epochs |  Test size  | Precision | Recall | Hmean  |                        Download                         |
| :---------------------------------------------------: | :------: | :--------------: | :-------------: | :------------: | :-----: | :---------: | :-------: | :----: | :----: | :-----------------------------------------------------: |
| [FCENet_r50](/configs/textdet/fcenet/fcenet_resnet50_fpn_1500e_totaltext.py) | ResNet50 |        -         | Totaltext Train | Totaltext Test |  1500   | (1280, 960) |  0.8485   | 0.7810 | 0.8134 | [model](https://download.openmmlab.com/mmocr/textdet/fcenet/fcenet_resnet50_fpn_1500e_totaltext/fcenet_resnet50_fpn_1500e_totaltext-91bd37af.pth) \| [log](https://download.openmmlab.com/mmocr/textdet/fcenet/fcenet_resnet50_fpn_1500e_totaltext/20221219_201107.log) |

## Citation

```bibtex
@InProceedings{zhu2021fourier,
      title={Fourier Contour Embedding for Arbitrary-Shaped Text Detection},
      author={Yiqin Zhu and Jianyong Chen and Lingyu Liang and Zhanghui Kuang and Lianwen Jin and Wayne Zhang},
      year={2021},
      booktitle = {CVPR}
      }
```
