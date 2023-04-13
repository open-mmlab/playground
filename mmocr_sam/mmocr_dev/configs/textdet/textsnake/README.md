# Textsnake

> [TextSnake: A Flexible Representation for Detecting Text of Arbitrary Shapes](https://arxiv.org/abs/1807.01544)

<!-- [ALGORITHM] -->

## Abstract

Driven by deep neural networks and large scale datasets, scene text detection methods have progressed substantially over the past years, continuously refreshing the performance records on various standard benchmarks. However, limited by the representations (axis-aligned rectangles, rotated rectangles or quadrangles) adopted to describe text, existing methods may fall short when dealing with much more free-form text instances, such as curved text, which are actually very common in real-world scenarios. To tackle this problem, we propose a more flexible representation for scene text, termed as TextSnake, which is able to effectively represent text instances in horizontal, oriented and curved forms. In TextSnake, a text instance is described as a sequence of ordered, overlapping disks centered at symmetric axes, each of which is associated with potentially variable radius and orientation. Such geometry attributes are estimated via a Fully Convolutional Network (FCN) model. In experiments, the text detector based on TextSnake achieves state-of-the-art or comparable performance on Total-Text and SCUT-CTW1500, the two newly published benchmarks with special emphasis on curved text in natural images, as well as the widely-used datasets ICDAR 2015 and MSRA-TD500. Specifically, TextSnake outperforms the baseline on Total-Text by more than 40% in F-measure.

<div align=center>
<img src="https://user-images.githubusercontent.com/22607038/142795949-2525ead4-865b-4762-baaa-e977cfd6ac66.png"/>
</div>

## Results and models

### CTW1500

|                 Method                  |                 BackBone                  | Pretrained Model | Training set  |   Test set   | #epochs | Test size | Precision | Recall | Hmean  |                  Download                  |
| :-------------------------------------: | :---------------------------------------: | :--------------: | :-----------: | :----------: | :-----: | :-------: | :-------: | :----: | :----: | :----------------------------------------: |
| [TextSnake](/configs/textdet/textsnake/textsnake_resnet50_fpn-unet_1200e_ctw1500.py) |                 ResNet50                  |        -         | CTW1500 Train | CTW1500 Test |  1200   |    736    |  0.8535   | 0.8052 | 0.8286 | [model](https://download.openmmlab.com/mmocr/textdet/textsnake/textsnake_resnet50_fpn-unet_1200e_ctw1500/textsnake_resnet50_fpn-unet_1200e_ctw1500_20220825_221459-c0b6adc4.pth) \| [log](https://download.openmmlab.com/mmocr/textdet/textsnake/textsnake_resnet50_fpn-unet_1200e_ctw1500/20220825_221459.log) |
| [TextSnake_r50-oclip](/configs/textdet/textsnake/textsnake_resnet50-oclip_fpn-unet_1200e_ctw1500.py) | [ResNet50-oCLIP](https://download.openmmlab.com/mmocr/backbone/resnet50-oclip-7ba0c533.pth) |        -         | CTW1500 Train | CTW1500 Test |  1200   |    736    |  0.8869   | 0.8215 | 0.8529 | [model](https://download.openmmlab.com/mmocr/textdet/textsnake/textsnake_resnet50-oclip_fpn-unet_1200e_ctw1500/textsnake_resnet50-oclip_fpn-unet_1200e_ctw1500_20221101_134814-a216e5b2.pth) \| [log](https://download.openmmlab.com/mmocr/textdet/textsnake/textsnake_resnet50-oclip_fpn-unet_1200e_ctw1500/20221101_134814.log) |

## Citation

```bibtex
@article{long2018textsnake,
  title={TextSnake: A Flexible Representation for Detecting Text of Arbitrary Shapes},
  author={Long, Shangbang and Ruan, Jiaqiang and Zhang, Wenjie and He, Xin and Wu, Wenhao and Yao, Cong},
  booktitle={ECCV},
  pages={20-36},
  year={2018}
}
```
