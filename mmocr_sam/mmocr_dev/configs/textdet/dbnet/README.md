# DBNet

> [Real-time Scene Text Detection with Differentiable Binarization](https://arxiv.org/abs/1911.08947)

<!-- [ALGORITHM] -->

## Abstract

Recently, segmentation-based methods are quite popular in scene text detection, as the segmentation results can more accurately describe scene text of various shapes such as curve text. However, the post-processing of binarization is essential for segmentation-based detection, which converts probability maps produced by a segmentation method into bounding boxes/regions of text. In this paper, we propose a module named Differentiable Binarization (DB), which can perform the binarization process in a segmentation network. Optimized along with a DB module, a segmentation network can adaptively set the thresholds for binarization, which not only simplifies the post-processing but also enhances the performance of text detection. Based on a simple segmentation network, we validate the performance improvements of DB on five benchmark datasets, which consistently achieves state-of-the-art results, in terms of both detection accuracy and speed. In particular, with a light-weight backbone, the performance improvements by DB are significant so that we can look for an ideal tradeoff between detection accuracy and efficiency. Specifically, with a backbone of ResNet-18, our detector achieves an F-measure of 82.8, running at 62 FPS, on the MSRA-TD500 dataset.

<div align=center>
<img src="https://user-images.githubusercontent.com/22607038/142791306-0da6db2a-20a6-4a68-b228-64ff275f67b3.png"/>
</div>

## Results and models

### SynthText

|                                  Method                                   | Backbone | Training set | #iters  |                                               Download                                               |
| :-----------------------------------------------------------------------: | :------: | :----------: | :-----: | :--------------------------------------------------------------------------------------------------: |
| [DBNet_r18](/configs/textdet/dbnet/dbnet_resnet18_fpnc_100k_synthtext.py) | ResNet18 |  SynthText   | 100,000 | [model](https://download.openmmlab.com/mmocr/textdet/dbnet/dbnet_resnet18_fpnc_100k_synthtext/dbnet_resnet18_fpnc_100k_synthtext-2e9bf392.pth) \| [log](https://download.openmmlab.com/mmocr/textdet/dbnet/dbnet_resnet18_fpnc_100k_synthtext/20221214_150351.log) |

### ICDAR2015

|             Method             |             Backbone             |             Pretrained Model             |  Training set   |    Test set    | #epochs | Test size | Precision | Recall | Hmean  |             Download             |
| :----------------------------: | :------------------------------: | :--------------------------------------: | :-------------: | :------------: | :-----: | :-------: | :-------: | :----: | :----: | :------------------------------: |
| [DBNet_r18](/configs/textdet/dbnet/dbnet_resnet18_fpnc_1200e_icdar2015.py) |             ResNet18             |                    -                     | ICDAR2015 Train | ICDAR2015 Test |  1200   |    736    |  0.8853   | 0.7583 | 0.8169 | [model](https://download.openmmlab.com/mmocr/textdet/dbnet/dbnet_resnet18_fpnc_1200e_icdar2015/dbnet_resnet18_fpnc_1200e_icdar2015_20220825_221614-7c0e94f2.pth) \| [log](https://download.openmmlab.com/mmocr/textdet/dbnet/dbnet_resnet18_fpnc_1200e_icdar2015/20220825_221614.log) |
| [DBNet_r50](/configs/textdet/dbnet/dbnet_resnet50_1200e_icdar2015.py) |             ResNet50             |                    -                     | ICDAR2015 Train | ICDAR2015 Test |  1200   |   1024    |  0.8744   | 0.8276 | 0.8504 | [model](https://download.openmmlab.com/mmocr/textdet/dbnet/dbnet_resnet50_1200e_icdar2015/dbnet_resnet50_1200e_icdar2015_20221102_115917-54f50589.pth) \| [log](https://download.openmmlab.com/mmocr/textdet/dbnet/dbnet_resnet50_1200e_icdar2015/20221102_115917.log) |
| [DBNet_r50dcn](/configs/textdet/dbnet/dbnet_resnet50-dcnv2_fpnc_1200e_icdar2015.py) |           ResNet50-DCN           | [Synthtext](https://download.openmmlab.com/mmocr/textdet/dbnet/tmp_1.0_pretrain/dbnet_r50dcnv2_fpnc_sbn_2e_synthtext_20210325-ed322016.pth) | ICDAR2015 Train | ICDAR2015 Test |  1200   |   1024    |  0.8784   | 0.8315 | 0.8543 | [model](https://download.openmmlab.com/mmocr/textdet/dbnet/dbnet_resnet50-dcnv2_fpnc_1200e_icdar2015/dbnet_resnet50-dcnv2_fpnc_1200e_icdar2015_20220828_124917-452c443c.pth) \| [log](https://download.openmmlab.com/mmocr/textdet/dbnet/dbnet_resnet50-dcnv2_fpnc_1200e_icdar2015/20220828_124917.log) |
| [DBNet_r50-oclip](/configs/textdet/dbnet/dbnet_resnet50-oclip_1200e_icdar2015.py) | [ResNet50-oCLIP](https://download.openmmlab.com/mmocr/backbone/resnet50-oclip-7ba0c533.pth) |                    -                     | ICDAR2015 Train | ICDAR2015 Test |  1200   |   1024    |  0.9052   | 0.8272 | 0.8644 | [model](https://download.openmmlab.com/mmocr/textdet/dbnet/dbnet_resnet50-oclip_1200e_icdar2015/dbnet_resnet50-oclip_1200e_icdar2015_20221102_115917-bde8c87a.pth) \| [log](https://download.openmmlab.com/mmocr/textdet/dbnet/dbnet_resnet50-oclip_1200e_icdar2015/20221102_115917.log) |

### Total Text

|                         Method                         | Backbone | Pretrained Model |  Training set   |    Test set    | #epochs | Test size | Precision | Recall | Hmean  |                         Download                         |
| :----------------------------------------------------: | :------: | :--------------: | :-------------: | :------------: | :-----: | :-------: | :-------: | :----: | :----: | :------------------------------------------------------: |
| [DBNet_r18](/configs/textdet/dbnet/dbnet_resnet18_fpnc_1200e_totaltext.py) | ResNet18 |        -         | Totaltext Train | Totaltext Test |  1200   |    736    |  0.8640   | 0.7770 | 0.8182 | [model](https://download.openmmlab.com/mmocr/textdet/dbnet/dbnet_resnet18_fpnc_1200e_totaltext/dbnet_resnet18_fpnc_1200e_totaltext-3ed3233c.pth) \| [log](https://download.openmmlab.com/mmocr/textdet/dbnet/dbnet_resnet18_fpnc_1200e_totaltext/20221219_201038.log) |

## Citation

```bibtex
@article{Liao_Wan_Yao_Chen_Bai_2020,
    title={Real-Time Scene Text Detection with Differentiable Binarization},
    journal={Proceedings of the AAAI Conference on Artificial Intelligence},
    author={Liao, Minghui and Wan, Zhaoyi and Yao, Cong and Chen, Kai and Bai, Xiang},
    year={2020},
    pages={11474-11481}}
```
