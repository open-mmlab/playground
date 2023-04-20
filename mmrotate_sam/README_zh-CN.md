# MMRotate-SAM

<div align=center>
<img src="https://user-images.githubusercontent.com/27466624/231659969-adf7dd4d-fcec-4677-9105-aa72b2ced00f.PNG"/>
</div>

本工程目录存放 MMRotate 和 SAM 相关的代码。

脚本说明：

1. `eval_zero-shot-oriented-detection_dota.py` 实现了 SAM 的 Zero-shot Oriented Object Detection。在 SAM 前级联水平框检测器（当使用旋转框检测器时，取旋转框的最小水平外接矩作为水平框输出），将检测器输出的边界框作为 prompt 输入 SAM 中，输出掩码的最小有向外接矩即为对应目标的旋转框。
2. `demo_zero-shot-oriented-detection.py` 对单张图片进行 SAM 的 Zero-shot Oriented Object Detection 推理。
3. `data_builder` 存放数据集、数据加载器的配置信息以及配置过程。

本工程参考了 [sam-mmrotate](https://github.com/Li-Qingyun/sam-mmrotate)

## 环境安装

```shell
conda create -n mmrotate-sam python=3.8 -y
conda activate mmrotate-sam
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html

pip install openmim
mim install mmengine 'mmcv>=2.0.0rc0' 'mmrotate>=1.0.0rc0'

pip install git+https://github.com/facebookresearch/segment-anything.git
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```

注意：如果本地还没有 MMRotate 的 repo 代码，也可以使用以下方式源码安装 MMRotate：

```shell
git clone https://github.com/open-mmlab/mmrotate.git
cd mmrotate; pip install -e .; cd ..
```

## 使用方式

1. 在单张图上推理检测器级联 SAM 的旋转检测结果，获得可视化结果图。

```shell
# 下载权重
cd mmrotate_sam

mkdir ../models
wget -P ../models https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
wget -P ../models https://download.openmmlab.com/mmrotate/v0.1.0/rotated_fcos/rotated_fcos_sep_angle_r50_fpn_1x_dota_le90/rotated_fcos_sep_angle_r50_fpn_1x_dota_le90-0be71a0c.pth

# demo
python demo_zero-shot-oriented-detection.py \
    ../mmrotate/data/split_ss_dota/test/images/P0006__1024__0___0.png \
    ../mmrotate/configs/rotated_fcos/rotated-fcos-hbox-le90_r50_fpn_1x_dota.py \
    ../models/rotated_fcos_sep_angle_r50_fpn_1x_dota_le90-0be71a0c.pth \
    --sam-type "vit_b" --sam-weight ../models/sam_vit_b_01ec64.pth --out-path output.png
```

<div align=center>
<img src="https://user-images.githubusercontent.com/79644233/231568599-58694ec9-a3b1-44a4-833f-74cfb4d4ca45.png"/>
</div>

2. 在 DOTA 数据集上对检测器级联 SAM 的旋转检测结果进行定量评估。

```shell
python eval_zero-shot-oriented-detection_dota.py \
    ../mmrotate/configs/rotated_fcos/rotated-fcos-hbox-le90_r50_fpn_1x_dota.py \
    ../models/rotated_fcos_sep_angle_r50_fpn_1x_dota_le90-0be71a0c.pth \
    --sam-type "vit_b" --sam-weight ../models/sam_vit_b_01ec64.pth
```
