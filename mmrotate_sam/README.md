# MMRotate-SAM

[中文文档](README_CN.md)

<div align=center>
<img src="https://user-images.githubusercontent.com/79644233/231636420-8b7f81f3-51d2-439c-87cc-6f7eebd32193.png"/>
</div>

The project folder holds codes related to MMRotate and SAM.

Script Descriptions:
1. `eval_zero-shot-oriented-detection_dota.py` implement Zero-shot Oriented Object Detection with SAM. It prompts SAM with predicted boxes from a horizontal object detector. 
2. `demo_zero-shot-oriented-detection.py` inference single image for Zero-shot Oriented Object Detection with SAM.
3. `data_builder` holds configuration information and process of dataset, dataloader.

The project is refer to [sam-mmrotate](https://github.com/Li-Qingyun/sam-mmrotate).

## Installation

```shell
conda create -n mmrotate-sam python=3.8 -y
conda activate mmrotate-sam
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html

pip install openmim
mim install mmengine 'mmcv>=2.0.0rc0' 'mmrotate>=1.0.0rc0'

pip install git+https://github.com/facebookresearch/segment-anything.git
pip install opencv-python pycocotools matplotlib onnxruntime onnx
``` 

## Usage

1. Inference MMRotate-SAM with a single image and obtain visualization result.
```shell
python demo_zero-shot-oriented-detection.py \
  data/split_ss_dota/test/images/P0006__1024__0___0.png \
  configs/rotated_fcos/rotated-fcos-hbox-le90_r50_fpn_1x_dota.py \
  rotated_fcos_sep_angle_r50_fpn_1x_dota_le90-0be71a0c.pth \
  --sam-type "vit_b" --sam-weight sam_vit_b_01ec64.pth --out-path output.png
```

<div align=center>
<img src="https://user-images.githubusercontent.com/79644233/231568599-58694ec9-a3b1-44a4-833f-74cfb4d4ca45.png"/>
</div>

2. Evaluate the quantitative evaluation metric on DOTA data set.
```shell
python eval_zero-shot-oriented-detection_dota.py \
  configs/rotated_fcos/rotated-fcos-hbox-le90_r50_fpn_1x_dota.py \
  rotated_fcos_sep_angle_r50_fpn_1x_dota_le90-0be71a0c.pth \
  --sam-type "vit_b" --sam-weight sam_vit_b_01ec64.pth
```
