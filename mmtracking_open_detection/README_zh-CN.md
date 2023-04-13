# MMTracking Open Detection

<div align=center>
<img src="https://user-images.githubusercontent.com/27466624/231666666-4f4c5696-df73-45cd-af04-758ea3806a82.png"/>
</div>

借助于开放目标检测，并利用运动信息（卡尔曼滤波器）来进行多目标跟踪。

受限于时间，目前只支持 GroundingDINO, GLIP 以及 Detic 结合 ByteTrack 方式进行跟踪

<div align="center">
<img src="https://github.com/zwhus/pictures/raw/main/bdd.gif">
<img src="https://github.com/zwhus/pictures/raw/main/demo.gif">
<img src="https://github.com/zwhus/pictures/raw/main/demo%2B(1).gif">
</div>

## 参数说明

`tracking_demo.py` 用于视频或者图片文件夹的多目标跟踪推理

本工程参考了 [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)，非常感谢！

#### demo文件的获取

```shell
cd playground
wget https://download.openmmlab.com/playground/mmtracking/tracking_demo.zip
```

## 基础环境安装

```shell
conda create -n mmtracking-sam python=3.8 -y
conda activate mmtracking-sam
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
git clone https://github.com/open-mmlab/playground.git
```

### 基础依赖安装

```shell
pip install -U openmim
mim install "mmcv>=2.0.0"

# 源码安装
git clone -b tracking https://github.com/open-mmlab/mmdetection.git
cd mmdetection; pip install -e .; cd ..
```

### Grounding Dino 环境安装

```shell
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO; pip install -e .; cd ..
```

### GLIP 环境安装

```shell
git clone https://github.com/microsoft/GLIP.git
cd GLIP
pip install einops shapely timm yacs tensorboardX ftfy prettytable pymongo
pip install transformers
python setup.py build develop --user
cd ..
```

### SAM 环境安装

```shell
pip install git+https://github.com/facebookresearch/segment-anything.git
```

## 模型推理演示

仅以 GroundingDINO 为例

### 多目标跟踪

```shell
cd mmtracking_open_detection

python tracking_demo.py "../tracking_demo/mot_challenge_track.mp4" "../GroundingDINO/groundingdino/config/GroundingDINO_SwinB.cfg.py" "../models/groundingdino_swinb_cogcoor.pth"  --text_prompt "person . rider . car . truck . bus . train . motorcycle . bicycle ." --out-dir "outputs/mot_challenge"

python tracking_demo.py "../tracking_demo/bdd_val_track" "../GroundingDINO/groundingdino/config/GroundingDINO_SwinB.cfg.py" "../models/groundingdino_swinb_cogcoor.pth"  --text_prompt "person . rider . car . truck . bus . train . motorcycle . bicycle ." --out-dir "outputs/bdd100k" --fps 30
```

### 多目标跟踪和分割

```shell
cd mmtracking_open_detection

python tracking_demo.py "../tracking_demo/bdd_val_track" "../GroundingDINO/groundingdino/config/GroundingDINO_SwinB.cfg.py" "../models/groundingdino_swinb_cogcoor.pth"  --text_prompt "person . rider . car . truck . bus . train . motorcycle . bicycle ." --out-dir "outputs/bdd100k" --fps 30 --mots
```
