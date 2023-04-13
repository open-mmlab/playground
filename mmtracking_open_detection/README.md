# MMTracking-Grouding

借助于开放集目标检测 Open-Set Object Detection, 利用运动信息（卡尔曼滤波器）来进行多目标跟踪。

受限于时间，目前只支持GroundingDINO, GLIP以及Detic 结合Byte 方式进行跟踪

## 参数说明

`tracking_demo.py` 用于视频或者图片文件夹的多目标跟踪推理

本工程参考了 [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)，非常感谢！

#### demo文件的获取
```shell
cd mmsam
wget https://download.openmmlab.com/playground/mmtracking/tracking_demo.zip

```

## 基础环境安装

```shell
conda create -n mmdet-sam python=3.8 -y
conda activate mmdet-sam
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
git clone https://github.com/open-mmlab/playground.git
cd playground
```

### 基础依赖安装

```shell
pip install -U openmim
mim install "mmcv>=2.0.0"

# 源码安装
git clone -b tracking https://github.com/open-mmlab/mmdetection.git
cd mmdetection; pip install -e .; cd ..

```

### Grouding Dino环境安装

```shell
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO; pip install -e .; cd ..
```

### GLIP环境安装
```shell
git clone https://github.com/microsoft/GLIP.git
cd GLIP
pip install einops shapely timm yacs tensorboardX ftfy prettytable pymongo
pip install transformers 
python setup.py build develop --user
cd ..
```


## 模型推理演示

仅以GroundingDINO为例
```shell
cd mmtracking_open_detection

python tracking_demo.py "../tracking_demo/mot_challenge_track.mp4" "../GroundingDINO/groundingdino/config/GroundingDINO_SwinB.cfg.py" "../models/groundingdino_swinb_cogcoor.pth"  --text_prompt "person . rider . car . truck . bus . train . motorcycle . bicycle ." --out-dir "outputs/mot_challenge"

python tracking_demo.py "../tracking_demo/bdd_val_track" "../GroundingDINO/groundingdino/config/GroundingDINO_SwinB.cfg.py" "../models/groundingdino_swinb_cogcoor.pth"  --text_prompt "person . rider . car . truck . bus . train . motorcycle . bicycle ." --out-dir "outputs/mot_challenge"
```
