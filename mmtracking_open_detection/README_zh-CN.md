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

## 基础环境安装

```shell
conda create -n mmtracking-sam python=3.8 -y
conda activate mmtracking-sam
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
git clone https://github.com/open-mmlab/playground.git
```

### MMDet 环境安装

```shell
pip install -U openmim
mim install "mmcv>=2.0.0"

# 源码安装
cd playground
git clone -b tracking https://github.com/open-mmlab/mmdetection.git
cd mmdetection; pip install -e .; cd ..
pip install lap seaborn
```

### Grounding Dino 环境安装

```shell
cd playground
pip install git+https://github.com/IDEA-Research/GroundingDINO.git
```
如果由于网络原因无法使用pip方法进行下载，可以采取以下方法。
```
git clone git+https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
python setup.py install
```

### GLIP 环境安装

```shell
cd playground
pip install einops shapely timm yacs tensorboardX ftfy prettytable pymongo transformers nltk inflect scipy pycocotools opencv-python matplotlib

git clone https://github.com/microsoft/GLIP.git
cd GLIP; python setup.py build develop --user; cd ..
```

### SAM 环境安装

```shell
pip install git+https://github.com/facebookresearch/segment-anything.git
```
如果由于网络原因无法使用pip方法进行下载，可以采取以下方法。
```
git clone git+https://github.com/facebookresearch/segment-anything.git
cd segment-anything
python setup.py install
```
## demo 视频和文件的获取

```shell
cd playground
wget https://download.openmmlab.com/playground/mmtracking/tracking_demo.zip
unzip tracking_demo.zip
```

## 权重下载

```shell
mkdir ../models
wget -P ../models/ https://download.openmmlab.com/mmdetection/v3.0/detic/detic_centernet2_swin-b_fpn_4x_lvis-coco-in21k/detic_centernet2_swin-b_fpn_4x_lvis-coco-in21k_20230120-0d301978.pth
wget -P ../models/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget -P ../models/ https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/models/glip_a_tiny_o365.pth
wget -P ../models/ https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
```

## 模型推理演示

仅以 GroundingDINO 为例

### 多目标跟踪

```shell
cd mmtracking_open_detection

# input a video
python tracking_demo.py "../tracking_demo/mot_challenge_track.mp4" "configs/GroundingDINO_SwinB.cfg.py" "../models/groundingdino_swinb_cogcoor.pth"  --text-prompt "person . rider . car . truck . bus . train . motorcycle . bicycle ." --out-dir "outputs/mot_challenge" --init_track_thr 0.35 --obj_score_thrs_high 0.3

# input a images folder
python tracking_demo.py "../tracking_demo/bdd_val_track" "configs/GroundingDINO_SwinB.cfg.py" "../models/groundingdino_swinb_cogcoor.pth"  --text-prompt "person . rider . car . truck . bus . train . motorcycle . bicycle ." --out-dir "outputs/bdd100k" --fps 30
```

### 多目标跟踪和分割

```shell
cd mmtracking_open_detection

# input a images folder
python tracking_demo.py "../tracking_demo/bdd_val_track" "configs/GroundingDINO_SwinB.cfg.py" "../models/groundingdino_swinb_cogcoor.pth"  --text-prompt "person . rider . car . truck . bus . train . motorcycle . bicycle ." --out-dir "outputs/bdd100k" --fps 30 --mots 
```
