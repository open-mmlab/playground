# MMTracking Open Detection

<div align=center>
<img src="https://user-images.githubusercontent.com/27466624/231666666-4f4c5696-df73-45cd-af04-758ea3806a82.png"/>
</div>

With the help of open object detection and utilizing motion information (Kalman filters), multi-object tracking can be performed.
Due to time constraints, currently only GroundingDINO, GLIP, and Detic combined with ByteTrack are supported for tracking.

<div align="center">
<img src="https://github.com/zwhus/pictures/raw/main/bdd.gif">
<img src="https://github.com/zwhus/pictures/raw/main/demo.gif">
<img src="https://github.com/zwhus/pictures/raw/main/demo%2B(1).gif">
</div>

## Parameter Description

`tracking_demo.py` : for multi-object tracking inference on video or image folders.

This project referenced [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO). Thanks!

## Base Development Environment Setup

```shell
conda create -n mmtracking-sam python=3.8 -y
conda activate mmtracking-sam
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
git clone https://github.com/open-mmlab/playground.git
````

### MMDet Dependencies Installation

```shell
pip install -U openmim
mim install "mmcv>=2.0.0"

# build from source
cd playground
git clone -b tracking https://github.com/open-mmlab/mmdetection.git
cd mmdetection; pip install -e .; cd ..
pip install lap seaborn
```

### Grounding Dino Dependencies Installation

```shell
cd playground
pip install git+https://github.com/IDEA-Research/GroundingDINO.git
```
If you can't use pip to download due to network reasons, you can choose the following method instead.
```
git clone git+https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
python setup.py install
```

### GLIP Dependencies Installation

```shell
cd playground
pip install einops shapely timm yacs tensorboardX ftfy prettytable pymongo transformers nltk inflect scipy pycocotools opencv-python matplotlib

git clone https://github.com/microsoft/GLIP.git
cd GLIP; python setup.py build develop --user; cd ..
```

### SAM Dependencies Installation

```shell
pip install git+https://github.com/facebookresearch/segment-anything.git
```
If you can't use pip to download due to network reasons, you can choose the following method instead.
```
git clone git+https://github.com/facebookresearch/segment-anything.git
cd segment-anything
python setup.py install
```

## Obtaining the demo video and images

```shell
cd playground
wget https://download.openmmlab.com/playground/mmtracking/tracking_demo.zip
unzip tracking_demo.zip
```

## Download weights

```shell
mkdir ../models
wget -P ../models/ https://download.openmmlab.com/mmdetection/v3.0/detic/detic_centernet2_swin-b_fpn_4x_lvis-coco-in21k/detic_centernet2_swin-b_fpn_4x_lvis-coco-in21k_20230120-0d301978.pth
wget -P ../models/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget -P ../models/ https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/models/glip_a_tiny_o365.pth
wget -P ../models/ https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
```

## Demonstration of model inference

Using GroundingDINO as an example only.

### MOT

```shell
cd mmtracking_open_detection

# input a video
python tracking_demo.py "../tracking_demo/mot_challenge_track.mp4" "configs/GroundingDINO_SwinB.cfg.py" "../models/groundingdino_swinb_cogcoor.pth"  --text-prompt "person . rider . car . truck . bus . train . motorcycle . bicycle ." --out-dir "outputs/mot_challenge" --init_track_thr 0.35 --obj_score_thrs_high 0.3

# input a images folder
python tracking_demo.py "../tracking_demo/bdd_val_track" "configs/GroundingDINO_SwinB.cfg.py" "../models/groundingdino_swinb_cogcoor.pth"  --text-prompt "person . rider . car . truck . bus . train . motorcycle . bicycle ." --out-dir "outputs/bdd100k" --fps 30
```

### MOTS

```shell
cd mmtracking_open_detection

# input a images folder
python tracking_demo.py "../tracking_demo/bdd_val_track" "configs/GroundingDINO_SwinB.cfg.py" "../models/groundingdino_swinb_cogcoor.pth"  --text-prompt "person . rider . car . truck . bus . train . motorcycle . bicycle ." --out-dir "outputs/bdd100k" --fps 30 --mots
```
