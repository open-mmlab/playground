# MMTracking-Grouding

借助于开发集目标检测 Open-Set Object Detection, 利用运动信息（卡尔曼滤波器）来进行多目标跟踪。

受限于时间，目前仅仅支持使用Byte的跟踪器

## 参数说明

`tracking_demo.py` 用于视频或者图片文件夹的多目标跟踪推理

本工程参考了 [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)，非常感谢！

## 基础环境安装

```shell
conda create -n mmdet-sam python=3.8 -y
conda activate mmdet-sam
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

#### 依赖安装

```shell
pip install -U openmim
mim install "mmcv>=2.0.0"

# 源码安装
git clone -b tracking https://github.com/open-mmlab/mmdetection.git
cd mmdetection; pip install -e .; cd ..

git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO; pip install -e .; cd ..
```

#### 模型推理演示

```shell
cd mmsam/mmtracking_grounding

python tracking_demo.py  -c "../GroundingDINO/groundingdino/config/GroundingDINO_SwinB.cfg.py" -p "../groundingdino_swinb_cogcoor.pth" -i "../images/bdd_val_track" -t "pedestrian. rider. car. truck. bus. train. motorcycle. bicycle." --out-dir "bdd_track"

python tracking_demo.py  -c "../GroundingDINO/groundingdino/config/GroundingDINO_SwinB.cfg.py" -p "../groundingdino_swinb_cogcoor.pth" -i "../images/demo_mot.mp4" -t "pedestrian." --out-dir "mot_challenge"
```
