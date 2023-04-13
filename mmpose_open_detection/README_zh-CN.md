# 开放目标检测联合 MMPose

<div align=center>
<img src="https://user-images.githubusercontent.com/27466624/231660029-03166059-e8cf-4b17-8aa5-b42f3a52f12a.PNG"/>
</div>

本工程将 MMPose 联合开放目标检测实现开放词汇姿态识别。

## 参数说明

下面对每个脚本功能进行说明：

1. `mmpose_open_demo.py` 用于单张图片或者文件夹的开放检测和姿态估计模型推理

(1) mmpose_open_demo.py

```shell
image 图片路径
det_config 目标检测模型配置文件路径
det_weight 目标检测模型权重文件路径
pose_config 姿态估计模型配置文件路径
pose_weight 姿态估计模型权重文件路径
--out-dir 输出图片存储目录,默认 outputs
--box-thr 目标检测阈值, 默认 0.3
--device 模型运行设备, 默认 cuda:0
--text-prompt -t text prompt, 默认 human
--text-thr text 阈值, 默认0.25
--kpt-thr 关键点显示阈值, 默认0.3
--skeleton-style 骨架绘制风格, 可选 mmpose 或 openpose, 其中 openpose 只适用于 coco 17 点数据, 默认 mmpose 风格
--radius 关键点显示半径, 默认3
--thickness 骨架线宽度, 默认1
--draw-bbox 是否绘制 bounding box
--alpha bounding box不透明度, 默认 0.8
```

## 基础环境安装

```shell
conda create -n mmpose python=3.8 -y
conda activate mmpose
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmengine

git clone https://github.com/open-mmlab/playground.git
cd playground

pip install -U openmim
mim install "mmcv>=2.0.0"

git clone https://github.com/open-mmlab/mmpose.git
cd mmpose; pip install -e .; cd ..
```

## 功能说明

本工程中包括了引入了诸多优秀的开源算法，为了减少用户安装环境负担，如果你不想使用某部分功能，则可以不安装对应的依赖。

### 1 Open-Vocabulary + MMPose

目前支持 Detic 算法。

#### 依赖安装

```shell
# 源码安装
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection; pip install -e .; cd ..

pip install git+https://github.com/facebookresearch/segment-anything.git
pip install git+https://github.com/openai/CLIP.git
```

#### 功能演示

```shell
cd mmpose_open_detection

mkdir ../models
wget -P ../models/ https://download.openmmlab.com/mmdetection/v3.0/detic/detic_centernet2_swin-b_fpn_4x_lvis-coco-in21k/detic_centernet2_swin-b_fpn_4x_lvis-coco-in21k_20230120-0d301978.pth
wget -P ../models/ https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth
```

为了避免版权争议，你可以去 https://www.pexels.com/photo/group-of-people-near-wall-2422290/ 下载示例图，并放置于 `../images/` 下, 假设图片名为 `pexels-photo-2422290.jpeg`，图片内容如下所示：

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/231630138-eec92ce2-9e26-4c54-b58c-1e39fb2add75.png"/>
</div>

```shell
python mmpose_open_demo.py ../images/pexels-photo-2422290.jpeg \
    ../mmdet_sam/configs/Detic_LI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.py \
    ../models/detic_centernet2_swin-b_fpn_4x_lvis-coco-in21k_20230120-0d301978.pth \
    ../mmpose/projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-m_8xb256-420e_coco-256x192.py \
    ../models/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth \
    -t person
```

会在当前路径生成 `outputs/pexels-photo-2422290.jpeg`，效果如下所示：

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/231633730-e4d60b19-2b1e-4d1e-87d9-6ab6cd6d717a.png"/>
</div>

我们即将会加入输入不同的 text prompt 从而实现对图片中不同类别物体的姿态检测。

### 2 Grounding DINO + MMPose

使用 Grounding DINO 检测目标，然后通过 MMPose 对所检测的目标进行姿态识别。可以利用 Grounding DINO 的 text prompt 来选择需要检测的物体。

#### 依赖安装

1. Grounding DINO

```shell
cd playground
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install git+https://github.com/IDEA-Research/GroundingDINO.git # 需要编译 CUDA OP，请确保你的 PyTorch 版本、GCC 版本和 NVCC 编译版本兼容
```

#### 功能演示

```shell
cd mmpose_open_detection

# 下载权重
mkdir ../models
wget -P ../models/ https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth
wget -P ../models/ https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth

python mmpose_open_demo.py ../images/pexels-photo-2422290.jpeg \
    ../mmdet_sam/configs/GroundingDINO_SwinT_OGC.py \
    ../models/groundingdino_swint_ogc.pth \
    ../mmpose/projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-m_8xb256-420e_coco-256x192.py \
    ../models/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth \
    -t human
```

会在当前路径生成 `outputs/pexels-photo-2422290.jpeg`，效果如下所示：

<div align=center>
<img src="https://user-images.githubusercontent.com/8425513/231439110-c0e7d6f8-5692-4bcb-b6cf-c3c243a990a5.jpg"/>
</div>

可以通过 text-prompt 修改需要检测的物体，例如改成检测猫。但需要提供可以进行猫姿态识别的模型才能获得正确的姿态识别结果。MMPose 目前支持多种动物以及人体，人脸，手的姿态识别任务，只需替换相应模型的配置文件及权重文件即可我们即将支持该功能。
