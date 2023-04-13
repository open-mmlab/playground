# Open-Set Object Detection with MMPose Pose Estimation


## 参数说明

下面对每个脚本功能进行说明：

1. `grounding_demo.py` 用于单张图片或者文件夹的检测和姿态估计模型推理

(1) grounding_demo.py

```shell
image 图片路径
det_config 目标检测模型配置文件路径
det_weight 目标检测模型权重文件路径
pose_config 姿态估计模型配置文件路径
pose_weight 姿态估计模型权重文件路径
--out-dir 输出图片存储目录，默认outputs
--box-thr 目标检测阈值，默认0.3
--device 模型运行设备, 如：cpu, cuda, cuda:0，默认cuda:0
--text-prompt -t GroundingDINO text prompt，默认human
--text-thr text阈值，默认0.25
--kpt-thr 关键点显示阈值，默认0.3
--skeleton-style 骨架绘制风格，可选mmpose或openpose，其中openpose只适用于coco 17点数据，默认mmpose风格
--radius 关键点显示半径，默认3
--thickness 骨架线宽度，默认1
--draw-bbox 是否绘制 bounding box
--alpha bounding box不透明度，默认0.8
```


## 基础环境安装

```shell
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmengine
```

## 功能说明

本工程中包括了引入了诸多优秀的开源算法，为了减少用户安装环境负担，如果你不想使用某部分功能，则可以不安装对应的依赖。

### 1 Grounding DINO + MMPose

使用 Grounding DINO 检测目标，然后通过 MMPose 对所检测的目标进行姿态识别。可以利用 Grounding DINO 的 text prompt 来选择需要检测的物体。

#### 依赖安装

1. Grounding DINO

```shell
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -e .

git clone https://github.com/open-mmlab/mmpose.git
cd mmpose
pip install -e .

git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -e .
```

#### 功能演示

```shell
cd mmsam/mmpose_open_detection

# 下载权重
mkdir ../models
wget -P ../models/ https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth
wget -P ../models/ https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth

# 单张图片输入
# 可以从https://www.pexels.com/photo/group-of-people-near-wall-2422290/下载示例图片
# 假设mmpose目录与mmsam目录同级

python grounding_demo.py ../images/pexels-jopwell-2422290.jpg configs/GroundingDINO_SwinT_OGC.py ../models/groundingdino_swint_ogc.pth ../../mmpose/projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-m_8xb256-420e_coco-256x192.py ../models/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth -t human
```

会在当前路径生成 `outputs/pexels-jopwell-2422290.jpg`，效果如下所示：
![group_people](https://user-images.githubusercontent.com/8425513/231439110-c0e7d6f8-5692-4bcb-b6cf-c3c243a990a5.jpg)


可以通过text-prompt修改需要检测的物体，例如改成检测猫。但需要提供可以进行猫姿态识别的模型才能获得正确的姿态识别结果。MMPose目前支持多种动物以及人体，人脸，手的姿态识别任务，只需替换相应模型的配置文件及权重文件即可。
