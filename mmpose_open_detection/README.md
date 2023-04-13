# Open detection jointed MMPose

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/231637220-bebcea80-afad-4039-92c0-5c9cb2c82cad.png"/>
</div>

This project aims to integate MMPose with open detection for developing open pose estimation.

## Parameter Description

Below is an explanation of the function of each script:

1. `mmpose_open_demo.py` is used for inference of the open detection and pose estimation models on a single image or a folder of images.

(1) mmpose_open_demo.py

```shell
image image path
det_config config file path of object detection model
det_weight weight file path of object detection model
pose_config config file path of pose estimation model
pose_weight weight file path of pose estimation model
--out-dir result dir, default outputs
--box-thr threshold for object detection, default 0.3
--device device, default cuda:0
--text-prompt -t text prompt, default human
--text-thr text threshold, default 0.25
--kpt-thr threshold for displaying keypoints, default 0.3
--skeleton-style drawing style of skeleton, option mmpose or openpose, openpose only applies to coco 17 point data, default mmpose style
--radius radius for displaying keypoints, default 3
--thickness thickness for skeleton drawing, default 1
--draw-bbox whether to draw bounding box
--alpha bounding box opacity, default 0.8
```

## Basic environment installation

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

## Function description

This project includes many excellent open source algorithms. To reduce the burden of installing the environment, you can choose not to install the corresponding dependencies if you do not need certain functions.

### 1 Open-Vocabulary + MMPose

Currently, Detic algorithm is supported.

#### Dependency installation

```shell
# source install
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection; pip install -e .; cd ..

pip install git+https://github.com/facebookresearch/segment-anything.git
pip install git+https://github.com/openai/CLIP.git
```

#### Function demonstration

```shell
cd mmpose_open_detection

mkdir ../models
wget -P ../models/ https://download.openmmlab.com/mmdetection/v3.0/detic/detic_centernet2_swin-b_fpn_4x_lvis-coco-in21k/detic_centernet2_swin-b_fpn_4x_lvis-coco-in21k_20230120-0d301978.pth
wget -P ../models/ https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth
```

To avoid copyright disputes, you can download the example image from https://www.pexels.com/photo/group-of-people-near-wall-2422290/ and place it in `../images/`. Assuming the image name is `pexels-photo-2422290.jpeg`, the image content is shown below:

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/231630138-eec92ce2-9e26-4c54-b58c-1e39fb2add75.png"/>
</div>

```shell
python mmpose_open_demo.py ../images/pexels-photo-2422290.jpeg ../mmdet_sam/configs/Detic_LI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.py ../models/detic_centernet2_swin-b_fpn_4x_lvis-coco-in21k_20230120-0d301978.pth ../mmpose/projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-m_8xb256-420e_coco-256x192.py ../models/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth -t person
```

It will generate `outputs/pexels-photo-2422290.jpeg` in current dir, is shown below:

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/231633730-e4d60b19-2b1e-4d1e-87d9-6ab6cd6d717a.png"/>
</div>

We will soon add different text prompts to achieve pose detection for different categories of objects.

### 2 Grounding DINO + MMPose

Grounding DINO is used to detect objects, and then MMPose is used to recognize the pose. The text prompt of Grounding DINO can be used to select the objects that need to be detected.

#### Dependency installation

1. Grounding DINO

```shell
cd playground
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install git+https://github.com/IDEA-Research/GroundingDINO.git # to compile CUDA OP, please make sure that your PyTorch version, GCC version and NVCC compilation version are compatible.
```

#### Function demonstration

```shell
cd mmpose_open_detection

# download weight
mkdir ../models
wget -P ../models/ https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth
wget -P ../models/ https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth

python mmpose_open_demo.py ../images/pexels-photo-2422290.jpeg ../mmdet_sam/configs/GroundingDINO_SwinT_OGC.py ../models/groundingdino_swint_ogc.pth ../mmpose/projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-m_8xb256-420e_coco-256x192.py ../models/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth -t human
```

It will generate `outputs/pexels-photo-2422290.jpeg` in current dir, and the effect is shown below:

<div align=center>
<img src="https://user-images.githubusercontent.com/8425513/231439110-c0e7d6f8-5692-4bcb-b6cf-c3c243a990a5.jpg"/>
</div>

You can modify the object to be detected by using the text-prompt, for example, to detect cats. However, a model that can perform cat pose recognition is required to obtain correct pose recognition results. MMPose currently supports pose recognition tasks for various animals, as well as human body, face and hand. You only need to replace the config file and weight file of the corresponding model to achieve.
