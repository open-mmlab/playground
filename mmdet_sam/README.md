# MMDetection-SAM

<div align=center>
<img src="https://user-images.githubusercontent.com/27466624/231659917-e3069822-2193-4261-b216-5f53baa64b53.PNG"/>
</div>

The current research direction of general-purpose object detection is moving toward large multimodality models. In addition to image inputs, most recent research outcomes incorporate text modality to improve the performance. Once the text modality is added, some very good properties of generic detection algorithms start to emerge, such as:

1. a large amount of easily accessible text data can be leveraged for joint training.
2. easy implementation of open-set object detection, leading to genuinely universal detection tasks.
3. can be used with the superb models in NLP to create some interesting and useful features.

Recently, Meta AI proposed [Segment Anything](https://github.com/facebookresearch/segment-anything) which claims to be able to segment any object, and there are many great implementations based on it. MMDet integrates many high-performance and easy-to-use detection models, so why not we combine MMDet models and Segment Anything together to create something interesting?

From the current point of view, generic object detection can be divided into two main categories:

1. Closed-Set Object Detection, which can only detect a fixed number of classes of objects that appear in the training set
2. Open-Set object detection, which can also detect objects and categories that are not in the training set.

With the popularity of multimodal algorithms, open-set object detection has become a new research area, in which there are three popular directions:

1. Zero-Shot Object Detection, also known as zero-sample object detection, which emphasizes that the object categories of the testing set are not in the training set.
2. Open-Vocabulary Object Detection, which detects all objects with given category list that appears in the target images.
3. Grounding Object Detection, which predicts the location of the objects with given text descriptions that appears in the target images.

These three directions are not completely distinguishable in practice, but only different in general terms. Based on the above descriptions, we provide inference and evaluation scripts for multiple models to work with Segment Anything, which accomplished the following features:

1. supports classic Closed-Set object detection models in MMDet to work with SAM models for automatic detection and instance segmentation, such as Faster R-CNN and DINO.
2. supports Open-Vocabulary detection model like Detic to work with SAM models for automatic detection and instance segmentation.
3. supports Grounding Object Detection models to work with SAM models for automatic detection and instance segmentation, such as Grounding DINO and GLIP.
4. supports distributed detection and segmentation evaluation and automatic COCO JSON exportation across all models for user-friendly custom data assessment.

## File Introduction

1. `detector_sam_demo.py`: for detection and instance segmentation on both single image and image folders.
2. `coco_style_eval.py`: for inference, evaluation and exportation on the given COCO JSON.
3. `browse_coco_json.py`: for visualizing exported COCO JSON.
4. `images2coco.py`: for customizd and unannotated COCO style JSON based on users' own image folder. This JSON can be used as the input to `coco_style_eval.py`.

This project referenced [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything). Thanks!

## Base Development Environment Setup

```shell
conda create -n mmdet-sam python=3.8 -y
conda activate mmdet-sam
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmengine

git clone https://github.com/open-mmlab/playground.git
cd playground
```

## Feature Introduction

This project has included many outstanding open-sourced algorithms, in order to reduce the burden of the environment installation. If you prefer not to use a certain part of the features, you can skip the corresponding part. Our project can be divided into the following three sections.

### 1 Open-Vocabulary + SAM

Use Open-Vocabulary object detectors with SAM models. Currently we support Detic.

#### Dependencies Installation

```shell
pip install -U openmim
mim install "mmcv>=2.0.0"

# build from source
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection; pip install -e .; cd ..

pip install git+https://github.com/facebookresearch/segment-anything.git
pip install git+https://github.com/openai/CLIP.git
```

#### Demonstration

```shell
cd mmdet_sam

# download weights
mkdir ../models
wget -P ../models/ https://download.openmmlab.com/mmdetection/v3.0/detic/detic_centernet2_swin-b_fpn_4x_lvis-coco-in21k/detic_centernet2_swin-b_fpn_4x_lvis-coco-in21k_20230120-0d301978.pth
wget -P ../models/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# input a single image
python detector_sam_demo.py ../images/cat_remote.jpg \
    configs/Detic_LI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.py \
    ../models/detic_centernet2_swin-b_fpn_4x_lvis-coco-in21k_20230120-0d301978.pth \
    -t cat \
    --sam-device cpu
```

The result will be generated at `outputs/cat_remote.jpg` like the following one:

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/231418323-97b489b1-43df-4065-853e-1e2539679ee3.png"/>
</div>

We can also detect the remote by editing `--test-prompt`. Please be aware that you must use empty space and `.` to separate different categories.

```shell
# input a single image
python detector_sam_demo.py ../images/cat_remote.jpg \
    configs/Detic_LI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.py \
    ../models/detic_centernet2_swin-b_fpn_4x_lvis-coco-in21k_20230120-0d301978.pth \
    -t "cat . remote" \
    --sam-device cpu
```

The generated `outputs/cat_remote.jpg` is now like this:

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/231419108-bc5ef1ed-cb0b-496a-a19e-9b3b55479426.png"/>
</div>

You can also run inferences on a folder by using this command:

```shell
# input a folder
python detector_sam_demo.py ../images \
    configs/Detic_LI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.py \
    ../models/detic_centernet2_swin-b_fpn_4x_lvis-coco-in21k_20230120-0d301978.pth \
    -t "cat . remote" \
    --sam-device cpu
```

The result images will be generated at the `outputs` folder.

If the graphics memory of your GPU can only support running one model, you can use `--cpu-off-load` to make sure every time there will be only one model running on the GPU:

```shell
# input a folder
python detector_sam_demo.py ../images \
    configs/Detic_LI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.py \
    ../models/detic_centernet2_swin-b_fpn_4x_lvis-coco-in21k_20230120-0d301978.pth \
    -t "cat . remote" \
    --cpu-off-load
```

We also support CPU inference by using `--det-device cpu --sam-device cpu`.

As Detic includes the mask results, we add a additional parameter `--use-detic-mask`. This allows us to run Detic only without running SAM models.

```shell
# input a folder
python detector_sam_demo.py ../images \
    configs/Detic_LI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.py \
    ../models/detic_centernet2_swin-b_fpn_4x_lvis-coco-in21k_20230120-0d301978.pth \
    -t "cat . remote" \
    --det-device cpu \
    --use-detic-mask
```

If you would like to visualize the results only, you can use set `--only-det` to run without SAM models.

```shell
# input a sinle image
python detector_sam_demo.py ../images/cat_remote.jpg \
    configs/Detic_LI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.py \
    ../models/detic_centernet2_swin-b_fpn_4x_lvis-coco-in21k_20230120-0d301978.pth \
    -t "cat" \
    --only-det
```

The `outputs/cat_remote.jpg` now is like this:

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/231426607-3b5ed4db-5077-463a-9462-f86b955a1f23.png"/>
</div>

### 2 MMdet models + SAM

Use MMDet models with SAM models for instance segmentation tasks. Currently all MMDet models are supported.

#### Dependencies Installation

```shell
pip install -U openmim
mim install "mmcv>=2.0.0"

# build from source
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection; pip install -e .; cd ..
```

#### Demonstration

You can run all features like what we have covered in the above Detic section. The only difference is that you do not need to set `--text-prompt`. Here we demonstrate some classic usages.

1. `Faster R-CNN` models

```shell
cd mmdet_sam

mkdir ../models
wget -P ../models/ https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth

# input a single image
python detector_sam_demo.py ../images/cat_remote.jpg \
    ../mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_2x_coco.py \
    ../models/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth \
    --sam-device cpu
```

2. `DINO` models

```shell
cd mmdet_sam

mkdir ../models
wget -P ../models/ https://download.openmmlab.com/mmdetection/v3.0/dino/dino-5scale_swin-l_8xb2-12e_coco/dino-5scale_swin-l_8xb2-12e_coco_20230228_072924-a654145f.pth

# input a single image
python detector_sam_demo.py ../images/cat_remote.jpg \
    ../mmdetection/configs/dino/dino-5scale_swin-l_8xb2-12e_coco.py \
    dino-5scale_swin-l_8xb2-12e_coco_20230228_072924-a654145f.pth  \
    --sam-device cpu
```

### 3 Grounding models + SAM

Use Grounding object detectors with SAM models for instance segmentation tasks. Currently we support Grounding DINO and GLIP.

#### Dependencies Installation

Grounding DINO:

```shell
cd playground
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install git+https://github.com/IDEA-Research/GroundingDINO.git # Please make sure your PyTorch, GCC and NVCC are all compatible to build CUDA ops successfully
```

GLIP:

```shell
cd playground

pip install git+https://github.com/facebookresearch/segment-anything.git
pip install einops shapely timm yacs tensorboardX ftfy prettytable pymongo transformers nltk inflect scipy pycocotools opencv-python matplotlib

git clone https://github.com/microsoft/GLIP.git
cd GLIP; python setup.py build develop --user # Please make sure your PyTorch, GCC and NVCC are all compatible to build CUDA ops successfully
```

#### Demonstration

Still, the usages are identical to the Detic part, and we only demonstrate part of the features here.

```shell
cd mmdet_sam

mkdir ../models
wget -P ../models/ https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# input a single image
python detector_sam_demo.py ../images/cat_remote.jpg \
    configs/GroundingDINO_SwinT_OGC.py \
    ../models/groundingdino_swint_ogc.pth \
    -t "cat . remote" \
    --sam-device cpu
```

The result will be generated at `outputs/cat_remote.jpg` like the following one:

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/231431590-1c583de0-0f3a-410e-aded-6c5257540632.png"/>
</div>

```shell
cd mmdet_sam

mkdir ../models
wget -P ../models/ https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/models/glip_a_tiny_o365.pth

# input a single image
python detector_sam_demo.py ../images/cat_remote.jpg \
    configs/glip_A_Swin_T_O365.yaml \
    ../models/glip_a_tiny_o365.pth \
    -t "cat . remote" \
    --sam-device cpu
```

### 4 COCO JSON evaluation

We support running `coco_style_eval.py` in both a distributed and a non-distributed manner. By default, this script runs on COCO Val2017 dataset organized in the format shown as follows:

```text
├── ${COCO_DATA_ROOT}
│   ├── annotations
│      ├──── instances_val2017.json
│   ├── val2017
```

We use Detic to demonstrate how it works here, other algorithms are same to this.

```shell
cd mmdet_sam

# non-distributed
python coco_style_eval.py ${COCO_DATA_ROOT} \
    configs/Detic_LI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.py \
    ../models/detic_centernet2_swin-b_fpn_4x_lvis-coco-in21k_20230120-0d301978.pth \
    -t coco_cls_name.txt

# distributed on eight cards on one machine
bash ./dist_coco_style_eval.sh 8 ${COCO_DATA_ROOT} \
    configs/Detic_LI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.py \
    ../models/detic_centernet2_swin-b_fpn_4x_lvis-coco-in21k_20230120-0d301978.pth \
    -t coco_cls_name.txt
```

### 5 COCO evaluation results

|                                Method                                | bbox thresh |   Test set   | Box AP | Seg AP |
| :------------------------------------------------------------------: | :---------: | :----------: | :----: | :----: |
| [Detic](./configs/Detic_LI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.py) |     0.2     | COCO2017 Val | 0.465  | 0.388  |
| [Detic](./configs/Detic_LI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.py) |    0.001    | COCO2017 Val | 0.481  | 0.403  |
|        [GroundingDino](./configs/GroundingDINO_SwinT_OGC.py)         |     0.3     | COCO2017 Val | 0.419  |        |
|        [GroundingDino](./configs/GroundingDINO_SwinT_OGC.py)         |     0.0     | COCO2017 Val | 0.469  |        |
|       [GroundingDino\*](./configs/GroundingDINO_SwinT_OGC.py)        |     0.3     | COCO2017 Val | 0.404  |        |
|              [GLIP](./configs/glip_A_Swin_T_O365.yaml)               |     0.0     | COCO2017 Val | 0.429  |        |

**Note**:
\*means use original GroundingDINO approach to test

### 6 Custom dataset

We use a example to illustrate how a custom dataset can be used to obtain an annotation file using GroundingDINO and SAM.

#### Data prepare

Use the following command to download the cat dataset:

```shell
cd playground

wget https://download.openmmlab.com/mmyolo/data/cat_dataset.zip
mkdir data
unzip cat_dataset.zip -d data/cat
rm cat_dataset.zip
```

**Note**:，Need to replace `1 cat` with `cat` in `cat/class_with_id.txt`

Use the `images2coco.py` script to generate the unlabeled json file:

```shell
cd mmdet_sam
python images2coco.py ../data/cat/images ../data/cat/class_with_id.txt cat_coco.json
```

#### Inference

Here we use GroundingDINO and SAM model as an example for inference, and get the prediction json file:

```shell
python coco_style_eval.py ../data/cat/ \
      configs/GroundingDINO_SwinT_OGC.py \
      ../models/groundingdino_swint_ogc.pth \
      -t ../data/cat/class_with_id.txt \
      --data-prefix images \
      --ann-file annotations/cat_coco.json \
      --out-dir ../cat_pred \
      --sam-device cpu
```