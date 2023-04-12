# MMDetection-SAM

--这里放图

目前通用目标检测研究方向朝着多模态大模型发展。除了图片输入外，目前新的研究大部分都会加入文本模态来提升性能。一旦加入文本模态后，通用检测算法就会出现一些非常好的性质，典型的如

1. 可以充分利用大量容易获取的文本数据来联合训练
2. 容易实现开放集目标检测，进而通向真正的通用检测
3. 可以和 NLP 中已经发布的超强模型联合使用，从而做到一些很有趣且实用的功能

最近 Meta AI 提出了 [Segment Anything](https://github.com/facebookresearch/segment-anything) 模型，号称可以对任意物体进行分割，基于此国内外也出现了不少下应用，MMDet 中集成了大量性能强且易用的检测模型，因此也可以基于 MMDet 模型和 Segment Anything 联合尝试做一些有趣的事情。

从目前来看，通用目标检测可以分成两大类：

1. 封闭集目标检测 Closed-Set Object Detection，即只能检测训练集出现的固定类别数的物体
2. 开发集目标检测 Open-Set Object Detection，即可以检测训练集外的类别的物体

随着多模态算法的流行，开放类别的目标检测已经成为了新的研究方向，在这其中有 3 个比较热门的研究方向：

1. Zero-Shot Object Detection，即零样本目标检测，其强调的是测试集类别不在训练集中
2. Open-Vocabulary Object Detection，即开发词汇目标检测，给定图片和类别词汇表，检测所有物体
3. Grounding Object Detection，即给定图片和文本描述，预测文本中所提到的在图片中的物体位置

实际上三个方向没法完全区分，只是通俗说法不同而已。 基于上述描述，结合 Segment Anything，我们提供了多个模型串联的推理和评估脚本。 具体包括如下功能：

1. 支持 MMDet 模型经典检测模型 (Closed-Set)，典型的如 Faster R-CNN 和 DINO 等串联 SAM 模型进行自动检测和实例分割标注
2. 支持 Open-Vocabulary 检测模型，典型的如 Detic 串联 SAM 模型进行自动检测和实例分割标注
3. 支持 Grounding Object Detection 模型，典型的如 Grounding DINO 和 GLIP 串联 SAM 模型进行自动检测和实例分割标注
4. 所有模型均支持分布式检测和分割评估和自动 COCO JSON 导出，方便用户对自定义数据进行评估

## 参数说明

下面对每个脚本功能进行说明：

1. `detector_sam_demo.py` 用于单张图片或者文件夹的检测和实例分割模型推理
2. `coco_style_eval.py` 用于对输入的 COCO JSON 进行检测和实例分割模型推理、评估和导出
3. `browse_coco_json.py` 用于可视化导出的 JSON 文件

(1) detector_sam_demo.py

(2) coco_style_eval.py

(3) browse_coco_json.py

本工程参考了 [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything)，非常感谢！

## 基础环境安装

```shell
conda create -n mmdet-sam python=3.8 -y
conda activate mmdet-sam
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmengine
```

## 功能说明

本工程中包括了引入了诸多优秀的开源算法，为了减少用户安装环境负担，如果不不想使用某部分功能，则可以不安装对应的依赖。下面分成 3 个部分说明。

### 1 MMDet 模型 + SAM

其表示 MMDet 中的检测模型串联 SAM 从而实现实例分割任务，目前支持所有 MMDet 中已经支持的算法。

#### 依赖安装

```shell
pip install -U openmim
mim install "mmcv>=2.0.0"

# 源码安装
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection; pip install -e .; cd ..
```

#### 模型推理演示

1 `Faster R-CNN` 模型

```shell
cd mmsam/mmdet_sam

# 单张图片评估
python detector_sam_demo.py ../images/cat_remote.jpg mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_2x_coco.py https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth

# 如果 GPU 显存不够，可以采用 CPU 推理
python detector_sam_demo.py ../images/cat_remote.jpg mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_2x_coco.py https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth  --sam-device cpu

# 文件夹推理
python detector_sam_demo.py ../images mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_2x_coco.py https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth  --sam-device cpu

# 如果你的 GPU 每次只能支持一个模型的推理，则可以开启 --cpu-off-load
python detector_sam_demo.py ../images mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_2x_coco.py https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth  --cpu-off-load
```

2 `DINO` 模型

```shell
cd mmsam/mmdet_sam

python detector_sam_demo.py ../images/cat_remote.jpg mmdetection/configs/dino/dino-5scale_swin-l_8xb2-12e_coco.py https://download.openmmlab.com/mmdetection/v3.0/dino/dino-5scale_swin-l_8xb2-12e_coco/dino-5scale_swin-l_8xb2-12e_coco_20230228_072924-a654145f.pth  --sam-device cpu
```

#### 分布式评估演示

对于 `coco_style_eval.py` 脚本，你可以采用分布式或者非分布式方式进行推理和评估。以 `Faster R-CNN` 为例

```shell
cd mmsam/mmdet_sam

python coco_style_eval.py ${DATA_ROOT} mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_2x_coco.py https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth
```

### 2 Open-Vocabulary + SAM

#### 依赖安装

#### 功能演示

### 2 Zero-shot + SAM

#### 依赖安装

1. Grounding DINO

```shell
cd mmsam
git clone https://github.com/facebookresearch/segment-anything.git
cd segment-anything; pip install -e .; cd ..

git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO; pip install -e .; cd ..
```

#### 模型推理演示

使用方式和前面完全相同。只不过其需要额外输入 `--text-prompt`

```shell
cd mmsam/mmdet_sam

python detector_sam_demo.py ../images ../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py ../models/groundingdino_swint_ogc.pth -t cat --sam-device cpu
python detector_sam_demo.py ../images/cat_remote.jpg ../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py ../models/groundingdino_swint_ogc.pth -t "cat . remote" --sam-device cpu

python coco_style_eval.py {DATA_ROOT} ../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py ../models/groundingdino_swint_ogc.pth -t coco_cls_name.txt --sam-device cpu

bash ./dist_coco_style_eval.sh 8 {DATA_ROOT} ../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py ../models/groundingdino_swint_ogc.pth -t coco_cls_name.txt
```

#### 分布式评估演示

```shell
cd mmsam/mmdet_sam

# 非分布式评估
python coco_style_eval.py ${DATA_ROOT} ../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py ../models/groundingdino_swint_ogc.pth -t coco_cls_name.txt --sam-device cpu

# 分布式单机8卡评估
bash ./dist_coco_style_eval.sh 8 ${DATA_ROOT} ../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py ../models/groundingdino_swint_ogc.pth -t coco_cls_name.txt
```
