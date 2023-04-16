# MMDetection-SAM

<div align=center>
<img src="https://user-images.githubusercontent.com/27466624/231659917-e3069822-2193-4261-b216-5f53baa64b53.PNG"/>
</div>

目前通用目标检测研究方向朝着多模态大模型发展。除了图片输入外，目前新的研究大部分都会加入文本模态来提升性能。一旦加入文本模态后，通用检测算法就会出现一些非常好的性质，典型的如

1. 可以充分利用大量容易获取的文本数据来联合训练
2. 容易实现开放词汇目标检测，进而通向真正的通用检测
3. 可以和 NLP 中已经发布的超强模型联合使用，从而做到一些很有趣且实用的功能

最近 Meta AI 提出了 [Segment Anything](https://github.com/facebookresearch/segment-anything) 模型，号称可以对任意物体进行分割，基于此国内外也出现了不少下应用。MMDet 中集成了大量性能强且易用的检测模型，因此也可以基于 MMDet 模型和 Segment Anything 联合尝试做一些有趣的事情。

从目前来看，通用目标检测可以分成两大类：

1. 封闭集目标检测 Closed-Set Object Detection，即只能检测训练集出现的固定类别数的物体
2. 开放集目标检测 Open-Set Object Detection，即可以检测训练集外的类别的物体

随着多模态算法的流行，开放类别的目标检测已经成为了新的研究方向，在这其中有 3 个比较热门的研究方向：

1. Zero-Shot Object Detection，即零样本目标检测，其强调的是测试集类别不在训练集中
2. Open-Vocabulary Object Detection，即开放词汇目标检测，给定图片和类别词汇表，检测所有物体
3. Grounding Object Detection，即给定图片和文本描述，预测文本中所提到的在图片中的物体位置

实际上三个方向没法完全区分，只是通俗说法不同而已。基于上述描述，结合 Segment Anything，我们提供了多个模型串联的推理和评估脚本。具体包括如下功能：

1. 支持 MMDet 模型经典检测模型 (Closed-Set)，典型的如 Faster R-CNN 和 DINO 等串联 SAM 模型进行自动检测和实例分割标注
2. 支持 Open-Vocabulary 检测模型，典型的如 Detic 串联 SAM 模型进行自动检测和实例分割标注
3. 支持 Grounding Object Detection 模型，典型的如 Grounding DINO 和 GLIP 串联 SAM 模型进行自动检测和实例分割标注
4. 所有模型均支持分布式检测和分割评估和自动 COCO JSON 导出，方便用户对自定义数据进行评估

## 项目文件说明

下面对每个脚本功能进行说明：

1. `detector_sam_demo.py` 用于单张图片或者文件夹的检测和实例分割模型推理
2. `coco_style_eval.py` 用于对输入的 COCO JSON 进行检测和实例分割模型推理、评估和导出
3. `browse_coco_json.py` 用于可视化导出的 JSON 文件
4. `images2coco.py` 用于用户自定义且不包括标注的文件夹列表生成 COCO 格式的 JSON，该 JSON 可以作为 `coco_style_eval.py` 输入

本工程参考了 [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything)，非常感谢！

## 基础环境安装

```shell
conda create -n mmdet-sam python=3.8 -y
conda activate mmdet-sam
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmengine

git clone https://github.com/open-mmlab/playground.git
cd playground
```

## 功能说明

本工程中包括了引入了诸多优秀的开源算法，为了减少用户安装环境负担，如果你不想使用某部分功能，则可以不安装对应的依赖。下面分成 3 个部分说明。

### 1 Open-Vocabulary + SAM

其表示采用 Open-Vocabulary 目标检测器串联 SAM 模型，目前支持 Detic 算法

#### 依赖安装

```shell
pip install -U openmim
mim install "mmcv>=2.0.0"

# 源码安装
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection; pip install -e .; cd ..

pip install git+https://github.com/facebookresearch/segment-anything.git
pip install git+https://github.com/openai/CLIP.git
```

#### 功能演示

```shell
cd mmdet_sam

# 下载权重
mkdir ../models
wget -P ../models/ https://download.openmmlab.com/mmdetection/v3.0/detic/detic_centernet2_swin-b_fpn_4x_lvis-coco-in21k/detic_centernet2_swin-b_fpn_4x_lvis-coco-in21k_20230120-0d301978.pth
wget -P ../models/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# 单张图片输入
python detector_sam_demo.py ../images/cat_remote.jpg \
    configs/Detic_LI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.py \
    ../models/detic_centernet2_swin-b_fpn_4x_lvis-coco-in21k_20230120-0d301978.pth \
    -t cat \
    --sam-device cpu
```

会在当前路径生成 `outputs/cat_remote.jpg`，效果如下所示：

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/231418323-97b489b1-43df-4065-853e-1e2539679ee3.png"/>
</div>

我们可以修改 `--text-prompt` 来检测出遥控器，注意不同类别间要用空格和 . 区分开。

```shell
# 单张图片输入
python detector_sam_demo.py ../images/cat_remote.jpg \
    configs/Detic_LI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.py \
    ../models/detic_centernet2_swin-b_fpn_4x_lvis-coco-in21k_20230120-0d301978.pth \
    -t "cat . remote" \
    --sam-device cpu
```

会在当前路径生成 `outputs/cat_remote.jpg`，效果如下所示：

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/231419108-bc5ef1ed-cb0b-496a-a19e-9b3b55479426.png"/>
</div>

你也可以输入文件夹进行推理，如下所示：

```shell
# 文件夹输入
python detector_sam_demo.py ../images \
    configs/Detic_LI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.py \
    ../models/detic_centernet2_swin-b_fpn_4x_lvis-coco-in21k_20230120-0d301978.pth \
    -t "cat . remote" \
    --sam-device cpu
```

会在当前路径生成 `outputs` 文件夹里面存放了两种图片。

如果你的 GPU 显存只能支持一个模型运行，可以指定 `--cpu-off-load` 来设置每次只将一个模型放置到 GPU 上

```shell
# 文件夹输入
python detector_sam_demo.py ../images \
    configs/Detic_LI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.py \
    ../models/detic_centernet2_swin-b_fpn_4x_lvis-coco-in21k_20230120-0d301978.pth \
    -t "cat . remote" \
    --cpu-off-load
```

目前也支持 CPU 推理，你可以设置 `--det-device cpu --sam-device cpu`。

由于 Detic 算法实际上包括了 mask 结果，因此我们增加了额外参数 `--use-detic-mask`，当指定该参数时候表示仅仅运行 Detic 而不运行 sam。

```shell
# 文件夹输入
python detector_sam_demo.py ../images \
    configs/Detic_LI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.py \
    ../models/detic_centernet2_swin-b_fpn_4x_lvis-coco-in21k_20230120-0d301978.pth \
    -t "cat . remote" \
    --det-device cpu \
    --use-detic-mask
```

如果你只想可视化检测结果，则可以指定 `--only-det` 则也不会运行 sam 模型。

```shell
# 单张图片输入
python detector_sam_demo.py ../images/cat_remote.jpg \
    configs/Detic_LI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.py \
    ../models/detic_centernet2_swin-b_fpn_4x_lvis-coco-in21k_20230120-0d301978.pth \
    -t "cat" \
    --only-det
```

会在当前路径生成 `outputs/cat_remote.jpg`，效果如下所示：

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/231426607-3b5ed4db-5077-463a-9462-f86b955a1f23.png"/>
</div>

### 2 MMDet 模型 + SAM

其表示 MMDet 中的检测模型串联 SAM 从而实现实例分割任务，目前支持所有 MMDet 中已经支持的检测算法。

#### 依赖安装

```shell
pip install -U openmim
mim install "mmcv>=2.0.0"

# 源码安装
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection; pip install -e .; cd ..
```

#### 模型推理演示

其用法和前面的 Detic 一样，只是不需要设置 `--text-prompt`, 下面仅仅列出典型用法

1 `Faster R-CNN` 模型

```shell
cd mmdet_sam

mkdir ../models
wget -P ../models/ https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth

# 单张图片评估
python detector_sam_demo.py ../images/cat_remote.jpg \
    ../mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_2x_coco.py \
    ../models/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth \
    --sam-device cpu
```

2 `DINO` 模型

```shell
cd mmdet_sam

mkdir ../models
wget -P ../models/ https://download.openmmlab.com/mmdetection/v3.0/dino/dino-5scale_swin-l_8xb2-12e_coco/dino-5scale_swin-l_8xb2-12e_coco_20230228_072924-a654145f.pth

python detector_sam_demo.py ../images/cat_remote.jpg \
    ../mmdetection/configs/dino/dino-5scale_swin-l_8xb2-12e_coco.py \
    dino-5scale_swin-l_8xb2-12e_coco_20230228_072924-a654145f.pth  \
    --sam-device cpu
```

### 3 Grounding 模型 + SAM

其表示引入 Grounding 目标检测模型串联 SAM 从而实现实例分割任务，目前支持 Grounding DINO 和 GLIP。

#### 依赖安装

如果是 Grounding DINO 则安装如下依赖即可

```shell
cd playground
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install git+https://github.com/IDEA-Research/GroundingDINO.git # 需要编译 CUDA OP，请确保你的 PyTorch 版本、GCC 版本和 NVCC 编译版本兼容
```

如果是 GLIP 则安装如下依赖即可

```shell
cd playground

pip install git+https://github.com/facebookresearch/segment-anything.git
pip install einops shapely timm yacs tensorboardX ftfy prettytable pymongo transformers nltk inflect scipy pycocotools opencv-python matplotlib

git clone https://github.com/microsoft/GLIP.git
cd GLIP; python setup.py build develop --user  # 需要编译 CUDA OP，请确保你的 PyTorch 版本、GCC 版本和 NVCC 编译版本兼容，暂时不支持 PyTorch 1.11+ 版本
```

#### 功能演示

使用方式和 Detic 完全相同，下面仅演示部分功能。

```shell
cd mmdet_sam

mkdir ../models
wget -P ../models/ https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# 单张图片输入
python detector_sam_demo.py ../images/cat_remote.jpg \
    configs/GroundingDINO_SwinT_OGC.py \
    ../models/groundingdino_swint_ogc.pth \
    -t "cat . remote" \
    --sam-device cpu
```

会在当前路径生成 `outputs/cat_remote.jpg`，效果如下所示：

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/231431590-1c583de0-0f3a-410e-aded-6c5257540632.png"/>
</div>

```shell
cd mmdet_sam

mkdir ../models
wget -P ../models/ https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/models/glip_a_tiny_o365.pth

# 单张图片输入
python detector_sam_demo.py ../images/cat_remote.jpg \
    configs/glip_A_Swin_T_O365.yaml \
    ../models/glip_a_tiny_o365.pth \
    -t "cat . remote" \
    --sam-device cpu
```

### 4 COCO JSON 评估

对于 `coco_style_eval.py` 脚本，你可以采用分布式或者非分布式方式进行推理和评估，默认参数是对 COCO Val2017 数据集进行评估，COCO 文件组织格式如下所示：

```text
├── ${COCO_DATA_ROOT}
│   ├── annotations
│      ├──── instances_val2017.json
│   ├── val2017
```

以 Detic 算法为例，其余算法用法相同。

```shell
cd mmdet_sam

# 非分布式评估
python coco_style_eval.py ${COCO_DATA_ROOT} \
    configs/Detic_LI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.py \
    ../models/detic_centernet2_swin-b_fpn_4x_lvis-coco-in21k_20230120-0d301978.pth \
    -t coco_cls_name.txt

# 分布式单机 8 卡评估
bash ./dist_coco_style_eval.sh 8 ${COCO_DATA_ROOT} \
    configs/Detic_LI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.py \
    ../models/detic_centernet2_swin-b_fpn_4x_lvis-coco-in21k_20230120-0d301978.pth \
    -t coco_cls_name.txt
```

### 5 COCO 评估结果

|                                Method                                | bbox thresh |   Test set   | Box AP | Seg AP |
| :------------------------------------------------------------------: | :---------: | :----------: | :----: | :----: |
| [Detic](./configs/Detic_LI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.py) |     0.2     | COCO2017 Val | 0.465  | 0.388  |
| [Detic](./configs/Detic_LI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.py) |    0.001    | COCO2017 Val | 0.481  | 0.403  |
|        [GroundingDino](./configs/GroundingDINO_SwinT_OGC.py)         |     0.3     | COCO2017 Val | 0.419  |        |
|        [GroundingDino](./configs/GroundingDINO_SwinT_OGC.py)         |     0.0     | COCO2017 Val | 0.469  |        |
|       [GroundingDino\*](./configs/GroundingDINO_SwinT_OGC.py)        |     0.3     | COCO2017 Val | 0.404  |        |
|              [GLIP](./configs/glip_A_Swin_T_O365.yaml)               |     0.0     | COCO2017 Val | 0.429  |        |

**Note**:
\*意思是使用原始GroundingDino的方式进行评估

### 6 自定义数据集

以下将使用一个具体例子来说明自定义的数据集如何得到模型推理的标注文件

#### 数据准备

使用以下命令下载 cat 数据集

```shell
cd playground

wget https://download.openmmlab.com/mmyolo/data/cat_dataset.zip
mkdir data
unzip cat_dataset.zip -d data/cat
rm cat_dataset.zip
```

**注意**:，需要将`cat/class_with_id.txt`里面的`1 cat`换成 `cat`

使用 `images2coco.py` 脚本生成没有标注的 json 文件

```shell
cd mmdet_sam
python images2coco.py ../data/cat/images ../data/cat/class_with_id.txt cat_coco.json
```

#### 模型推理

这里使用 GroundingDINO 串联 SAM 模型为例进行推理，得到预测结果的 json 文件

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
