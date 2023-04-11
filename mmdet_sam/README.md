# MMDetection-SAM

本工程目录下用于存放 MMDetection 和 SAM 相关的代码。其包括如下功能：

1. 支持 MMDet 模型(Closed-set)、Open-Vocabulary(Open-set) 和 Zero-shot 检测模型(Open-set) 串联 SAM 模型进行自动检测和实例分割标注
2. 支持上述模型的检测和分割分布式评估和自动 COCO JSON 导出

下面对每个脚本功能进行说明：

1. `detector_sam_demo.py` 用于单张图片或者文件夹的检测和实例分割模型推理
2. `mmdet_sam/coco_style_eval.py` 用于对输入的 COCO JSON 进行检测和实例分割模型推理、评估和导出
3. `browse_coco_json.py` 用于可视化导出的 JSON 文件

本工程参考了 [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything)，非常感谢！

## 基础环境安装

```shell
conda create -n mmdet-sam python=3.8 -y
conda activate mmdet-sam
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmengine
```

## 功能说明

### 1 MMDet 模型 + SAM

#### 依赖安装

#### 功能演示

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

#### 功能演示

```shell
cd mmsam/mmdet_sam

python detector_sam_demo.py ../images ../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py ../models/groundingdino_swint_ogc.pth -t cat --sam-device cpu
python detector_sam_demo.py ../images/cat_remote.jpg ../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py ../models/groundingdino_swint_ogc.pth -t "cat . remote" --sam-device cpu

python coco_style_eval.py {DATA_ROOT} ../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py ../models/groundingdino_swint_ogc.pth -t coco_cls_name.txt --sam-device cpu

bash ./dist_coco_style_eval.sh 8 {DATA_ROOT} ../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py ../models/groundingdino_swint_ogc.pth -t coco_cls_name.txt
```
