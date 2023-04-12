# MMRotate-SAM

本工程目录存放 MMRotate 和 SAM 相关的代码。

脚本说明：
1. `eval_zero-shot-oriented-detection_dota.py` 实现了 SAM 的 Zero-shot Oriented Object Detection。在 SAM 前级联水平框检测器（当使用旋转框检测器时，取旋转框的最小水平外接矩作为水平框输出），将检测器输出的边界框作为 prompt 输入 SAM 中，输出掩码的最小有向外接矩即为对应目标的旋转框。
2. `data_builder` 存放数据集、数据加载器的配置信息以及配置过程。

本工程参考了 [sam-mmrotate](https://github.com/Li-Qingyun/sam-mmrotate)

## 环境安装

```shell
conda create -n mmrotate-sam python=3.8 -y
conda activate mmrotate-sam
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html

pip install openmim
mim install mmengine 'mmcv>=2.0.0rc0' 'mmrotate>=1.0.0rc0'

pip install git+https://github.com/facebookresearch/segment-anything.git
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```