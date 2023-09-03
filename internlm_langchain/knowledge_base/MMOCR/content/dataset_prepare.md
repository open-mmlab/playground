# 数据集准备

## 前言

经过数十年的发展，OCR 领域涌现出了一系列的相关数据集，这些数据集往往采用风格各异的格式来提供文本的标注文件，使得用户在使用这些数据集时不得不进行格式转换。因此，为了方便用户进行数据集准备，我们提供了[一键式的数据准备脚本](./data_prepare/dataset_preparer.md)，使得用户仅需使用一行命令即可完成数据集准备的全部步骤。

在这一节，我们将介绍一个典型的数据集准备流程：

1. [下载数据集并将其格式转换为 MMOCR 支持的格式](#数据集下载及格式转换)
2. [修改配置文件](#修改配置文件)

然而，如果你已经有了 MMOCR 支持的格式的数据集，那么第一步就不是必须的。你可以阅读[数据集类及标注格式](../basic_concepts/datasets.md#数据集类及标注格式)来了解更多细节。

## 数据集下载及格式转换

以 ICDAR 2015 数据集的文本检测任务准备步骤为例，你可以执行以下命令来完成数据集准备：

```shell
python tools/dataset_converters/prepare_dataset.py icdar2015 --task textdet
```

命令执行完成后，数据集将被下载并转换至 MMOCR 格式，文件目录结构如下：

```text
data/icdar2015
├── textdet_imgs
│   ├── test
│   └── train
├── textdet_test.json
└── textdet_train.json
```

数据准备完毕以后，你也可以通过使用我们提供的数据集浏览工具 [browse_dataset.py](./useful_tools.md#数据集可视化工具) 来可视化数据集的标签是否被正确生成，例如：

```bash
python tools/analysis_tools/browse_dataset.py configs/textdet/_base_/datasets/icdar2015.py
```

## 修改配置文件

### 单数据集训练

在使用新的数据集时，我们需要对其图像、标注文件的路径等基础信息进行配置。`configs/xxx/_base_/datasets/` 路径下已预先配置了 MMOCR 中常用的数据集（当你使用 `prepare_dataset.py` 来准备数据集时，这个配置文件通常会在数据集准备就绪后自动生成），这里我们以 ICDAR 2015 数据集为例（见 `configs/textdet/_base_/datasets/icdar2015.py`）：

```Python
icdar2015_textdet_data_root = 'data/icdar2015' # 数据集根目录

# 训练集配置
icdar2015_textdet_train = dict(
    type='OCRDataset',
    data_root=icdar2015_textdet_data_root,               # 数据根目录
    ann_file='textdet_train.json',                       # 标注文件名称
    filter_cfg=dict(filter_empty_gt=True, min_size=32),  # 数据过滤
    pipeline=None)
# 测试集配置
icdar2015_textdet_test = dict(
    type='OCRDataset',
    data_root=icdar2015_textdet_data_root,
    ann_file='textdet_test.json',
    test_mode=True,
    pipeline=None)
```

在配置好数据集后，我们还需要在相应的算法模型配置文件中导入想要使用的数据集。例如，在 ICDAR 2015 数据集上训练 "DBNet_R18" 模型：

```Python
_base_ = [
    '_base_dbnet_r18_fpnc.py',
    '../_base_/datasets/icdar2015.py',  # 导入数据集配置文件
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_sgd_1200e.py',
]

icdar2015_textdet_train = _base_.icdar2015_textdet_train            # 指定训练集
icdar2015_textdet_train.pipeline = _base_.train_pipeline   # 指定训练集使用的数据流水线
icdar2015_textdet_test = _base_.icdar2015_textdet_test              # 指定测试集
icdar2015_textdet_test.pipeline = _base_.test_pipeline     # 指定测试集使用的数据流水线

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=icdar2015_textdet_train)    # 在 train_dataloader 中指定使用的训练数据集

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=icdar2015_textdet_test)    # 在 val_dataloader 中指定使用的验证数据集

test_dataloader = val_dataloader
```

### 多数据集训练

此外，基于 [`ConcatDataset`](mmocr.datasets.ConcatDataset)，用户还可以使用多个数据集组合来训练或测试模型。用户只需在配置文件中将 dataloader 中的 dataset 类型设置为 `ConcatDataset`，并指定对应的数据集列表即可。

```Python
train_list = [ic11, ic13, ic15]
train_dataloader = dict(
    dataset=dict(
        type='ConcatDataset', datasets=train_list, pipeline=train_pipeline))
```

例如，以下配置使用了 MJSynth 数据集进行训练，并使用 6 个学术数据集（CUTE80, IIIT5K, SVT, SVTP, ICDAR2013, ICDAR2015）进行测试。

```Python
_base_ = [ # 导入所有需要使用的数据集配置
    '../_base_/datasets/mjsynth.py',
    '../_base_/datasets/cute80.py',
    '../_base_/datasets/iiit5k.py',
    '../_base_/datasets/svt.py',
    '../_base_/datasets/svtp.py',
    '../_base_/datasets/icdar2013.py',
    '../_base_/datasets/icdar2015.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_adadelta_5e.py',
    '_base_crnn_mini-vgg.py',
]

# 训练集列表
train_list = [_base_.mjsynth_textrecog_train]
# 测试集列表
test_list = [
    _base_.cute80_textrecog_test, _base_.iiit5k_textrecog_test, _base_.svt_textrecog_test,
    _base_.svtp_textrecog_test, _base_.icdar2013_textrecog_test, _base_.icdar2015_textrecog_test
]

# 使用 ConcatDataset 来级联列表中的多个数据集
train_dataset = dict(
       type='ConcatDataset', datasets=train_list, pipeline=_base_.train_pipeline)
test_dataset = dict(
       type='ConcatDataset', datasets=test_list, pipeline=_base_.test_pipeline)

train_dataloader = dict(
    batch_size=192 * 4,
    num_workers=32,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset)

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=test_dataset)

val_dataloader = test_dataloader
```
