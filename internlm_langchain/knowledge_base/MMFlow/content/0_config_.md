# 教程0: 如何使用 Configs

我们的配置文件 (configs) 中支持了模块化和继承设计，这便于进行各种实验。如果需要检查配置文件，可以通过运行 `python tools/misc/print_config.py /PATH/TO/CONFIG` 来查看完整的配置文件。

## 配置文件结构

在目录 `config/_base_` 下有四种基本模块类型，即数据集 (datasets) 、模型 (models) 、训练计划 (schedules) 以及默认运行配置 (default_runtime)。很多模型可以很容易地参考其中一种方法 (如 PWC-Net )来构建。这些由 `_base_` 中的模块构成的配置文件被称为 *原始配置 (primitive configs)*。

对于同一文件夹下的所有配置，建议只有**1个**原始配置。所有其他配置都应该从原始配置继承。这样，最大继承级别 (inheritance level) 为 3。

简单来说，我们建议贡献者去继承现有模型的配置文件。例如，如果在 PWC-Net 的基础上做了一些改动，可以首先通过在配置文件中指定原始配置 `_base_ = ../pwcnet/pwcnet_slong_8x1_flyingchairs_384x448.py` 来继承基本的 PWC-Net 结构，然后再根据需要修改配置文件中的指定字段 (fields)。

如果需要搭建不能与现有模型共享任何结构的全新的模型，您可以在 `configs` 下创建一个新文件夹 `xxx`。

您也可以参考 [mmcv](https://mmcv.readthedocs.io/en/latest/understand_mmcv/config.html) 文档中的更多细节。

## 配置文件命名规则

我们按照下面的风格来命名配置文件。建议贡献者使用相同的风格。

```text
{model}_{schedule}_[gpu x batch_per_gpu]_{training datasets}_[input_size].py
```

`{xxx}` 表示必填字段，而 `[yyy]` 表示可选字段。

- `{model}`: 模型类型，如 `pwcnet`, `flownets` 等等。
- `{schedule}`: 训练计划。按照 FlowNet2 中的约定，我们使用 `slong`、 `sfine` 和 `sshort`，或者像 `150k` 表示150k(iterations) 这样指定迭代次数。
- `[gpu x batch_per_gpu]`: GPU 数量以及每个 GPU上分配的样本数， 如 `8x1`。
- `{training datasets}`: 训练数据集，如 `flyingchairs`， `flyingthings3d_subset` 或 `flyingthings3d`。
- `[input_size]`: 训练时图片大小。

## 配置文件结构

为了帮助用户对完整的配置和 MMFlow 中的模块有一个基本的了解，我们以在 `flyingchairs` 上使用 `slong` 训练的 PWC-Net 的配置为例进行简单的讲解。有关每个模块的更详细的用法和相应的替代方案，请参阅 API 文档和 [MMDetection 教程](https://github.com/open-mmlab/mmdetection/blob/master/docs/tutorials/config.md)。

```python
_base_ = [
    '../_base_/models/pwcnet.py', '../_base_/datasets/flyingchairs_384x448.py',
    '../_base_/schedules/schedule_s_long.py', '../_base_/default_runtime.py'
]# 新增配置文件依赖的基本配置文件
```

`_base_/models/pwc_net.py` 是 PWC-Net 模型的基本配置文件。

```python
model = dict(
    type='PWCNet',  # 算法名称
    encoder=dict(  # 编码器 (Encoder) 模块配置
        type='PWCNetEncoder',  # PWC-Net 编码器名称
        in_channels=3,  # 输入通道数
        # 子卷积模块的类型: 如果 net_type 为 Basic，各尺度的卷积层数量为3；如果 net_type 为
        # Small，各尺度的卷积层数量为 2
        net_type='Basic',
        pyramid_levels=[
            'level1', 'level2', 'level3', 'level4', 'level5', 'level6'
        ], # 特征金字塔尺度，同时也是编码器输出字典 (dict) 的键值 (keys)
        out_channels=(16, 32, 64, 96, 128, 196),  #  各金字塔尺度 (pyramid level) 的输出通道数列表
        strides=(2, 2, 2, 2, 2, 2),  # 各金字塔尺度 (pyramid level) 的步长 (stride) 列表
        dilations=(1, 1, 1, 1, 1, 1),  # 各金字塔尺度 (pyramid level) 的膨胀率 (dilation) 列表
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1)),  # 针对编码器内 ConvModule 模块的激活函数配置
    decoder=dict(  # 解码器 (Decoder) 模块配置
        type='PWCNetDecoder',  # 光流估计解码器 (Decoder) 名称
        in_channels=dict(
            level6=81, level5=213, level4=181, level3=149, level2=117),  # PWC-Net 的 basic dense block 的输入通道数
        flow_div=20.,  # 用于缩放真实值 (Ground Truth) 的常数除数
        corr_cfg=dict(type='Correlation', max_displacement=4, padding=0),
        warp_cfg=dict(type='Warp'),
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
        scaled=False,  # 是否使用按参与相关性 (correlation) 计算的元素数量缩放相关性计算结果
        post_processor=dict(type='ContextNet', in_channels=565),  # 后处理网络配置
        flow_loss=dict(  # 损失函数配置
            type='MultiLevelEPE',
            p=2,
            reduction='sum',
            weights={ # 不同尺度下的光流损失加权权重
                'level2': 0.005,
                'level3': 0.01,
                'level4': 0.02,
                'level5': 0.08,
                'level6': 0.32
            }),
    ),
    # 模型训练测试配置。
    train_cfg=dict(),
    test_cfg=dict(),
    init_cfg=dict(
        type='Kaiming',
        nonlinearity='leaky_relu',
        layer=['Conv2d', 'ConvTranspose2d'],
        mode='fan_in',
        bias=0))
```

原始配置文件 `_base_/datasets/flyingchairs_384x448.py` 中是:

```python
dataset_type = 'FlyingChairs'  # 数据集名称
data_root = 'data/FlyingChairs/data'  # 数据集根目录

img_norm_cfg = dict(mean=[0., 0., 0.], std=[255., 255., 255], to_rgb=False)  # 输入图像标准化所需要的均值、标准差信息

train_pipeline = [ # 训练图片处理管道 (Pipeline)
    dict(type='LoadImageFromFile'),  # 加载图片
    dict(type='LoadAnnotations'),  # 加载光流数据
    dict(type='ColorJitter',  # 随机改变输入图片亮度、对比度、饱和度以及色调
     brightness=0.5,  # 亮度调整范围
     contrast=0.5,  # 对比度调整范围
     saturation=0.5,  # 饱和度调整范围
         hue=0.5),  # 色调调整范围
    dict(type='RandomGamma', gamma_range=(0.7, 1.5)),  # 随机伽马校正配置
    dict(type='Normalize', **img_norm_cfg),  # 图像标准化配置，具体数值来自 img_norm_cfg
    dict(type='GaussianNoise', sigma_range=(0, 0.04), clamp_range=(0., 1.)),  # 增加高斯噪声，高斯噪声的标准差采样自 [0, 0.04] 区间
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),  # 随机水平翻转配置
    dict(type='RandomFlip', prob=0.5, direction='vertical'),   # 随机竖直翻转配置
    # 随机仿射变换配置
    # global_transform 和 relative_transform 的键值应该是下列其中之一:
    #     ('translates', 'zoom', 'shear', 'rotate')。同时，每个键值和对应的数
    #     值需要满足下面的规则:
    #         - 平移: 沿图像坐标系 x 轴、y 轴的平移比率。默认为 (0., 0.)。
    #         - 缩放: 最小、最大的图片缩放比。默认为 (1.0, 1.0)。
    #         - 剪切: 最小、最大的图片剪切比。 默认为 (1.0, 1.0)。
    #         - 旋转: 最小、最大的旋转角度。 默认为 (0., 0.)。
    dict(type='RandomAffine',
         global_transform=dict(
            translates=(0.05, 0.05),
            zoom=(1.0, 1.5),
            shear=(0.86, 1.16),
            rotate=(-10., 10.)
        ),
         relative_transform=dict(
            translates=(0.00375, 0.00375),
            zoom=(0.985, 1.015),
            shear=(1.0, 1.0),
            rotate=(-1.0, 1.0)
        )),
    dict(type='RandomCrop', crop_size=(384, 448)),  # 随即裁剪输入图像与光流到 (384, 448) 大小
    dict(type='DefaultFormatBundle'),  # 它提供了格式化常用输入字段的简化接口，支持 'img1'、'img2' 和 'flow_gt' 字段
    dict(
        type='Collect',  # 从加载器 (loader) 收集与特定任务相关的数据
        keys=['imgs', 'flow_gt'],
        meta_keys=('img_fields', 'ann_fields', 'filename1', 'filename2',
                   'ori_filename1', 'ori_filename2', 'filename_flow',
                   'ori_filename_flow', 'ori_shape', 'img_shape',
                   'img_norm_cfg')),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='InputResize', exponent=4),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='TestFormatBundle'),  # 它提供了格式化常用输入字段的简化接口，支持 'img1'、'img2' 和 'flow_gt' 字段
    dict(
        type='Collect',
        keys=['imgs'],  # 从加载器 (loader) 收集与特定任务相关的数据
        meta_keys=('flow_gt', 'filename1', 'filename2', 'ori_filename1',
                   'ori_filename2', 'ori_shape', 'img_shape', 'img_norm_cfg',
                   'scale_factor', 'pad_shape'))  # meta_keys 中的 'flow_gt' 是用于在线模型评估 (online evaluation) 的字段
]

data = dict(
    train_dataloader=dict(
        samples_per_gpu=1,  # 单一 GPU 上的样本数
        workers_per_gpu=5,  # 每个 GPU 上预取数据的线程数
        drop_last=True),  # 是否删除最后一个非完整的批次 (batch)

    val_dataloader=dict(
        samples_per_gpu=1,  # 单一 GPU 上的样本数
        workers_per_gpu=2,  # 每个 GPU 上预取数据的线程数
        shuffle=False),  # 是否删除最后一个非完整的批次 (batch)

    test_dataloader=dict(
        samples_per_gpu=1,  # 单一 GPU 上的样本数
        workers_per_gpu=2,  # 每个 GPU 上预取数据的线程数
        shuffle=False),  # 是否打乱输入顺序

    train=dict(  # 训练数据集配置
        type=dataset_type,
        pipeline=train_pipeline,
        data_root=data_root,
        split_file='data/FlyingChairs_release/FlyingChairs_train_val.txt',  # 训练、验证子集
    ),

    val=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        data_root=data_root,
        test_mode=True),

    test=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        data_root=data_root,
        test_mode=True)
)
```

原始配置文件 `_base_/schedules/schedule_s_long.py` 中是:

```python
# optimizer
optimizer = dict(
    type='Adam', lr=0.0001, weight_decay=0.0004, betas=(0.9, 0.999))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    by_epoch=False,
    gamma=0.5,
    step=[400000, 600000, 800000, 1000000])
runner = dict(type='IterBasedRunner', max_iters=1200000)
checkpoint_config = dict(by_epoch=False, interval=100000)
evaluation = dict(interval=100000, metric='EPE')
```

原始配置文件 `_base_/default_runtime.py` 中是:

```python
log_config = dict(  # 配置注册记录器钩子
    interval=50,  # 打印训练日志信息的频率
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])  # 记录训练过程的记录器配置
dist_params = dict(backend='nccl')  # 设置分布式训练的参数，端口 (port) 也可以在这里配置
log_level = 'INFO'  # 日志信息等级
load_from = None  # 从给定路径加载模型作为预训练模型，这不同于恢复训练 (resume training)
workflow = [('train', 1)]  # 运行期的工作流设置。 [('train', 1)] 是指只有一个工作流 'train'，且它只执行一次
```

## 通过脚本参数来修改配置

在使用 `tools/train.py` 或 `tools/test.py` 时，可以通过指定 `--cfg-options` 来就地 (in-place) 修改配置。

- 更新配置字典链 (dict chains) 中的键 (keys)。

  可以按照原始配置中字典键的顺序指定配置选项。
  例如， `--cfg-option model.encoder.in_channels=6`。

- 更新配置列表中的键 (keys)。

  一些配置字典在配置文件中组成一个列表。例如，训练数据处理管道 (pipeline) `data.train.pipeline` 往往是一个像 `[dict(type='LoadImageFromFile'), ...]` 一样的列表。 如果希望将其中的 `'LoadImageFromFile'` 替换为 `'LoadImageFromWebcam'`，可以通过指定 `--cfg-option data.train.pipeline.0.type='LoadImageFromWebcam'` 来实现。

- 更新列表或元组的值。

  如果需要更新的值是一个元组或是列表。例如，配置文件中通常设置训练工作流为 `workflow=[('train', 1)]`。如果希望修改这个键值，可以通过指定 `--cfg-options workflow="[(train,1),(val,1)]"` 来实现。 注意，引号 " 对于列表、元组数据类型是必需的，且在指定值的引号内**不允许**有空格。

## 常见问题 (FAQ)

### 忽略原始配置中的部分字段

如果需要，你可以通过设置 `_delete_=True` 来忽略原始配置文件中的部分字段。
可以参考 [mmcv](https://mmcv.readthedocs.io/en/latest/utils.html#inherit-from-base-config-with-ignored-fields) 中的简单说明。

请仔细阅读 [config 教程](https://github.com/open-mmlab/mmdetection/blob/master/docs/tutorials/config.md) 以更好地了解这一方法。

### 使用配置文件中的中间变量

配置文件中使用了一些中间变量，例如数据集配置中的 `train_pipeline`/`test_pipeline`。
值得注意的是，在修改子配置文件中的中间变量时，用户需要再次将中间变量传递到相应的字段中。更为直观的例子参见 [config 教程](https://github.com/open-mmlab/mmdetection/blob/master/docs/tutorials/config.md)。
