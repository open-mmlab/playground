# 快速入门

本文介绍 MMFlow 的基本使用，安装指引请参看[安装文档](install.md)

<!--- TOC --->

- [快速入门](#%E5%BF%AB%E9%80%9F%E5%85%A5%E9%97%A8)
  - [准备数据集](#%E5%87%86%E5%A4%87%E6%95%B0%E6%8D%AE%E9%9B%86)
  - [模型推理](#%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86)
    - [演示样例](#%E6%BC%94%E7%A4%BA%E6%A0%B7%E4%BE%8B)
    - [数据集上测试](#%E6%95%B0%E6%8D%AE%E9%9B%86%E4%B8%8A%E6%B5%8B%E8%AF%95)
  - [模型训练](#%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83)
  - [其他教程](#%E5%85%B6%E4%BB%96%E6%95%99%E7%A8%8B)

<!--- TOC --->

## 准备数据集

我们推荐用将目录 `$MMFlow/data` 指向（symlink）存放目录的地址。请根据以下指引准备训练数据集。

- [FlyingChairs](../en/data_prepare/FlyingChairs/README.md)
- [FlyingThings3d_subset](../en/data_prepare/FlyingThings3d_subset/README.md)
- [FlyingThings3d](../en/data_prepare/FlyingThings3d/README.md)
- [Sintel](../en/data_prepare/Sintel/README.md)
- [KITTI2015](../en/data_prepare/KITTI2015/README.md)
- [KITTI2012](../en/data_prepare/KITTI2012/README.md)
- [FlyingChairsOcc](../en/data_prepare/FlyingChairsOcc/README.md)
- [ChairsSDHom](../en/data_prepare/ChairsSDHom/README.md)
- [HD1K](../en/data_prepare/hd1k/README.md)

## 模型推理

我们提供了测试脚本，可以在标准的数据集 Sintel, KITTI 等上进行测试，也提供可直接对图像或者视频进行推理的 API 和 脚本。

### 演示样例

我们提供了样例脚本可以之间来推理两帧之间的光流

1. [图像推理样例](../demo/image_demo.py)

   ```shell
   python demo/image_demo.py ${IMAGE1} ${IMAGE2} ${CONFIG_FILE} ${CHECKPOINT_FILE} ${OUTPUT_DIR} \
       [--out_prefix] ${OUTPUT_PREFIX} [--device] ${DEVICE}
   ```

   可选参数：

   - `--out_prefix`: 输出结果的前缀，包括光流文件和可视化后的光流图片。
   - `--device`: 用于推理的设备

   例子：

   假设您已经将 checkpoints 下载到目录中 `checkpoints/`,　并将输出的结果存在 `raft_demo`

   ```shell
   python demo/image_demo.py demo/frame_0001.png demo/frame_0002.png \
       configs/raft/raft_8x2_100k_mixed_368x768.pth \
       checkpoints/raft_8x2_100k_TSKH_368x768.pth raft_demo
   ```

2. [视频推理样例](../demo/video_demo.py)

   ```shell
   python demo/video_demo.py ${VIDEO} ${CONFIG_FILE} ${CHECKPOINT_FILE} ${OUTPUT_FILE} \
       [--gt] ${GROUND_TRUTH} [--device] ${DEVICE}
   ```

   可选参数：

   - `--gt`: 输入视频的 ground truth 视频文件。如果指定，可视化结果将会把 ground truth
     与预测结果连接在一起输出作为比较。
   - `--device`: 用于推理的设备

   例子：

   假设您已经将 checkpoints 下载到目录中 `checkpoints/`,　并将输出的结果存为 `raft_demo.mp4`

   ```shell
   python demo/video_demo.py demo/demo.mp4 \
       configs/raft/raft_8x2_100k_mixed_368x768.py \
       checkpoints/raft_8x2_100k_mixed_368x768.pth \
       raft_demo.mp4 --gt demo/demo_gt.mp4
   ```

### 数据集上测试

可以利用以下命令来测试模型在数据集上的精度，更多细节可以查看 [tutorials/1_inference](../en/tutorials/1_inference.md)。

```shell
# single-gpu testing
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

可选参数：

- `--out_dir`: 输出文件保存目录，被指定后，模型输出的光流将以 '.flo' 文件的形式会保存在该目录。
- `--fuse-conv-bn`: 是否将 conv 和 bn 融合，如果定义为 True,　这项操作会对推理轻微加速。
- `--show_dir`: 可视化光流图的保存目录，被指定后，模型输出的光流可视化后将以图片的形式保存在该目录。
- `--eval`: 评估的指标，例如：EPE。
- `--cfg-option`: 键值 xxx=yyy 将被合并到配置文件中。例如：'--cfg-option model.encoder.in_channels=6'

例子：

假设您已经将 checkpoints 下载到目录中 `checkpoints/`。
测试 PWC-Net 在 Sintel 数据集的 clean 和 final 两个子数据集上的 EPE 指标，但不保存预测结果。

```shell
python tools/test.py configs/pwcnet_8x1_sfine_sintel_384x768.py \
    checkpoints/pwcnet_8x1_sfine_sintel_384x768.pth --eval EPE

```

## 模型训练

你可以使用[训练脚本](../tools/train.py)直接来启动一个单卡训练任务，更多信息请查看 [tutorials/2_finetune](../en/tutorials/2_finetune.md)

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

可选参数：

- `--work-dir`: 工作目录，如果指定会覆盖配置文件中的工作目录参数。
- `--load-from`: 将要加载的 checkpoint 文件参数。
- `--resume-from`: 从之前的 checkpoint 文件恢复训练。
- `--no-validate`: 是否在训练过程不对模型做评估。
- `--seed`: Python, Numpy 和 Pytorch 中生成随机数的随机种子。
- `--deterministic`: 如果指定，它将为 CUDNN 后端设置确定性选项。
- `--cfg-options`: 键值 xxx=yyy 将被合并到配置文件中。例如：'--cfg-option model.encoder.in_channels=6'

`resume-from` 和 `load-from` 的不同：

`resume-from` 同时加载模型参数、优化器状态以及当前 epoch/iter。通常用于恢复被意外中断的训练过程。
`load-from` 只加载模型的参数,训练将会从头开始。通常用于模型微调。

## 其他教程

我们为用户提供了一些教程文档:

- [learn about configs](../en/tutorials/0_config.md)
- [inference model](../en/tutorials/1_inference.md)
- [finetune model](../en/tutorials/2_finetune.md)
- [customize data pipelines](../en/tutorials/3_data_pipeline.md)
- [add new modules](../en/tutorials/4_new_modules.md)
- [customize runtime settings](../en/tutorials/5_customize_runtime.md).
