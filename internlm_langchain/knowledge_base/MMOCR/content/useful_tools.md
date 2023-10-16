# 常用工具

## 可视化工具

### 数据集可视化工具

MMOCR 提供了数据集可视化工具 `tools/visualizations/browse_datasets.py` 以辅助用户排查可能遇到的数据集相关的问题。用户只需要指定所使用的训练配置文件（通常存放在如 `configs/textdet/dbnet/xxx.py` 文件中）或数据集配置（通常存放在 `configs/textdet/_base_/datasets/xxx.py` 文件中）路径。该工具将依据输入的配置文件类型自动将经过数据流水线（data pipeline）处理过的图像及其对应的标签，或原始图片及其对应的标签绘制出来。

#### 支持参数

```bash
python tools/visualizations/browse_dataset.py \
    ${CONFIG_FILE} \
    [-o, --output-dir ${OUTPUT_DIR}] \
    [-p, --phase ${DATASET_PHASE}] \
    [-m, --mode ${DISPLAY_MODE}] \
    [-t, --task ${DATASET_TASK}] \
    [-n, --show-number ${NUMBER_IMAGES_DISPLAY}] \
    [-i, --show-interval ${SHOW_INTERRVAL}] \
    [--cfg-options ${CFG_OPTIONS}]
```

| 参数名              | 类型                                  | 描述                                                                                                                                             |
| ------------------- | ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| config              | str                                   | (必须) 配置文件路径。                                                                                                                            |
| -o, --output-dir    | str                                   | 如果图形化界面不可用，请指定一个输出路径来保存可视化结果。                                                                                       |
| -p, --phase         | str                                   | 用于指定需要可视化的数据集切片，如 "train", "test", "val"。当数据集存在多个变种时，也可以通过该参数来指定待可视化的切片。                        |
| -m, --mode          | `original`, `transformed`, `pipeline` | 用于指定数据可视化的模式。`original`：原始模式，仅可视化数据集的原始标注；`transformed`：变换模式，展示经过所有数据变换步骤的最终图像；`pipeline`：流水线模式，展示数据变换过程中每一个中间步骤的变换图像。默认使用 `transformed` 变换模式。 |
| -t, --task          | `auto`, `textdet`, `textrecog`        | 用于指定可视化数据集的任务类型。`auto`：自动模式，将依据给定的配置文件自动选择合适的任务类型，如果无法自动获取任务类型，则需要用户手动指定为 `textdet` 文本检测任务 或 `textrecog` 文本识别任务。默认采用 `auto` 自动模式。 |
| -n, --show-number   | int                                   | 指定需要可视化的样本数量。若该参数缺省则默认将可视化全部图片。                                                                                   |
| -i, --show-interval | float                                 | 可视化图像间隔时间，默认为 2 秒。                                                                                                                |
| --cfg-options       | float                                 | 用于覆盖配置文件中的参数，详见[示例](./config.md#command-line-modification)。                                                                    |

#### 用法示例

以下示例演示了如何使用该工具可视化 "DBNet_R50_icdar2015" 模型使用的训练数据。

```Bash
# 使用默认参数可视化 "dbnet_r50dcn_v2_fpnc_1200e_icadr2015" 模型的训练数据
python tools/visualizations/browse_dataset.py configs/textdet/dbnet/dbnet_resnet50-dcnv2_fpnc_1200e_icdar2015.py
```

默认情况下，可视化模式为 "transformed"，您将看到经由数据流水线变换过后的图像和标注：

<center class="half">
    <img src="https://user-images.githubusercontent.com/24622904/187611542-01e9aa94-fc12-4756-964b-a0e472522a3a.jpg" width="250"/><img src="https://user-images.githubusercontent.com/24622904/187611555-3f5ea616-863d-4538-884f-bccbebc2f7e7.jpg" width="250"/><img src="https://user-images.githubusercontent.com/24622904/187611581-88be3970-fbfe-4f62-8cdf-7a8a7786af29.jpg" width="250"/>
</center>

如果您只想可视化原始数据集，只需将模式设置为 "original"：

```Bash
python tools/visualizations/browse_dataset.py configs/textdet/dbnet/dbnet_resnet50-dcnv2_fpnc_1200e_icdar2015.py -m original
```

<div align=center><img src="https://user-images.githubusercontent.com/22607038/206646570-382d0f26-908a-4ab4-b1a7-5cc31fa70c5f.jpg" style=" width: auto; height: 40%; "></div>

或者，您也可以使用 "pipeline" 模式来可视化整个数据流水线的中间结果：

```Bash
python tools/visualizations/browse_dataset.py configs/textdet/dbnet/dbnet_resnet50-dcnv2_fpnc_1200e_icdar2015.py -m pipeline
```

<div align=center><img src="https://user-images.githubusercontent.com/22607038/206637571-287640c0-1f55-453f-a2fc-9f9734b9593f.jpg" style=" width: auto; height: 40%; "></div>

另外，用户还可以通过指定数据集配置文件的路径来可视化数据集的原始图像及其对应的标注，例如：

```Bash
python tools/visualizations/browse_dataset.py configs/textrecog/_base_/datasets/icdar2015.py
```

部分数据集可能有多个变体。例如，`icdar2015` 文本识别数据集的[配置文件](/configs/textrecog/_base_/datasets/icdar2015.py)中包含两个测试集变体，分别为 `icdar2015_textrecog_test` 和 `icdar2015_1811_textrecog_test`，如下所示：

```python
icdar2015_textrecog_test = dict(
    ann_file='textrecog_test.json',
    # ...
    )

icdar2015_1811_textrecog_test = dict(
    ann_file='textrecog_test_1811.json',
    # ...
)
```

在这种情况下，用户可以通过指定 `-p` 参数来可视化不同的变体，例如，使用以下命令可视化 `icdar2015_1811_textrecog_test` 变体：

```Bash
python tools/visualizations/browse_dataset.py configs/textrecog/_base_/datasets/icdar2015.py -p icdar2015_1811_textrecog_test
```

基于该工具，用户可以轻松地查看数据集的原始图像及其对应的标注，以便于检查数据集的标注是否正确。

### 优化器参数策略可视化工具

MMOCR提供了优化器参数可视化工具 `tools/visualizations/vis_scheduler.py` 以辅助用户排查优化器的超参数调度器（无需训练），支持学习率（learning rate）和动量（momentum）。

#### 工具简介

```bash
python tools/visualizations/vis_scheduler.py \
    ${CONFIG_FILE} \
    [-p, --parameter ${PARAMETER_NAME}] \
    [-d, --dataset-size ${DATASET_SIZE}] \
    [-n, --ngpus ${NUM_GPUs}] \
    [-s, --save-path ${SAVE_PATH}] \
    [--title ${TITLE}] \
    [--style ${STYLE}] \
    [--window-size ${WINDOW_SIZE}] \
    [--cfg-options]
```

**所有参数的说明**：

- `config` : 模型配置文件的路径。
- **`-p, parameter`**: 可视化参数名，只能为 `["lr", "momentum"]` 之一， 默认为 `"lr"`.
- **`-d, --dataset-size`**: 数据集的大小。如果指定，`build_dataset` 将被跳过并使用这个大小作为数据集大小，默认使用 `build_dataset` 所得数据集的大小。
- **`-n, --ngpus`**: 使用 GPU 的数量, 默认为1。
- **`-s, --save-path`**: 保存的可视化图片的路径，默认不保存。
- `--title`: 可视化图片的标题，默认为配置文件名。
- `--style`: 可视化图片的风格，默认为 `whitegrid`。
- `--window-size`: 可视化窗口大小，如果没有指定，默认为 `12*7`。如果需要指定，按照格式 \`W\*H'。
- `--cfg-options`: 对配置文件的修改，参考[学习配置文件](../user_guides/config.md)。

```{note}
部分数据集在解析标注阶段比较耗时，可直接将 `-d, dataset-size` 指定数据集的大小，以节约时间。
```

#### 如何在开始训练前可视化学习率曲线

你可以使用如下命令来绘制配置文件 `configs/textdet/dbnet/dbnet_resnet50-dcnv2_fpnc_1200e_icdar2015.py` 将会使用的变化率曲线：

```bash
python tools/visualizations/vis_scheduler.py configs/textdet/dbnet/dbnet_resnet50-dcnv2_fpnc_1200e_icdar2015.py -d 100
```

<div align=center><img src="https://user-images.githubusercontent.com/43344034/232755081-cad8fe62-349d-400a-bc38-7f5d17824011.png" style=" width: auto; height: 40%; "></div>

## 分析工具

### 离线评测工具

对于已保存的预测结果，我们提供了离线评测脚本 `tools/analysis_tools/offline_eval.py`。例如，以下代码演示了如何使用该工具对 "PSENet" 模型的输出结果进行离线评估：

```Bash
# 初次运行测试脚本时，用户可以通过指定 --save-preds 参数来保存模型的输出结果
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --save-preds
# 示例：对 PSENet 进行测试
python tools/test.py configs/textdet/psenet/psenet_r50_fpnf_600e_icdar2015.py epoch_600.pth --save-preds

# 之后即可使用已保存的输出文件进行离线评估
python tools/analysis_tool/offline_eval.py ${CONFIG_FILE} ${PRED_FILE}
# 示例：对已保存的 PSENet 结果进行离线评估
python tools/analysis_tools/offline_eval.py configs/textdet/psenet/psenet_r50_fpnf_600e_icdar2015.py work_dirs/psenet_r50_fpnf_600e_icdar2015/epoch_600.pth_predictions.pkl
```

`--save-preds` 默认将输出结果保存至 `work_dir/CONFIG_NAME/MODEL_NAME_predictions.pkl`

此外，基于此工具，用户也可以将其他算法库获取的预测结果转换成 MMOCR 支持的格式，从而使用 MMOCR 内置的评估指标来对其他算法库的模型进行评测。

| 参数          | 类型  | 说明                                                             |
| ------------- | ----- | ---------------------------------------------------------------- |
| config        | str   | （必须）配置文件路径。                                           |
| pkl_results   | str   | （必须）预先保存的预测结果文件。                                 |
| --cfg-options | float | 用于覆写配置文件中的指定参数。[示例](./config.md#命令行修改配置) |

### 计算 FLOPs 和参数量

我们提供一个计算 FLOPs 和参数量的方法，首先我们使用以下命令安装依赖。

```shell
pip install fvcore
```

计算 FLOPs 和参数量的脚本使用方法如下：

```shell
python tools/analysis_tools/get_flops.py ${config} --shape ${IMAGE_SHAPE}
```

| 参数    | 类型   | 说明                                                               |
| ------- | ------ | ------------------------------------------------------------------ |
| config  | str    | （必须) 配置文件路径。                                             |
| --shape | int\*2 | 计算 FLOPs 使用的图片尺寸，如 `--shape 320 320`。 默认为 `640 640` |

获取 `dbnet_resnet18_fpnc_100k_synthtext.py` FLOPs 和参数量的示例命令如下。

```shell
python tools/analysis_tools/get_flops.py configs/textdet/dbnet/dbnet_resnet18_fpnc_100k_synthtext.py --shape 1024 1024
```

输出如下：

```shell
input shape is  (1, 3, 1024, 1024)
| module                    | #parameters or shape | #flops  |
| :------------------------ | :------------------- | :------ |
| model                     | 12.341M              | 63.955G |
| backbone                  | 11.177M              | 38.159G |
| backbone.conv1            | 9.408K               | 2.466G  |
| backbone.conv1.weight     | (64, 3, 7, 7)        |         |
| backbone.bn1              | 0.128K               | 83.886M |
| backbone.bn1.weight       | (64,)                |         |
| backbone.bn1.bias         | (64,)                |         |
| backbone.layer1           | 0.148M               | 9.748G  |
| backbone.layer1.0         | 73.984K              | 4.874G  |
| backbone.layer1.1         | 73.984K              | 4.874G  |
| backbone.layer2           | 0.526M               | 8.642G  |
| backbone.layer2.0         | 0.23M                | 3.79G   |
| backbone.layer2.1         | 0.295M               | 4.853G  |
| backbone.layer3           | 2.1M                 | 8.616G  |
| backbone.layer3.0         | 0.919M               | 3.774G  |
| backbone.layer3.1         | 1.181M               | 4.842G  |
| backbone.layer4           | 8.394M               | 8.603G  |
| backbone.layer4.0         | 3.673M               | 3.766G  |
| backbone.layer4.1         | 4.721M               | 4.837G  |
| neck                      | 0.836M               | 14.887G |
| neck.lateral_convs        | 0.246M               | 2.013G  |
| neck.lateral_convs.0.conv | 16.384K              | 1.074G  |
| neck.lateral_convs.1.conv | 32.768K              | 0.537G  |
| neck.lateral_convs.2.conv | 65.536K              | 0.268G  |
| neck.lateral_convs.3.conv | 0.131M               | 0.134G  |
| neck.smooth_convs         | 0.59M                | 12.835G |
| neck.smooth_convs.0.conv  | 0.147M               | 9.664G  |
| neck.smooth_convs.1.conv  | 0.147M               | 2.416G  |
| neck.smooth_convs.2.conv  | 0.147M               | 0.604G  |
| neck.smooth_convs.3.conv  | 0.147M               | 0.151G  |
| det_head                  | 0.329M               | 10.909G |
| det_head.binarize         | 0.164M               | 10.909G |
| det_head.binarize.0       | 0.147M               | 9.664G  |
| det_head.binarize.1       | 0.128K               | 20.972M |
| det_head.binarize.3       | 16.448K              | 1.074G  |
| det_head.binarize.4       | 0.128K               | 83.886M |
| det_head.binarize.6       | 0.257K               | 67.109M |
| det_head.threshold        | 0.164M               |         |
| det_head.threshold.0      | 0.147M               |         |
| det_head.threshold.1      | 0.128K               |         |
| det_head.threshold.3      | 16.448K              |         |
| det_head.threshold.4      | 0.128K               |         |
| det_head.threshold.6      | 0.257K               |         |
!!!Please be cautious if you use the results in papers. You may need to check if all ops are supported and verify that the flops computation is correct.
```
