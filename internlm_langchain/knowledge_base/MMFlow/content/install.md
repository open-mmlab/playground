# 安装

<!-- TOC -->

- [安装](#%E5%AE%89%E8%A3%85)
  - [依赖](#%E4%BE%9D%E8%B5%96)
  - [准备环境](#%E5%87%86%E5%A4%87%E7%8E%AF%E5%A2%83)
  - [安装 MMFlow](#%E5%AE%89%E8%A3%85-mmflow)
  - [从零开始安装](#%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AE%89%E8%A3%85)
  - [安装验证](#%E5%AE%89%E8%A3%85%E9%AA%8C%E8%AF%81)

<!-- TOC -->

## 依赖

- Linux
- Python 3.6+
- PyTorch 1.5 或更高
- CUDA 9.0 或更高
- NCCL 2
- GCC 5.4 或更高
- [mmcv](https://github.com/open-mmlab/mmcv) 1.3.15 或更高

## 准备环境

a. 创建并激活 conda 虚拟环境

```shell
conda create -n openmmlab python=3.7 -y
conda activate openmmlab
```

b. 按照 [PyTorch 官方文档](https://pytorch.org/) 安装 PyTorch 和 torchvision

注：确保 CUDA 编译版本和 CUDA 运行版本相匹配。 用户可以参照 [PyTorch 官网](https://pytorch.org/) 对预编译包所支持的 CUDA 版本进行核对。

`例1`：如果 `/usr/local/cuda` 文件夹下已安装了 CUDA 10.2 版本，并需要安装最新版本的 PyTorch。

```shell
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

`例2`：如果 `/usr/local/cuda` 文件夹下已安装了 CUDA 9.2 版本，并需要安装 PyTorch 1.7.0。

```shell
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=9.2 -c pytorch
```

如果你从源码编译 PyTorch 而不是安装的预编译版本的话，你可以使用更多 CUDA 版本（例如9.0）。

c. 安装 MMCV，我们推荐按照如下方式安装预编译的 MMCV

```shell
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
```

请按照 CUDA 和 Pytorch 的版本 替换链接中的 `{cu_version}` and `{torch_version}`， 例如，当环境中已安装 CUDA 10.2 和 PyTorch 1.10.0 时安装最新版本的 `mmcv-full`，可以使用以下命令：

```shell
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.10.0/index.html
```

[这里](https://github.com/open-mmlab/mmcv#installation) 可以查看不同 MMCV 的安装方式。

或者，也可以按以下命令从源码编译 MMCV

```shell
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
MMCV_WITH_OPS=1 pip install -e .  # 这种方式可以安装包含 CUDA op 的 mmcv-full
# pip install -e .  # 或者安装不包含 CUDA op 的 mmcv
cd ..
```

**重要**　如果之前安装了 `mmcv`，在安装 `mmcv` 之前需要先卸载 `pip uninstall mmcv`。因为同时安装了 `mmcv` 和 `mmcv-full`，
会遇到 `ModuleNotFoundError` 的错误。

## 安装 MMFlow

a. 克隆 MMFlow 代码库

```shell
git clone https://github.com/open-mmlab/mmflow.git
cd mmflow
```

b. 安装相关依赖和 MMFlow

```shell
pip install -r requirements/build.txt
pip install -v -e .  # or "python setup.py develop"
```

注：

1. git commit 的 id 将会被写到版本号中，如 0.6.0+2e7045c。这个版本号也会被保存到训练好的模型中。

2. 根据上述步骤， MMFlow 就会以 `dev` 模式被安装，任何本地的代码修改都会立刻生效，不需要再重新安装一遍（除非用户提交了 commits，并且想更新版本号）。

3. 如果用户想使用 `opencv-python-headless` 而不是 `opencv-python`，可在安装 `MMCV` 前安装 `opencv-python-headless`。

## 从零开始安装

假设已安装了 10.1 版本的 CUDA，以下是完整的安装 MMFlow 的命令。

```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab

conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch -y

# install latest mmcv
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html

# install mmflow
git clone https://github.com/open-mmlab/mmflow.git
cd mmflow
pip install -r requirements/build.txt
pip install -v -e .
```

## 安装验证

我们可以利用以下步骤初始化模型并对图像做推理，来检查 MMFlow 是否正确安装。

```python
from mmflow.apis import inference_model, init_model

config_file = 'configs/pwcnet/pwcnet_ft_4x1_300k_sintel_final_384x768.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
# url: https://download.openmmlab.com/mmflow/pwcnet/pwcnet_ft_4x1_300k_sintel_final_384x768.pth
checkpoint_file = 'checkpoints/pwcnet_ft_4x1_300k_sintel_final_384x768.pth'
device = 'cuda:0'
# init a model
model = init_model(config_file, checkpoint_file, device=device)
# inference the demo image
inference_model(model, 'demo/frame_0001.png', 'demo/frame_0002.png')
```

上述命令应该在完成安装后成功运行。
