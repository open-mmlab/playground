# MMagic-SAM

<div align=center>
<img src="https://user-images.githubusercontent.com/12782558/234457157-efb21b41-f06a-40be-8274-8e63a8fe19e6.png"/>
</div>

这个文件夹下包含了将 MMagic 和 SAM 一起使用的有趣玩法

## 📄 目录

- [🛠️ 安装](#安装)
- [⬇️ 下载](#下载)
- [🚀 玩法](#玩法)

## 安装

首先创建一个 conda 环境，然后把 MMagic 和 SAM 安装到里面。

```shell
# create env and install torch
conda create -n mmedit-sam python=3.8 -y
conda activate mmedit-sam
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# install mmagic
pip install openmim
mim install mmengine "mmcv>=2.0.0"
git clone -b dev-1.x https://github.com/open-mmlab/mmagic.git
pip install -e ./mmagic

# install sam
pip install git+https://github.com/facebookresearch/segment-anything.git

# you may need ffmpeg to get frames or make video
sudo apt install ffmpeg
```

## 下载

下载 SAM 的模型。

```shell
mkdir -p checkpoints/sam
wget -O checkpoints/sam/sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

```

## 玩法

### 结合 SAM 一起玩 controlnet 动画

***使用方法***

找一个视频拆出视频帧。

```shell
mkdir -p inputs/demo_video
ffmpeg -i your_video.mp4 inputs/demo_video/%04d.jpg
```

运行脚本。

```shell
python play_controlnet_animation_sam.py
```

用输出的视频帧组合成视频。

```shell
ffmpeg -r 10 -i results/final_frames/%04d.jpg -b:v 30M -vf fps=10 results/final_frames.mp4
```

***输出样例***

下面是我们一个视频的输入输出示例。试一下你自己的视频吧！

<div align="center">
  <video src="https://user-images.githubusercontent.com/12782558/232666513-a735fadb-b92b-4807-ba32-8a38b1514622.mp4" width=1024/>
</div>

***方法解析***

我们通过下面的步骤得到最终的视频：

1. 将输入视频拆成帧

2. 通过 MMagic 的前向接口调用 controlnet animation 模型对每帧视频进行修改，使其变为 AI 动画

3. 使用 MMagic 内的 stable diffusion 生成一张和动画内容语意贴合的背景图片

4. 用 SAM 预测动画中人物的 mask

5. 将动画中的背景替换为我们生成的背景图片
