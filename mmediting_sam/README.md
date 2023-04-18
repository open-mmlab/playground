# MMEditing-SAM

<div align=center>
<img src="https://user-images.githubusercontent.com/12782558/232700025-a7bfe119-9eb5-46d2-b57c-ba7dc8c40d83.png"/>
</div>

This folder contains interesting usages of using MMEditing and SAM together.

## 📄 Table of Contents

- [🛠️ Installation](#installation)
- [⬇️ Download](#download)
- [🚀 Play](#play)

## Installation

We first create a conda env, and then install MMEditing and SAM in it.

```shell
# create env and install torch
conda create -n mmedit-sam python=3.8 -y
conda activate mmedit-sam
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# install mmediting
pip install openmim
mim install mmengine "mmcv>=2.0.0"
git clone -b dev-1.x https://github.com/open-mmlab/mmediting.git
pip install -e ./mmediting

# install sam
pip install git+https://github.com/facebookresearch/segment-anything.git

# you may need ffmpeg to get frames or make video
sudo apt install ffmpeg
```

## Download

Download SAM checkpoints.

```shell
mkdir -p checkpoints/sam
wget -O checkpoints/sam/sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

```

## Play

### Play controlnet animation with SAM

Find a video clip that you want to edit with and get frames.

```shell
mkdir -p inputs/demo_video
ffmpeg -i your_video.mp4 inputs/demo_video/%04d.jpg
```

Run the script.

```shell
python play_controlnet_animation_sam.py
```

Make video with output frames.

```shell
ffmpeg -r 10 -i results/final_frames/%04d.jpg -b:v 30M -vf fps=10 results/final_frames.mp4
```

Below is a video input and output result for example. Try to make your new videos!

<div align="center">
  <video src="https://user-images.githubusercontent.com/12782558/232666513-a735fadb-b92b-4807-ba32-8a38b1514622.mp4" width=1024/>
</div>
