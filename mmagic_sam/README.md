# MMagic-SAM

<div align=center>
<img src="https://user-images.githubusercontent.com/12782558/234457157-efb21b41-f06a-40be-8274-8e63a8fe19e6.png"/>
</div>

This folder contains interesting usages of using MMagic and SAM together.

## 📄 Table of Contents

- [🛠️ Installation](#installation)
- [⬇️ Download](#download)
- [🚀 Play](#play)

## Installation

We first create a conda env, and then install MMagic and SAM in it.

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

## Download

Download SAM checkpoints.

```shell
mkdir -p checkpoints/sam
wget -O checkpoints/sam/sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

```

## Play

### Play controlnet animation with SAM

***Instruction***

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

***output example***

Below is a video input and output result for example. Try to make your new videos!

<div align="center">
  <video src="https://user-images.githubusercontent.com/12782558/232666513-a735fadb-b92b-4807-ba32-8a38b1514622.mp4" width=1024/>
</div>

***Method explanation***

We get the final video through the following steps:

1. Split the input video into frames

2. Call the controlnet animation model through the inference API of MMagic to modify each frame of the video to make it an AI animation

3. Use the stable diffusion in MMagic to generate a background image that matches the semantics of the animation content

4. Use SAM to predict the mask of the person in the animation

5. Replace the background in the animation with the background image we generated
