# MMOCR-SAM

<div align=center>
<img src="https://user-images.githubusercontent.com/65173622/231803460-495cf11f-8e2e-4c95-aa48-b163fc7fbbab.png"/>
</div>

该仓库搬运自 [OCR-SAM](https://github.com/yeungchenwa/OCR-SAM)


## 安装

```shell
conda create --n ocr-sam python=3.8 -y
conda activate ocr-sam
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

pip install -U openmim
mim install mmengine
mim install mmocr
mim install 'mmcv==2.0.0rc4'
mim install 'mmdet==3.0.0rc5'
mim install 'mmcls==1.0.0rc5'

pip install git+https://github.com/facebookresearch/segment-anything.git
pip install -r requirements.txt

pip install gradio
pip install diffusers
pip install pytorch-lightning==2.0.1.post0
```

## 下载权重


我们使用 SwinV2-B 作为骨干网络，在一系列数据集上联合训练了一个 DBNet++ 作为通用检测器，**下载地址 [Google Drive (1G)](https://drive.google.com/file/d/1r3B1xhkyKYcQ9SR7o9hw9zhNJinRiHD-/view?usp=share_link)**.  

创建路径
```bash
mkdir checkpoints
mkdir checkpoints/mmocr
mkdir checkpoints/sam
mkdir checkpoints/ldm
mv db_swin_mix_pretrain.pth checkpoints/mmocr
```

下载权重
```bash

# mmocr recognizer ckpt
wget -O checkpoints/mmocr/abinet_20e_st-an_mj_20221005_012617-ead8c139.pth https://download.openmmlab.com/mmocr/textrecog/abinet/abinet_20e_st-an_mj/abinet_20e_st-an_mj_20221005_012617-ead8c139.pth

# sam ckpt, more details: https://github.com/facebookresearch/segment-anything#model-checkpoints
wget -O checkpoints/sam/sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# ldm ckpt
wget -O checkpoints/ldm/last.ckpt https://heibox.uni-heidelberg.de/f/4d9ac7ea40c64582b7c9/?dl=1
```

## 使用说明

### SAM + MMOCR
- 对单张图片或者一个图像文件夹进行简单的 MMOCR + SAM 的前向推理

  ```shell
  python mmocr_sam.py \
      --inputs /YOUR/INPUT/IMG_PATH \ 
      --outdir /YOUR/OUTPUT_DIR \ 
      --device cuda \ 
  ```

  - `--inputs`: the path to your input image. 
  - `--outdir`: the dir to your output. 
  - `--device`: the device used for inference. 

### Text Removal

- 在这个演示 Demo 中, 我们使用 [latent-diffusion-inpainting](https://github.com/CompVis/latent-diffusion#inpainting) 或者带文本 prompt 的 [Stable-Diffusion-inpainting](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/inpaint) 来进行文本擦除, 可以通过 `--diffusion_model` 来选择擦除模型.

- 
  ```bash
  python mmocr_sam_erase.py \ 
      --inputs /YOUR/INPUT/IMG_PATH \ 
      --outdir /YOUR/OUTPUT_DIR \ 
      --device cuda \ 
      --use_sam True \ 
      --dilate_iteration 2 \ 
      --diffusion_model \ 
      --sd_ckpt None \ 
      --prompt None \ 
      --img_size (512, 512) \ 
  ```
  - `--inputs `: 输入图片路径.
  - `--outdir`: 输出图片路径. 
  - `--device`: 运行设备. 
  - `--use_sam`: 是否使用SAM进行分割，如不使用，则使用 MMOCR 检测结果.
  - `--dilate_iteration`: 对分割结果进行膨胀变换的次数.
  - `--diffusion_model`: 选择 'latent-diffusion' 或者 'stable-diffusion'.
  - `--sd_ckpt`: Stable Diffusion 权重路径，如不指定，则会默认下载.
  - `--prompt`: 用于 Stable Diffusion 的文本提示
  - `--img_size`: Latent Diffusion 的输入尺寸.  

- 我们强烈建议使用 Gradio 搭建的 **WebUI** 来运行 demo.

  ```shell 
  python mmocr_sam_erase_app.py
  ```

<div align=center>
<img src="https://user-images.githubusercontent.com/65173622/231764540-a5403ad3-fab5-4dc8-9b82-f8a9643ab0f4.png"/>
</div>


### Text Inpainting
- 我们使用 StablediffusionInpainter 来对图像中的文本进行编辑

  ```bash
  python mmocr_sam_inpainting.py \
      --img_path /YOUR/INPUT/IMG_PATH \ 
      --outdir /YOUR/OUTPUT_DIR \ 
      --device cuda \ 
      --prompt YOUR_PROMPT \ 
      --select_index 0 \ 
  ```
  - `--img_path`: 输入图片路径 
  - `--outdir`: 输出图片路径
  - `--device`: 推理设备 
  - `--prompt`: 文本提示词
  - `--select_index`: 选取第几个文本框进行编辑

- 我们强烈建议使用 Gradio 搭建的 **WebUI** 来运行 demo.

  ```shell 
  python mmocr_sam_inpainting_app.py
  ```

<div align=center>
<img src="https://user-images.githubusercontent.com/65173622/231764419-76860cd3-3f9f-4662-8fd3-6b74795b36e9.png"/>
</div>

