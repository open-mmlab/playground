# MMOCR-SAM


[中文文档](README_zh-CN.md)

<div align=center>
<img src="https://user-images.githubusercontent.com/65173622/231803460-495cf11f-8e2e-4c95-aa48-b163fc7fbbab.png"/>
</div>

The project is migrated from [OCR-SAM](https://github.com/yeungchenwa/OCR-SAM), which combines MMOCR with Segment Anything.


## Installation

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

## Download checkpoints

We retrain DBNet++ with Swin Transformer V2 as the backbone on a combination of multiple scene text datsets (e.g. HierText, TextOCR). **Checkpoint for DBNet++ on [Google Drive (1G)](https://drive.google.com/file/d/1r3B1xhkyKYcQ9SR7o9hw9zhNJinRiHD-/view?usp=share_link)**.  

And you should make dir following:  
```bash
mkdir checkpoints
mkdir checkpoints/mmocr
mkdir checkpoints/sam
mkdir checkpoints/ldm
mv db_swin_mix_pretrain.pth checkpoints/mmocr
```

Download the rest of checkpints to the related path (If you've done, ignore the following):
```bash

# mmocr recognizer ckpt
wget -O checkpoints/mmocr/abinet_20e_st-an_mj_20221005_012617-ead8c139.pth https://download.openmmlab.com/mmocr/textrecog/abinet/abinet_20e_st-an_mj/abinet_20e_st-an_mj_20221005_012617-ead8c139.pth

# sam ckpt, more details: https://github.com/facebookresearch/segment-anything#model-checkpoints
wget -O checkpoints/sam/sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# ldm ckpt
wget -O checkpoints/ldm/last.ckpt https://heibox.uni-heidelberg.de/f/4d9ac7ea40c64582b7c9/?dl=1
```

## Usage

### SAM for Text
- Inference MMOCR-SAM with a single image or an image folder and obtain visualization result.

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

- In this application demo, we use the [latent-diffusion-inpainting](https://github.com/CompVis/latent-diffusion#inpainting) to erase, or the [Stable-Diffusion-inpainting](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/inpaint) with text prompt to erase, which you can choose one of both by the parameter `--diffusion_model`. Also, you can choose whether to use the SAM ouput mask to erase by the parameter `--use_sam`.

- Run the following script:
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
  - `--inputs `: the path to your input image.
  - `--outdir`: the dir to your output. 
  - `--device`: the device used for inference. 
  - `--use_sam`: whether to use sam for segment.
  - `--dilate_iteration`: iter to dilate the SAM's mask.
  - `--diffusion_model`: choose 'latent-diffusion' or 'stable-diffusion'.
  - `--sd_ckpt`: path to the checkpoints of stable-diffusion.
  - `--prompt`: the text prompt when use the stable-diffusion, set 'None' if use the default for erasing.
  - `--img_size`: image size of latent-diffusion.  

- We suggest use our **WebUI** build with gradio to run the demo.

  ```shell 
  python mmocr_sam_erase_app.py
  ```

<div align=center>
<img src="https://user-images.githubusercontent.com/65173622/231764540-a5403ad3-fab5-4dc8-9b82-f8a9643ab0f4.png"/>
</div>


### Text Inpainting
- We use StablediffusionInpainter to inpaint the text in the image.

  ```bash
  python mmocr_sam_inpainting.py \
      --img_path /YOUR/INPUT/IMG_PATH \ 
      --outdir /YOUR/OUTPUT_DIR \ 
      --device cuda \ 
      --prompt YOUR_PROMPT \ 
      --select_index 0 \ 
  ```
  - `--img_path`: the path to your input image. 
  - `--outdir`: the dir to your output. 
  - `--device`: the device used for inference. 
  - `--prompt`: the text prompt.
  - `--select_index`: select the index of the text to inpaint.

- We suggest use our **WebUI** build with gradio to run the demo.

  ```shell 
  python mmocr_sam_inpainting_app.py
  ```

<div align=center>
<img src="https://user-images.githubusercontent.com/65173622/231764419-76860cd3-3f9f-4662-8fd3-6b74795b36e9.png"/>
</div>

