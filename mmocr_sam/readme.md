![](imgs/logo.png)

# Optical Character Recognition with Segment Anything (OCR-SAM)

## 🐇 Introduction 🐙
Can [SAM](https://github.com/facebookresearch/segment-anything) be applied to OCR? We take a simple try to combine two off-the-shelf OCR models in [MMOCR](https://github.com/open-mmlab/mmocr) with SAM to develop some OCR-related application demos, including **[SAM for Text](#sam-for-text)**, **[Text Removal](#erasing)** and **[Text Inpainting](#inpainting)**. And we also provide a **[WebUI by gradio](#run-webui)** to give a better interaction.  

*Note: We will continue to update and maintain this repo, and develop more OCR-related advanced applications demo to the community. **Welcome anyones to join who have the idea and want to contribute to our repo**.*

## 📅 Updates 👀
- **2023.04.12**: Repository Release
- **2023.04.12**: Supported the [Inpainting](#inpainting🥸) combined with DBNet++, SAM and ControlNet.
- **2023.04.11**: Supported the [Erasing](#erasing🤓) combined with DBNet++, SAM and Latent-Diffusion / Stable-Diffusion.
- **2023.04.10**: Supported the [SAM for text](#sam-for-text🧐) combined tieh DBNet++ and SAM.

## 📸 Demo Zoo 🔥

This project includes:
- [x] [SAM for Text](#sam-for-text🧐): DBNet++ + SAM
![](imgs/sam_vis.png)
- [x] [Erasing](#erasing🤓): DBNet++ + SAM + Latent-Diffusion / Stable Diffusion 
![](imgs/erase_vis.png)
- [x] [Inpainting](#inpainting🥸): DBNet++ + SAM + Stable Diffusion
![](imgs/inpainting_vis.png)


## 🚧 Installation 🛠️
### Prerequisites

- Linux | Windows
- Python 3.7
- Pytorch 1.6 or higher
- CUDA 11.3

### Environment Setup
Clone this repo:
```
git clone https://github.com/yeungchenwa/OCR-SAM.git
```
**Step 0**: Create a conda environment and activate it.
```
conda create --n ocr-sam python=3.8 -y
conda activate ocr-sam
```
**Step 1**: Install related version Pytorch following [here](https://pytorch.org/get-started/previous-versions/).

**Step 2**: Install the mmengine, mmcv, mmdet, mmcls, mmocr.
```
pip install -U openmim
mim install mmengine
mim install 'mmcv==2.0.0rc4'
mim install 'mmdet==3.0.0rc5'
mim install 'mmcls==1.0.0rc5'

# Install the mmocr from source
cd OCR-SAM/mmocr_dev
pip install -v -e .
```

**Step 3**: Prepare for the diffusers and latent-diffusion.
```
# Install the diffusers
pip install diffusers

# Install the pytorch_lightning for ldm
conda install pytorch-lightning -c conda-forge
```

## 📒 Model checkpoints 🖥

The SceneTextDetector is trained by a lot of scene text datasets, the effects are generic and not limited to one dataset (e.g. ICDAR2015). **Checkpoint download link is [here](https://drive.google.com/file/d/1r3B1xhkyKYcQ9SR7o9hw9zhNJinRiHD-/view?usp=share_link)**.  

And you should make dir following:  
```
mkdir mmocr_dev/checkpoints 
```
Then put the ckpt to the path `mmocr_dev/checkpoints`.

Download the rest of checkpints to the related path (If you've done, ignore the following):
```
mkdir segment-anything-main/checkpoints latent_diffusion/checkpoints

# mmocr recognizer ckpt
wget -O mmocr_dev/checkpoints/abinet_20e_st-an_mj_20221005_012617-ead8c139.pth https://download.openmmlab.com/mmocr/textrecog/abinet/abinet_20e_st-an_mj/abinet_20e_st-an_mj_20221005_012617-ead8c139.pth

# sam ckpt, more details: https://github.com/facebookresearch/segment-anything#model-checkpoints
wget -O segment-anything-main/checkpoints/sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# ldm ckpt
wget -O latent_diffusion/checkpoints/last.ckpt https://heibox.uni-heidelberg.de/f/4d9ac7ea40c64582b7c9/?dl=1
```

## 🏃🏻‍♂️ Run Demo 🏊‍♂️

### **SAM for Text**🧐

Run the following script:
```
python mmocr_sam.py \
    --inputs /YOUR/INPUT/IMG_PATH \ 
    --outdir /YOUR/OUTPUT_DIR \ 
    --device cuda \ 
```
- `--inputs`: the path to your input image. 
- `--outdir`: the dir to your output. 
- `--device`: the device used for inference. 

### **Erasing**🤓

In this application demo, we use the [latent-diffusion-inpainting](https://github.com/CompVis/latent-diffusion#inpainting) to erase, or the [Stable-Diffusion-inpainting](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/inpaint) with text prompt to erase, which you can choose one of both by the parameter `--diffusion_model`. Also, you can choose whether to use the SAM ouput mask to erase by the parameter `--use_sam`. More implementation **details** are listed [here](docs/erase_details.md)

Run the following script:
```
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

**Run the WebUI**: see [here](#📺-run-webui-📱)

**Note: The first time you run may cost some time, becasuse downloading the stable-diffusion ckpt cost a lot, wait patiently👀**

### **Inpainting**
More implementation **details** are listed [here](docs/inpainting_details.md)

Run the following script:
```
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

### **Run WebUI**
This repo also provides the WebUI(decided by gradio), inculding the Erasing and Inpainting.  

Before running the script, you should install the gradio package:
```
pip install gradio
```

#### Erasing
```
python mmocr_sam_erase_app.py
```
- **Example**:  

**Detector and Recognizer WebUI Result**
![](imgs/webui_detect_vis.png) 

**Erasing WebUI Result**
![](imgs/webui_erase_visit.png)  

In our WebUI, user can interactly choose the SAM output and the diffusion model. Especially, user can choose which text to be erased.

#### Inpainting🥸
```
python mmocr_sam_inpainting_app.py
```
- Example:  

**Inpainting WebUI Result**
![](imgs/webui_inpainting_vis.png)

**Note: Before you open the web, it may cost some time, wait patiently👀** 

## 💗 Acknowledgement
- [segment-anything](https://github.com/facebookresearch/segment-anything)
- [latent-diffusion](https://github.com/CompVis/latent-diffusion)
- [mmocr](https://github.com/open-mmlab/mmocr)