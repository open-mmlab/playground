# Introduction

I make a QR Code Generator by Stable Diffusion and Controlnet.

Must set `mmagic/models/archs/wrapper.py` line 90:    

`self.model = module_cls.from_pretrained(from_pretrained,use_safetensors=True, *args,**kwargs)`

# Demo

A simple demo is provided.

```shell
python demo/qrcode_inference_demo.py \
       --config controlnet-brightness.py \
       --qrcode_img 'test.png' \
       --prompt 'dreamlikeart, an zebra' \
       --negative_prompt 'ugly, bad quality' \
       --resize 440 640 \
       --output_size 440 640 \
       --num_inference_steps 50 \
       --guidance_scale 7.5 \
       --unet_model 'dreamlike-art/dreamlike-diffusion-1.0' \
       --vae_model 'dreamlike-art/dreamlike-diffusion-1.0' \
       --controlnet_model 'ioclab/control_v1p_sd15_brightness' \
       --controlnet_conditioning_scale 0.7 \
       --num_generated_img 5 \
       --save_path 'output'
```

The generated images will be save in `output/[num]_sample.png`.

If the generated QR code is not recognizable, try increasing `controlnet_conditioning_scale`.

One result display (using the parameters of the demo above)`qrcode_example.png`.
