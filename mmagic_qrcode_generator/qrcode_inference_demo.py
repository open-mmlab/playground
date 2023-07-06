# Copyright (c) OpenMMLab. All rights reserved.

import cv2
import numpy as np
import mmcv
from mmengine import Config
from PIL import Image
import os
from argparse import ArgumentParser

from mmagic.registry import MODELS
from mmagic.utils import register_all_modules


def parse_args():
    parser = ArgumentParser()

    # input
    parser.add_argument(
        '--qrcode_img', type=str, default=None, help='Input QRcode image file.')
    parser.add_argument(
        '--prompt', type=str, default=None, help='Input prompt.')
    parser.add_argument(
        '--negative_prompt', type=str, default=None, help='Input negative prompt.')
    parser.add_argument(
        '--config', type=str, default=None, help='Input config.')

    # parameters
    parser.add_argument(
        '--resize', nargs='+', help='Resize the input QRcode image, must be a multiple of 8')
    parser.add_argument(
        '--output_size', nargs='+', help='Output image size, must be a multiple of 8')
    parser.add_argument(
        '--num_inference_steps', type=int, default=50, help='Number of inference steps.')
    parser.add_argument(
        '--guidance_scale', type=float, default=7.5, help='guidance scale.')
    parser.add_argument(
        '--controlnet_conditioning_scale', type=float, default=0.6, help='Controlnet conditioning scale.')
    parser.add_argument(
        '--num_generated_img', type=int, default=5, help='Number of generated images.')
    parser.add_argument(
        '--save_path', type=str, default=None, help='Generated image save path.')

    # models
    parser.add_argument(
        '--unet_model', type=str, default=None, help='Change unet mdoel.')
    parser.add_argument(
        '--vae_model', type=str, default=None, help='Change vae mdoel.')
    parser.add_argument(
        '--controlnet_model', type=str, default=None, help='Change controlnet mdoel.')
    
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    register_all_modules()

    cfg = Config.fromfile(args.config)
    cfg.model.unet.from_pretrained = args.unet_model
    cfg.model.vae.from_pretrained =  args.vae_model
    cfg.model.controlnet.from_pretrained = args.controlnet_model


    cfg.model.init_cfg['type'] = 'convert_from_unet'
    controlnet = MODELS.build(cfg.model).cuda()

    # call init_weights manually to convert weight
    controlnet.init_weights()

    prompt =  args.prompt
    negative_prompt = args.negative_prompt
    control_path = args.qrcode_img
    control_img = mmcv.imread(control_path)
    control_img = cv2.resize(control_img, (int(args.resize[0]),int(args.resize[1])))
    control_img = control_img[:,:,0:1]
    control_img = np.concatenate([control_img]*3, axis=2)
    control = Image.fromarray(control_img)

    num_inference_steps = args.num_inference_steps
    guidance_scale = args.guidance_scale
    num_images_per_prompt = 1
    controlnet_conditioning_scale = args.controlnet_conditioning_scale
    height=int(args.resize[1])
    width=int(args.resize[0])

    num = args.num_generated_img
    save_path = args.save_path

    for i in range(num):
        output_dict = controlnet.infer(
                    prompt = prompt, 
                    control = control,
                    height = height,
                    width = width,
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
                    num_inference_steps=num_inference_steps, 
                    guidance_scale=guidance_scale,
                    num_images_per_prompt=num_images_per_prompt,
                    negative_prompt=negative_prompt,
                    )
        samples = output_dict['samples']
        savepath = os.path.join(save_path, str(i)+'_sample.png')
        samples[0].save(savepath)

if __name__ == '__main__':
    main()