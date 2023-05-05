import os
import time
from PIL import Image
from argparse import ArgumentParser

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
# MMOCR
from mmocr.apis.inferencers import MMOCRInferencer
from mmocr.utils import poly2bbox
from mmocr.utils.polygon_utils import offset_polygon
# SAM
from segment_anything import SamPredictor, sam_model_registry
# Diffusion model
from diffusers import StableDiffusionInpaintPipeline
import sys

sys.path.append('latent_diffusion')
from latent_diffusion.ldm_erase_text import erase_text_from_image, instantiate_from_config, OmegaConf


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--inputs',
        type=str,
        default='imgs/erase_1.jpg',
        help='Input image file or folder path.')
    parser.add_argument(
        '--outdir',
        type=str,
        default='results/erase_1_sam_dilated_iter2_5x5',
        help='Output directory of results.')
    # MMOCR parser
    parser.add_argument(
        '--det',
        type=str,
        default=
        'mmocr_dev/configs/textdet/dbnetpp/dbnetpp_swinv2_base_w16_in21k.py',
        help='Pretrained text detection algorithm. It\'s the path to the '
        'config file or the model name defined in metafile.')
    parser.add_argument(
        '--det-weights',
        type=str,
        # required=True,
        default='checkpoints/mmocr/db_swin_mix_pretrain.pth',
        help='Path to the custom checkpoint file of the selected det model.')
    parser.add_argument(
        '--rec',
        type=str,
        default='mmocr_dev/configs/textrecog/abinet/abinet_20e_st-an_mj.py',
        help='Pretrained text recognition algorithm. It\'s the path to the '
        'config file or the model name defined in metafile.')
    parser.add_argument(
        '--rec-weights',
        type=str,
        default=
        'checkpoints/mmocr/abinet_20e_st-an_mj_20221005_012617-ead8c139.pth',
        help='Path to the custom checkpoint file of the selected recog model.')
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device used for inference. '
        'If not specified, the available device will be automatically used.')
    # SAM Parser
    parser.add_argument(
        "--use_sam",
        type=bool,
        default=True,
        help='Whether to use SAM to segment the character. If you use the '
        'latent-diffusion for erasing, don\'t use the sam can greatly improve '
        'the erasing quality.')
    parser.add_argument(
        "--sam_checkpoint",
        type=str,
        default='checkpoints/sam/sam_vit_h_4b8939.pth',
        help="path to checkpoint file")
    parser.add_argument(
        "--sam_type",
        type=str,
        default='vit_h',
        help="path to checkpoint file")
    parser.add_argument(
        "--dilate_iteration",
        type=int,
        default=2,
        help="The dilate iteration to dilate the SAM ouput mask")
    parser.add_argument(
        "--show", action='store_true', help="whether to show the result")
    # Diffusion Erase Model Parser
    parser.add_argument(
        "--diffusion_model",
        type=str,
        default=
        'latent-diffusion',  # Options: latent-diffusion, stable-diffusion
        help="path to checkpoint file")
    parser.add_argument(
        "--sd_ckpt",
        type=str,
        default='diffusers/checkpoints/stable-diffusion-2-inpainting',
        help='If use stable-diffusion for erasing, you can set the local '
        'ckpt file. If want to download from hub, set `None`')
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help='If use stable-diffusion for erasing, you can set prompt '
        'by yourself. If you want the default(`No text, clean background`), '
        'set `None`')
    parser.add_argument(
        "--img_size",
        type=tuple,
        default=(512, 512),
        help='If use latetn-diffusion for erasing, set the ldm-inpainting '
        'image size, also if want to use original size, set `None`')
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help='If use latetn-diffusion for erasing, choose the number of '
        'ddim sampling steps')
    args = parser.parse_args()
    return args


def show_sam_result(img, masks, rec_texts, det_polygons, args):
    # Show results
    plt.figure(figsize=(10, 10))
    # convert img to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    for mask, rec_text, polygon in zip(masks, rec_texts, det_polygons):
        show_mask(mask.cpu(), plt.gca(), random_color=True)
        polygon = np.array(polygon).reshape(-1, 2)
        # convert polygon to closed polygon
        polygon = np.concatenate([polygon, polygon[:1]], axis=0)
        plt.plot(polygon[:, 0], polygon[:, 1], color='r', linewidth=2)
        plt.text(polygon[0, 0], polygon[0, 1], rec_text, color='r')
    if args.show:
        plt.show()
    plt.savefig(os.path.join(args.outdir, f'sam_{i}.png'))


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def numpy2PIL(numpy_image):
    out = Image.fromarray(numpy_image.astype(np.uint8))
    return out


def multi_mask2one_mask(masks):
    masks_length, _, h, w = masks.shape
    for i, mask in enumerate(masks):
        mask_image = mask.cpu().numpy().reshape(h, w, 1)
        whole_mask = mask_image if i == 0 else whole_mask + mask_image
    whole_mask = np.where(whole_mask == False, 0, 255)
    return whole_mask


if __name__ == '__main__':
    args = parse_args()

    # Build MMOCR
    mmocr_inferencer = MMOCRInferencer(
        args.det,
        args.det_weights,
        args.rec,
        args.rec_weights,
        device=args.device)

    # Build SAM
    if args.use_sam:
        sam = sam_model_registry[args.sam_type](checkpoint=args.sam_checkpoint)
        sam = sam.to(args.device)
        sam_predictor = SamPredictor(sam)

    if args.diffusion_model == "stable-diffusion":
        # Build Stable Diffusion Inpainting
        if args.sd_ckpt is not None:
            sd_ckpt = args.sd_ckpt
        else:
            sd_ckpt = "stabilityai/stable-diffusion-2-inpainting"
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            sd_ckpt, torch_dtype=torch.float16)
        pipe = pipe.to(args.device)
    else:
        config = OmegaConf.load("latent_diffusion/inpainting_big/config.yaml")
        model = instantiate_from_config(config.model)
        model.load_state_dict(
            torch.load("checkpoints/ldm/last.ckpt")["state_dict"],
            strict=False)
        model = model.to(args.device)

    # Run
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    ori_inputs = mmocr_inferencer._inputs_to_list(args.inputs)
    for i, ori_input in enumerate(ori_inputs):
        print(f'Processing {ori_input} [{i + 1}]/[{len(ori_inputs)}]')
        img = cv2.imread(ori_input)

        # MMOCR inference
        start = time.time()
        result = mmocr_inferencer(
            img, save_vis=True, out_dir=args.outdir)['predictions'][0]
        end = time.time()
        rec_texts = result['rec_texts']
        det_polygons = result['det_polygons']
        print(
            f"The MMOCR for detecting the text has finished, costing time {end-start}s"
        )

        h, w, c = img.shape
        if args.use_sam:
            # Transform the bbox
            det_bboxes = torch.tensor(
                np.array([poly2bbox(poly) for poly in det_polygons]),
                device=sam_predictor.device)
            transformed_boxes = sam_predictor.transform.apply_boxes_torch(
                det_bboxes, img.shape[:2])
            # SAM inference
            start = time.time()
            sam_predictor.set_image(img, image_format='BGR')
            masks, _, _ = sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )
            end = time.time()
            ori_mask = multi_mask2one_mask(masks=masks)
            # Dilate the mask region to promote the following erasing quality
            mask_img = ori_mask[:, :, 0].astype('uint8')
            kernel = np.ones((5, 5), np.int8)
            whole_mask = cv2.dilate(
                mask_img, kernel, iterations=args.dilate_iteration)
            cv2.imwrite(
                os.path.join(args.outdir, f'whole_mask.jpg'), whole_mask)
            # Show result
            show_sam_result(
                img=img,
                masks=masks,
                rec_texts=rec_texts,
                det_polygons=det_polygons,
                args=args)
            print(
                f"The SAM for segment the text has finished, costing time {end-start}s"
            )
        else:
            whole_mask = np.zeros((h, w, c), np.uint8)
            for polygon in det_polygons:
                # expand the polygon with distance 0.1
                expand_poly = offset_polygon(poly=polygon, distance=4).tolist()
                px = [
                    int(expand_poly[i]) for i in range(0, len(expand_poly), 2)
                ]
                py = [
                    int(expand_poly[i]) for i in range(1, len(expand_poly), 2)
                ]
                poly = [[x, y] for x, y in zip(px, py)]
                cv2.fillPoly(whole_mask, [np.array(poly)], (255, 255, 255))
            cv2.imwrite(
                os.path.join(args.outdir, f'whole_mask.jpg'), whole_mask)

        if args.diffusion_model == "stable-diffusion":
            # Data preparation
            sd_img = Image.open(ori_input).convert("RGB").resize((512, 512))
            sd_mask_img = numpy2PIL(
                numpy_image=whole_mask).convert("RGB").resize((512, 512))
            # sd_mask_img.save(os.path.join(args.outdir, f'whole_mask.png'))

            # Stable Diffusion for Erasing
            start = time.time()
            if args.prompt is not None:
                prompt = args.prompt
            else:
                prompt = 'No text, clean background'
            image = pipe(
                prompt=prompt, image=sd_img, mask_image=sd_mask_img).images[0]
            end = time.time()
            # Save image
            image = image.resize((w, h))
            image.save(os.path.join(args.outdir, f'erase_output.jpg'))
            print(
                f"The Stable Diffusion for erasing the text has finished, costing time {end-start}"
            )

        else:
            start = time.time()
            mask_pil_image = numpy2PIL(numpy_image=whole_mask)
            image = erase_text_from_image(
                img_path=ori_input,
                mask_pil_img=mask_pil_image,
                model=model,
                device=args.device,
                opt=args)
            end = time.time()
            image = image.resize((w, h))
            image.save(f"{args.outdir}/erased_image.jpg")
            print(
                f"The Latent Diffusion for erasing the text has finished, costing time {end-start}"
            )
