import cv2
from argparse import ArgumentParser
import PIL.Image as Image
import torch
import numpy as np
# MMOCR
from mmocr.apis.inferencers import MMOCRInferencer
from mmocr.utils import poly2bbox
# SAM
from segment_anything import SamPredictor, sam_model_registry

from diffusers import StableDiffusionInpaintPipeline


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--img_path', type=str, help='Input image file path.')
    parser.add_argument(
        '--outdir',
        type=str,
        default='results/',
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
        'checkpoints/mmocr/abinet_20e_st-an_mj_20221005_012617-ead8c139.pth',  # noqa
        help='Path to the custom checkpoint file of the selected recog model.')
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device used for inference. '
        'If not specified, the available device will be automatically used.')
    # SAM Parser
    parser.add_argument(
        "--sam_checkpoint",
        type=str,
        required=True,
        default='checkpoints/sam/sam_vit_h_4b8939.pth',
        help="path to checkpoint file")
    parser.add_argument(
        "--sam_type",
        type=str,
        default='vit_h',
        help="path to checkpoint file")
    parser.add_argument(
        "--prompt",
        type=str,
        default='Text like a cake',
        help="Prompt for inpainting")
    parser.add_argument(
        "--select_index",
        type=int,
        default=0,
        help="Select the index of the text to inpaint")
    parser.add_argument(
        "--show", action='store_true', help="whether to show the result")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    # MMOCR
    mmocr_inferencer = MMOCRInferencer(
        args.det_config,
        args.det_weight,
        args.rec_config,
        args.rec_weight,
        device=args.device)
    # SAM
    sam = sam_model_registry[args.sam_type](checkpoint=args.sam_checkpoint)
    sam = sam.to(args.device)
    sam_predictor = SamPredictor(sam)
    # Diffuser
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    img = cv2.imread(args.img_path)
    result = mmocr_inferencer(img)['predictions'][0]
    rec_texts = result['rec_texts']
    det_polygons = result['det_polygons']
    det_bboxes = torch.tensor(
        np.array([poly2bbox(poly) for poly in det_polygons]),
        device=sam_predictor.device)
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(
        det_bboxes, img.shape[:2])
    # SAM inference
    sam_predictor.set_image(img, image_format='BGR')
    masks, _, _ = sam_predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    # Diffuser inference
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    ori_img_size = img.size
    mask = masks[args.select_index][0].cpu().numpy()
    mask = Image.fromarray(mask)
    image = pipe(
        prompt=args.prompt,
        image=img.resize((512, 512)),
        mask_image=mask.resize((512, 512))).images[0]
    image = image.resize(ori_img_size)
    image.save('test_out.png')
