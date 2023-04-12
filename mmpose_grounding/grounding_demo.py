# Copyright (c) OpenMMLab. All rights reserved.
# Refer from https://github.com/IDEA-Research/Grounded-Segment-Anything
import argparse
import os

import groundingdino.datasets.transforms as T
import torch
from groundingdino.models import build_model
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from mmengine.config import Config
from mmengine.utils import ProgressBar
from PIL import Image

from core.utils import get_file_list


def parse_args():
    parser = argparse.ArgumentParser('mmpose grounding Demo', add_help=True)
    parser.add_argument('image', type=str, help='path to image file')
    parser.add_argument('det_config', type=str, help='path to det config file')
    parser.add_argument('det_weight', type=str, help='path to det weight file')
    parser.add_argument(
        '--out-dir',
        '-o',
        type=str,
        default='outputs',
        help='output directory')
    parser.add_argument(
        '--box-thr', '-b', type=float, default=0.3, help='box threshold')
    parser.add_argument(
        '--device', '-d', default='cuda:0', help='Device used for inference')

    # GroundingDINO param
    parser.add_argument(
        '--text-prompt', '-t', default='human', type=str, help='text prompt')
    parser.add_argument(
        '--text-thr', type=float, default=0.25, help='text threshold')

    return parser.parse_args()


grounding_dino_transform = T.Compose([
    T.RandomResize([800], max_size=1333),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def __build_grounding_dino_model(args):
    gdino_args = Config.fromfile(args.det_config)
    model = build_model(gdino_args)
    checkpoint = torch.load(args.det_weight, map_location='cpu')
    model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    model.eval()
    return model


def build_detecter(args):
    if 'GroundingDINO' in args.det_config:
        detecter = __build_grounding_dino_model(args)
    else:
        pass
    return detecter


def run_detector(model, image_path, args):
    pred_dict = {}

    if 'GroundingDINO' in args.det_config:
        image_pil = Image.open(image_path).convert('RGB')  # load image
        image, _ = grounding_dino_transform(image_pil, None)  # 3, h, w

        text_prompt = args.text_prompt
        text_prompt = text_prompt.lower()
        text_prompt = text_prompt.strip()
        if not text_prompt.endswith('.'):
            text_prompt = text_prompt + '.'

        image = image.to(next(model.parameters()).device)

        with torch.no_grad():
            outputs = model(image[None], captions=[text_prompt])

        logits = outputs['pred_logits'].cpu().sigmoid()[0]  # (nq, 256)
        boxes = outputs['pred_boxes'].cpu()[0]  # (nq, 4)

        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > args.box_thr
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(text_prompt)

        # build pred
        pred_labels = []
        pred_scores = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > args.text_thr,
                                                  tokenized, tokenlizer)
            pred_labels.append(pred_phrase)
            pred_scores.append(str(logit.max().item())[:4])

        pred_dict['labels'] = pred_labels
        pred_dict['scores'] = pred_scores

        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]
        pred_dict['boxes'] = boxes_filt
    else:
        pass

    return model, pred_dict


def main():
    args = parse_args()

    out_dir = args.out_dir

    det_model = build_detecter(args)
    det_model = det_model.to(args.device)

    os.makedirs(out_dir, exist_ok=True)

    files, source_type = get_file_list(args.image)
    progress_bar = ProgressBar(len(files))
    for image_path in files:
        save_path = os.path.join(out_dir, os.path.basename(image_path))
        det_model, pred_dict = run_detector(det_model, image_path, args)
        print(pred_dict)

        if pred_dict['boxes'].shape[0] == 0:
            print('No objects detected !')
            continue

        progress_bar.update()


if __name__ == '__main__':
    main()
