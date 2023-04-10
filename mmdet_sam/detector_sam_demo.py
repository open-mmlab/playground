import argparse
import os
import torch
from PIL import Image

# Grounding DINO
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import build_sam, SamPredictor
import cv2
import numpy as np
import matplotlib.pyplot as plt
from core.utils import get_file_list
from mmengine.config import Config
from mmengine.utils import ProgressBar


def parse_args():
    parser = argparse.ArgumentParser("Detect-Segment-Anything Demo", add_help=True)
    parser.add_argument("image", type=str, help="path to image file")
    parser.add_argument("det_config", type=str, help="path to det config file")
    parser.add_argument("det_weight", type=str, help="path to det weight file")
    parser.add_argument('--only-det', action="store_true")
    parser.add_argument(
        "--sam-weight", type=str, default='../models/sam_vit_h_4b8939.pth', help="path to checkpoint file"
    )
    parser.add_argument(
        "--out-dir", "-o", type=str, default="outputs", help="output directory"
    )
    parser.add_argument("--box-thr", '-b', type=float, default=0.3, help="box threshold")
    parser.add_argument('--det-device', '-d', default='cuda:0', help='Device used for inference')
    parser.add_argument('--sam-device', '-s', default='cuda:0', help='Device used for inference')
    parser.add_argument("--cpu-off-load", '-c', action="store_true")

    # GroundingDINO param
    parser.add_argument("--text-prompt", '-t', type=str, help="text prompt")
    parser.add_argument("--text-thr", type=float, default=0.25, help="text threshold")

    return parser.parse_args()


def __build_grounding_dino_model(args):
    gdino_args = Config.fromfile(args.det_config)
    model = build_model(gdino_args)
    checkpoint = torch.load(args.det_weight, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model


grounding_dino_transform = T.Compose(
    [
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def build_detecter(args):
    if 'GroundingDINO' in args.det_config:
        detecter = __build_grounding_dino_model(args)
    else:
        raise NotImplementedError
    return detecter


def run_detecter(model, image_path, args):
    if args.cpu_off_load:
        model = model.to(args.det_device)

    if 'GroundingDINO' in args.det_config:
        with_logits = True

        image_pil = Image.open(image_path).convert("RGB")  # load image
        image, _ = grounding_dino_transform(image_pil, None)  # 3, h, w
        # boxes_filt, pred_phrases = get_grounding_output(
        #     model, image, text_prompt, box_threshold, text_threshold, device=device
        # )

        text_prompt = args.text_prompt
        text_prompt = text_prompt.lower()
        text_prompt = text_prompt.strip()
        if not text_prompt.endswith("."):
            text_prompt = text_prompt + "."

        image = image.to(next(model.parameters()).device)

        with torch.no_grad():
            outputs = model(image[None], captions=[text_prompt])

        logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)

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
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > args.text_thr, tokenized, tokenlizer)
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)

        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        if args.cpu_off_load:
            model = model.to('cpu')
        return model, boxes_filt, pred_phrases


def draw_and_save(image, pred_dict, save_path, random_color=True):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    bboxes = pred_dict['boxes'].cpu().numpy()
    for box in bboxes:
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        plt.gca().add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

    if 'masks' in pred_dict:
        masks = pred_dict['masks'].cpu().numpy()
        for mask in masks:
            if random_color:
                color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
            else:
                color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
            h, w = mask.shape[-2:]
            mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
            plt.gca().imshow(mask_image)

    plt.axis('off')
    plt.savefig(save_path)


def main():
    args = parse_args()
    if args.cpu_off_load is True:
        if 'cpu' in args.det_device and 'cpu ' in args.sam_device:
            raise RuntimeError('args.cpu_off_load is an invalid parameter due to '
                               'detection and sam model are on the cpu.')

    only_det = args.only_det
    cpu_off_load = args.cpu_off_load
    out_dir = args.out_dir

    if 'GroundingDINO' in args.det_config:
        assert args.text_prompt

    det_model = build_detecter(args)
    if not cpu_off_load:
        det_model = det_model.to(args.det_device)

    if not only_det:
        sam_model = SamPredictor(build_sam(checkpoint=args.sam_weight))
        if not cpu_off_load:
            sam_model.mode = sam_model.model.to(args.sam_device)

    os.makedirs(out_dir, exist_ok=True)

    files, source_type = get_file_list(args.image)
    progress_bar = ProgressBar(len(files))
    for image_path in files:
        save_path = os.path.join(out_dir, os.path.basename(image_path))
        det_model, boxes_filt, pred_phrases = run_detecter(det_model, image_path, args)

        pred_result_dict = {
            "boxes": boxes_filt,
            "labels": pred_phrases,
        }

        image = cv2.imread(image_path)

        if not only_det:

            if cpu_off_load:
                sam_model.mode = sam_model.model.to(args.sam_device)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            sam_model.set_image(image)

            transformed_boxes = sam_model.transform.apply_boxes_torch(boxes_filt, image.shape[:2])
            transformed_boxes = transformed_boxes.to(sam_model.model.device)

            masks, _, _ = sam_model.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False
            )
            pred_result_dict['masks'] = masks

            if cpu_off_load:
                sam_model.model = sam_model.model.to('cpu')

        draw_and_save(image, pred_result_dict, save_path)
        progress_bar.update()


if __name__ == '__main__':
    main()
