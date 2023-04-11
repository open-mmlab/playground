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

from pycocotools.coco import COCO
import json
import pycocotools.mask as mask_util
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader, Dataset
from mmengine.dataset import DefaultSampler, default_collate, worker_init_fn
from functools import partial
from mmengine.dist import (broadcast, get_dist_info, get_rank, init_dist,
                           is_distributed, master_only, collect_results, barrier)
from mmengine.device import get_device
from torch.nn.parallel import DistributedDataParallel


def parse_args():
    parser = argparse.ArgumentParser("Detect-Segment-Anything Demo", add_help=True)
    parser.add_argument("data_root", type=str)
    parser.add_argument("det_config", type=str, help="path to det config file")
    parser.add_argument("det_weight", type=str, help="path to det weight file")
    parser.add_argument("--ann-file", type=str, default='annotations/instances_val2017.json')
    parser.add_argument("--data-prefix", type=str, default='val2017/')
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
    parser.add_argument("--text-prompt", '-t', type=str, help="text prompt or cls path")
    parser.add_argument("--text-thr", type=float, default=0.25, help="text threshold")

    # dist param
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


class SimpleDataset(Dataset):
    def __init__(self, img_ids):
        self.img_ids = img_ids

    def __getitem__(self, item):
        return self.img_ids[item]

    def __len__(self):
        return len(self.img_ids)


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
    pred_dict = {}

    if args.cpu_off_load:
        model = model.to(args.det_device)

    if 'GroundingDINO' in args.det_config:
        image_pil = Image.open(image_path).convert("RGB")  # load image
        image, _ = grounding_dino_transform(image_pil, None)  # 3, h, w

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
        pred_labels = []
        pred_scores = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > args.text_thr, tokenized, tokenlizer)
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

        if args.cpu_off_load:
            model = model.to('cpu')
        return model, pred_dict


def draw_and_save(image, pred_dict, save_path, random_color=True, show_label=True):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    with_mask = 'masks' in pred_dict
    labels = pred_dict['labels']
    scores = pred_dict['scores']

    bboxes = pred_dict['boxes'].cpu().numpy()
    for box, label, score in zip(bboxes, labels, scores):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        plt.gca().add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

        if show_label and not with_mask:
            pass
            # todo

    if with_mask:
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

    if args.launcher == 'none':
        _distributed = False
    else:
        _distributed = True
        assert not args.cpu_off_load
        assert 'cpu' in args.det_device and 'cpu' in args.sam_device

    if _distributed and not is_distributed():
        init_dist(args.launcher)

    det_model = build_detecter(args)
    if not cpu_off_load:
        det_model = det_model.to(args.det_device)

    if not only_det:
        sam_model = SamPredictor(build_sam(checkpoint=args.sam_weight))
        if not cpu_off_load:
            sam_model.mode = sam_model.model.to(args.sam_device)

    if _distributed:
        det_model = det_model.to(get_device())
        det_model = DistributedDataParallel(
            module=det_model,
            device_ids=[int(os.environ['LOCAL_RANK'])],
            broadcast_buffers=False,
            find_unused_parameters=False)
        sam_model.model = sam_model.model.to(get_device())
        sam_model.model = DistributedDataParallel(
            module=sam_model.model,
            device_ids=[int(os.environ['LOCAL_RANK'])],
            broadcast_buffers=False,
            find_unused_parameters=False)

    coco = COCO(os.path.join(args.data_root, args.ann_file))
    coco_dataset = SimpleDataset(coco.getImgIds())

    if get_rank() == 0:
        print('data len: ', len(coco_dataset), 'num_word_size: ', get_dist_info()[1])

    files, source_type = get_file_list(args.image)
    progress_bar = ProgressBar(len(files))
    for image_path in files:
        save_path = os.path.join(out_dir, os.path.basename(image_path))
        det_model, pred_dict = run_detecter(det_model, image_path, args)

        image = cv2.imread(image_path)

        if not only_det:

            if cpu_off_load:
                sam_model.mode = sam_model.model.to(args.sam_device)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            sam_model.set_image(image)

            transformed_boxes = sam_model.transform.apply_boxes_torch(pred_dict['boxes'], image.shape[:2])
            transformed_boxes = transformed_boxes.to(sam_model.model.device)

            masks, _, _ = sam_model.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False
            )
            pred_dict['masks'] = masks

            if cpu_off_load:
                sam_model.model = sam_model.model.to('cpu')

        draw_and_save(image, pred_dict, save_path)
        progress_bar.update()


if __name__ == '__main__':
    main()
