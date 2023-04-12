# Copyright (c) OpenMMLab. All rights reserved.
# Refer from https://github.com/IDEA-Research/Grounded-Segment-Anything
import argparse
import json
import os
import warnings
from functools import partial

import cv2
# Grounding DINO
import groundingdino.datasets.transforms as T
import numpy as np
import pycocotools.mask as mask_util
import torch
from groundingdino.models import build_model
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from mmdet.apis import inference_detector, init_detector
from mmengine.config import Config
from mmengine.dataset import DefaultSampler, worker_init_fn
from mmengine.dist import (collect_results, get_dist_info, get_rank, init_dist,
                           is_distributed)
from mmengine.utils import ProgressBar
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
# segment anything
from segment_anything import SamPredictor, sam_model_registry
from torch.utils.data import DataLoader, Dataset


def parse_args():
    parser = argparse.ArgumentParser(
        'Detect-Segment-Anything Demo', add_help=True)
    parser.add_argument('data_root', type=str)
    parser.add_argument('det_config', type=str, help='path to det config file')
    parser.add_argument('det_weight', type=str, help='path to det weight file')
    parser.add_argument(
        '--ann-file', type=str, default='annotations/instances_val2017.json')
    parser.add_argument('--data-prefix', type=str, default='val2017/')
    parser.add_argument('--only-det', action='store_true')
    parser.add_argument(
        '--sam-type',
        type=str,
        default='vit_h',
        choices=['vit_h', 'vit_l', 'vit_b'],
        help='sam type')
    parser.add_argument(
        '--sam-weight',
        type=str,
        default='../models/sam_vit_h_4b8939.pth',
        help='path to checkpoint file')
    parser.add_argument(
        '--out-dir',
        '-o',
        type=str,
        default='outputs',
        help='output directory')
    parser.add_argument(
        '--box-thr', '-b', type=float, default=0.3, help='box threshold')
    parser.add_argument(
        '--det-device', '-d', default='cuda', help='Device used for inference')
    parser.add_argument(
        '--sam-device', '-s', default='cuda', help='Device used for inference')
    parser.add_argument('--cpu-off-load', '-c', action='store_true')
    parser.add_argument('--num-worker', '-n', type=int, default=2)

    # GroundingDINO param
    parser.add_argument('--text-prompt', '-t', type=str, help='cls path')
    parser.add_argument(
        '--text-thr', type=float, default=0.25, help='text threshold')

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
    checkpoint = torch.load(args.det_weight, map_location='cpu')
    model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    model.eval()
    return model


grounding_dino_transform = T.Compose([
    T.RandomResize([800], max_size=1333),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def build_detector(args):
    if 'GroundingDINO' in args.det_config:
        detector = __build_grounding_dino_model(args)
    else:
        config = Config.fromfile(args.det_config)
        if 'init_cfg' in config.model.backbone:
            config.model.backbone.init_cfg = None
        detector = init_detector(
            config, args.det_weight, device='cpu', cfg_options={})
    return detector


def run_detector(model, image_path, args):
    pred_dict = {}

    if args.cpu_off_load:
        model = model.to(args.det_device)

    if 'GroundingDINO' in args.det_config:
        image_pil = Image.open(image_path).convert('RGB')  # load image
        image, _ = grounding_dino_transform(image_pil, None)  # 3, h, w

        text_prompt = args.text_prompt
        with open(text_prompt) as f:
            coco_cls_str = f.read()
        text_prompt = coco_cls_str.replace('\n', ' . ')

        if get_rank() == 0:
            warnings.warn(f'text prompt is {text_prompt}')

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
        result = inference_detector(model, image_path)
        pred_instances = result.pred_instances[
            result.pred_instances.scores > args.box_thr]

        pred_dict['boxes'] = pred_instances.bboxes
        pred_dict['scores'] = pred_instances.scores.cpu().numpy().tolist()
        pred_dict['labels'] = [
            model.dataset_meta['classes'][label]
            for label in pred_instances.labels
        ]

    if args.cpu_off_load:
        model = model.to('cpu')
    return model, pred_dict


def fake_collate(x):
    return x


def main():
    args = parse_args()
    if args.cpu_off_load is True:
        if 'cpu' in args.det_device and 'cpu ' in args.sam_device:
            raise RuntimeError(
                'args.cpu_off_load is an invalid parameter due to '
                'detection and sam model are on the cpu.')

    only_det = args.only_det
    cpu_off_load = args.cpu_off_load

    if 'GroundingDINO' in args.det_config:
        assert args.text_prompt

    if args.launcher == 'none':
        _distributed = False
    else:
        _distributed = True

    if _distributed and not is_distributed():
        init_dist(args.launcher)

    det_model = build_detector(args)
    if not cpu_off_load:
        det_model = det_model.to(args.det_device)

    if not only_det:
        build_sam = sam_model_registry[args.sam_type]
        sam_model = SamPredictor(build_sam(checkpoint=args.sam_weight))
        if not cpu_off_load:
            sam_model.model = sam_model.model.to(args.sam_device)

    coco = COCO(os.path.join(args.data_root, args.ann_file))
    coco_dataset = SimpleDataset(coco.getImgIds())

    name2id = {}
    for categories in coco.dataset['categories']:
        name2id[categories['name']] = categories['id']

    if get_rank() == 0:
        print('data len: ', len(coco_dataset), 'num_word_size: ',
              get_dist_info()[1])

    sampler = DefaultSampler(coco_dataset, False)
    init_fn = partial(
        worker_init_fn,
        num_workers=args.num_worker,
        rank=get_rank(),
        seed=0,
        disable_subprocess_warning=True)
    data_loader = DataLoader(
        dataset=coco_dataset,
        sampler=sampler,
        collate_fn=fake_collate,
        worker_init_fn=init_fn,
        batch_size=1,
        num_workers=args.num_worker,
        persistent_workers=False if args.num_worker == 0 else True,
        drop_last=False)

    if get_rank() == 0:
        progress_bar = ProgressBar(len(data_loader))

    part_json_data = []

    for i, data in enumerate(data_loader):
        new_json_data = dict(annotation=[])
        image_id = data[0]
        raw_img_info = coco.loadImgs([image_id])[0]
        raw_img_info['img_id'] = image_id
        new_json_data['image'] = raw_img_info

        file_name = raw_img_info['file_name']
        image_path = os.path.join(args.data_root, args.data_prefix, file_name)

        det_model, pred_dict = run_detector(det_model, image_path, args)

        if pred_dict['boxes'].shape[0] == 0:
            part_json_data.append(new_json_data)
            continue

        image = cv2.imread(image_path)

        if not only_det:

            if cpu_off_load:
                sam_model.mode = sam_model.model.to(args.sam_device)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            sam_model.set_image(image)

            transformed_boxes = sam_model.transform.apply_boxes_torch(
                pred_dict['boxes'], image.shape[:2])
            transformed_boxes = transformed_boxes.to(sam_model.model.device)

            masks, _, _ = sam_model.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False)
            pred_dict['masks'] = masks.cpu().numpy()

            if cpu_off_load:
                sam_model.model = sam_model.model.to('cpu')

        pred_dict['boxes'] = pred_dict['boxes'].int().cpu().numpy().tolist()

        for i in range(len(pred_dict['boxes'])):
            label = pred_dict['labels'][i]
            score = pred_dict['scores'][i]
            bbox = pred_dict['boxes'][i]

            coco_bbox = [
                bbox[0],
                bbox[1],
                bbox[2] - bbox[0],
                bbox[3] - bbox[1],
            ]

            if label not in name2id:
                warnings.warn(f'not match predicted label of {label}')
                continue

            annotation = dict(
                image_id=image_id,
                bbox=coco_bbox,
                score=float(score),
                iscrowd=0,
                category_id=name2id[label],
                area=coco_bbox[2] * coco_bbox[3])

            if 'masks' in pred_dict:
                mask = pred_dict['masks'][i][0]
                encode_mask = mask_util.encode(
                    np.array(mask[:, :, np.newaxis], order='F',
                             dtype='uint8'))[0]
                encode_mask['counts'] = encode_mask['counts'].decode()
                annotation['segmentation'] = encode_mask
            else:
                annotation['segmentation'] = []
            new_json_data['annotation'].append(annotation)

        part_json_data.append(new_json_data)

        if get_rank() == 0:
            progress_bar.update()

    all_json_results = collect_results(part_json_data, len(coco_dataset),
                                       'cpu')

    if get_rank() == 0:
        new_json_data = {
            'info': coco.dataset['info'],
            'licenses': coco.dataset['licenses'],
            'categories': coco.dataset['categories'],
            'images':
            [json_results['image'] for json_results in all_json_results]
        }

        annotations = []
        annotation_id = 1
        for annotation in all_json_results:
            annotation = annotation['annotation']
            for ann in annotation:
                ann['id'] = annotation_id
                annotation_id += 1
                annotations.append(ann)

        if len(annotations) > 0:
            new_json_data['annotations'] = annotations

        output_json_name = args.ann_file[:-5] + '_pred.json'
        output_name = os.path.join(args.out_dir, output_json_name)
        os.makedirs(os.path.dirname(output_name), exist_ok=True)

        with open(output_name, 'w') as f:
            json.dump(new_json_data, f)

        if len(coco.dataset['annotations']) > 0:
            cocoDt = COCO(output_name)
            if only_det:
                metrics = ['bbox']
            else:
                metrics = ['bbox', 'segm']

            for metric in metrics:
                coco_eval = COCOeval(coco, cocoDt, iouType=metric)
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
        else:
            warnings.warn("No gt label, can't evaluate")


if __name__ == '__main__':
    main()
