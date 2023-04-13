# Copyright (c) OpenMMLab. All rights reserved.
# Refer from https://github.com/Li-Qingyun/sam-mmrotate
import argparse
import os
from copy import deepcopy

import cv2
import numpy as np
import torch
from data_builder import build_data_loader, build_evaluator
from mmdet.apis import init_detector
from mmdet.models.utils import samplelist_boxtype2tensor
from mmengine import Config
from mmengine.dist import get_dist_info, get_rank, init_dist, is_distributed
from mmengine.structures import InstanceData
from mmengine.utils import ProgressBar
from mmrotate.structures import RotatedBoxes
from mmrotate.utils import register_all_modules
from segment_anything import SamPredictor, sam_model_registry


def parse_args():
    parser = argparse.ArgumentParser(
        'Evaluation for Zero-shot Oriented Detector with Segment-Anything-'
        'Model Prompt by Predicted HBox of Horizontal Detector',
        add_help=True)
    # TODO: get data cfg from a file, instead of hard-code
    # parser.add_argument(
    #     'data_config', type=str,
    #     help='path to config file contains `data`
    #           cfg of the oriented object '
    #          'detection data set for evaluation')
    parser.add_argument(
        'det_config',
        type=str,
        help='path to config file contains `model` cfg of the detector')
    parser.add_argument(
        'det_weight', type=str, help='path to detector weight file')
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
        '--device', '-d', default='cuda', help='device used for inference')
    parser.add_argument('--format-only', action='store_true')
    parser.add_argument('--merge-patches', action='store_true')
    parser.add_argument('--set-min-box', action='store_true')
    parser.add_argument('--result-with-mask', action='store_true')
    parser.add_argument(
        '--max-batch-num-pred',
        type=int,
        default=100,
        help='max prediction number of mask generation (avoid OOM)')
    parser.add_argument('--only-det', action='store_true')
    parser.add_argument(
        '--out-dir',
        '-o',
        type=str,
        default='outputs',
        help='output directory')
    parser.add_argument(
        '--box-thr', '-b', type=float, default=0.3, help='box threshold')
    parser.add_argument('--num-worker', '-n', type=int, default=2)

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


@torch.no_grad()
def single_sample_step(data, det_model, sam_predictor, evaluator, args):
    device = sam_predictor.model.device
    copied_data = deepcopy(data)  # for sam

    # Stage 1
    for item in data.values():
        item[0].to(device)
    pred_results = det_model.test_step(data)
    pred_r_bboxes = pred_results[0].pred_instances.bboxes
    pred_r_bboxes = RotatedBoxes(pred_r_bboxes)
    h_bboxes = pred_r_bboxes.convert_to('hbox').tensor
    labels = pred_results[0].pred_instances.labels
    scores = pred_results[0].pred_instances.scores

    # Stage 2
    if len(h_bboxes) == 0:
        masks = h_bboxes.new_tensor((0, *data['inputs'][0].shape[:2]))
        data_samples = data['data_samples']
        r_bboxes = []
    else:
        img = copied_data['inputs'][0].permute(1, 2, 0).numpy()[:, :, ::-1]
        data_samples = copied_data['data_samples']
        data_sample = data_samples[0]
        data_sample.to(device=device)

        sam_predictor.set_image(img)

        # Too many predictions may result in OOM, hence,
        # we process the predictions in multiple batches.
        masks = []
        N = args.max_batch_num_pred
        num_pred = len(h_bboxes)
        num_batches = int(np.ceil(num_pred / N))
        for i in range(num_batches):
            left_index = i * N
            right_index = (i + 1) * N
            if i == num_batches - 1:
                batch_boxes = h_bboxes[left_index:]
            else:
                batch_boxes = h_bboxes[left_index:right_index]

            transformed_boxes = sam_predictor.transform.apply_boxes_torch(
                batch_boxes, img.shape[:2])
            batch_masks, qualities, lr_logits = sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False)
            batch_masks = batch_masks.squeeze(1).cpu()
            masks.extend([*batch_masks])
        masks = torch.stack(masks, dim=0)
        r_bboxes = [mask2rbox(mask.numpy()) for mask in masks]

    results_list = get_instancedata_resultlist(r_bboxes, labels, masks, scores,
                                               args.result_with_mask)
    data_samples = add_pred_to_datasample(results_list, data_samples)

    evaluator.process(data_samples=data_samples, data_batch=data)
    return evaluator


def mask2rbox(mask):
    y, x = np.nonzero(mask)
    points = np.stack([x, y], axis=-1)
    (cx, cy), (w, h), a = cv2.minAreaRect(points)
    r_bbox = np.array([cx, cy, w, h, a / 180 * np.pi])
    return r_bbox


def get_instancedata_resultlist(r_bboxes,
                                labels,
                                masks,
                                scores,
                                result_with_mask=False):
    results = InstanceData()
    results.bboxes = RotatedBoxes(r_bboxes)
    results.scores = scores
    results.labels = labels
    if result_with_mask:
        results.masks = masks.cpu().numpy()
    results_list = [results]
    return results_list


def add_pred_to_datasample(results_list, data_samples):
    for data_sample, pred_instances in zip(data_samples, results_list):
        data_sample.pred_instances = pred_instances
    samplelist_boxtype2tensor(data_samples)
    return data_samples


if __name__ == '__main__':
    args = parse_args()
    if args.launcher == 'none':
        _distributed = False
    else:
        _distributed = True
    if _distributed and not is_distributed():
        init_dist(args.launcher)

    register_all_modules(init_default_scope=True)

    # prepare the detector model
    detector_cfg = Config.fromfile(args.det_config)
    if 'init_cfg' in detector_cfg.model.backbone:
        detector_cfg.model.backbone.init_cfg = None
    if args.set_min_box:
        detector_cfg.model.test_cfg['min_bbox_size'] = 10
    det_model = init_detector(
        detector_cfg, args.det_weight, device='cpu', cfg_options={})
    det_model = det_model.to(args.device)

    # prepare the sam model
    build_sam = sam_model_registry[args.sam_type]
    sam_model = SamPredictor(build_sam(checkpoint=args.sam_weight))
    sam_model.model = sam_model.model.to(args.device)

    # prepare dataset
    # TODO: get data cfg from a file, instead of hard-code
    # data_cfg = Config.fromfile(args.data_config)
    dataloader = build_data_loader('test_without_hbox')
    evaluator = build_evaluator(args.merge_patches, args.format_only)
    # TODO: add assert to make sure the CLASSES in ckpt and
    #  in dataset are the same
    evaluator.dataset_meta = dataloader.dataset.metainfo

    if get_rank() == 0:
        print('data len: ', len(dataloader.dataset), 'num_word_size: ',
              get_dist_info()[1])

        progress_bar = ProgressBar(len(dataloader))

    det_model.eval()
    for i, data in enumerate(dataloader):

        evaluator = single_sample_step(data, det_model, sam_model, evaluator,
                                       args)
        if get_rank() == 0:
            progress_bar.update()

    metrics = evaluator.evaluate(len(dataloader.dataset))
