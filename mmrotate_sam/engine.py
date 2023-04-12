import os
import torch
from pathlib import Path
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import cv2

from mmrotate.structures import RotatedBoxes
from mmdet.models.utils import samplelist_boxtype2tensor
from mmengine.runner import load_checkpoint
from utils import show_box, show_mask
import matplotlib.pyplot as plt
from mmengine.structures import InstanceData
from data import build_visualizer


VIS_SCORE_THR = 0.3


@torch.no_grad()
def single_sample_step(data, det_model, sam_predictor, evaluator, args):
    device = det_model.model.device
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
                batch_boxes = h_bboxes[left_index: right_index]

            transformed_boxes = sam_predictor.transform.apply_boxes_torch(batch_boxes, img.shape[:2])
            batch_masks, qualities, lr_logits = sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False)
            batch_masks = batch_masks.squeeze(1).cpu()
            masks.extend([*batch_masks])
        masks = torch.stack(masks, dim=0)
        r_bboxes = [mask2rbox(mask.numpy()) for mask in masks]

    results_list = get_instancedata_resultlist(r_bboxes, labels, masks, scores, args.result_with_mask)
    data_samples = add_pred_to_datasample(results_list, data_samples)

    evaluator.process(data_samples=data_samples, data_batch=data)
    return evaluator


def mask2rbox(mask):
    y, x = np.nonzero(mask)
    points = np.stack([x, y], axis=-1)
    (cx, cy), (w, h), a = cv2.minAreaRect(points)
    r_bbox = np.array([cx, cy, w, h, a / 180 * np.pi])
    return r_bbox


def get_instancedata_resultlist(r_bboxes, labels, masks, scores, result_with_mask=False):
    results = InstanceData()
    results.bboxes = RotatedBoxes(r_bboxes)
    # results.scores = qualities
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
