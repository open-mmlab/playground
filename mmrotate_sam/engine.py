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


RESULT_WITH_MASK = True
MAX_BATCH_NUM_PRED = 100
VIS_SCORE_THR = 0.3


@torch.no_grad()
def single_sample_step(img_id, data, model, predictor, evaluator, dataloader, device, SHOW):
    copied_data = deepcopy(data)  # for sam
    for item in data.values():
        item[0].to(device)

    # Stage 1
    # data['inputs'][0] = torch.flip(data['inputs'][0], dims=[0])
    with torch.no_grad():
        pred_results = model.test_step(data)
    pred_r_bboxes = pred_results[0].pred_instances.bboxes
    pred_r_bboxes = RotatedBoxes(pred_r_bboxes)
    h_bboxes = pred_r_bboxes.convert_to('hbox').tensor
    labels = pred_results[0].pred_instances.labels
    scores = pred_results[0].pred_instances.scores

    # Stage 2
    if len(h_bboxes) == 0:
        qualities = h_bboxes[:, 0]
        masks = h_bboxes.new_tensor((0, *data['inputs'][0].shape[:2]))
        data_samples = data['data_samples']
        r_bboxes = []
    else:
        img = copied_data['inputs'][0].permute(1, 2, 0).numpy()[:, :, ::-1]
        data_samples = copied_data['data_samples']
        data_sample = data_samples[0]
        data_sample = data_sample.to(device=device)

        predictor.set_image(img)

        # Too many predictions may result in OOM, hence,
        # we process the predictions in multiple batches.
        masks = []
        num_pred = len(h_bboxes)
        num_batches = int(np.ceil(num_pred / MAX_BATCH_NUM_PRED))
        for i in range(num_batches):
            left_index = i * MAX_BATCH_NUM_PRED
            right_index = (i + 1) * MAX_BATCH_NUM_PRED
            if i == num_batches - 1:
                batch_boxes = h_bboxes[left_index:]
            else:
                batch_boxes = h_bboxes[left_index: right_index]

            transformed_boxes = predictor.transform.apply_boxes_torch(batch_boxes, img.shape[:2])
            batch_masks, qualities, lr_logits = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False)
            batch_masks = batch_masks.squeeze(1).cpu()
            masks.extend([*batch_masks])
        masks = torch.stack(masks, dim=0)
        r_bboxes = [mask2rbox(mask.numpy()) for mask in masks]

    results_list = get_instancedata_resultlist(r_bboxes, labels, masks, scores)
    data_samples = add_pred_to_datasample(results_list, data_samples)

    evaluator.process(data_samples=data_samples, data_batch=data)

    if SHOW:
        if len(h_bboxes) != 0 and img_id < 100:
            img_name = data_samples[0].img_id
            show_results(img, masks, h_bboxes, results_list, img_id, img_name, dataloader)

    return evaluator


def mask2rbox(mask):
    y, x = np.nonzero(mask)
    points = np.stack([x, y], axis=-1)
    (cx, cy), (w, h), a = cv2.minAreaRect(points)
    r_bbox = np.array([cx, cy, w, h, a / 180 * np.pi])
    return r_bbox

def show_results(img, masks, h_bboxes, results_list, i, img_name, dataloader):
    output_dir = './output_vis/'
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    results = results_list[0]

    # vis first stage
    # plt.figure(figsize=(10, 10))
    # plt.imshow(img)
    # for mask in masks:
    #     show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    # for box in h_bboxes:
    #     show_box(box.cpu().numpy(), plt.gca())
    # plt.axis('off')
    # # plt.show()
    # plt.savefig(f'./out_mask_{i}.png')
    # plt.close()

    # draw rbox with mmrotate
    visualizer = build_visualizer()
    visualizer.dataset_meta = dataloader.dataset.metainfo

    scores = results.scores
    keep_results = results[scores >= VIS_SCORE_THR]
    out_img = visualizer._draw_instances(
        img, keep_results,
        dataloader.dataset.metainfo['classes'],
        dataloader.dataset.metainfo['palette'],
        box_alpha=0.9, mask_alpha=0.3)
    # visualizer.show()
    # cv2.imwrite(os.path.join(output_dir, f'out_rbox_{i}.png'), out_img[:, :, ::-1])
    cv2.imwrite(os.path.join(output_dir, f'rdet-sam_{img_name}.png'),
                out_img[:, :, ::-1])


def add_pred_to_datasample(results_list, data_samples):
    for data_sample, pred_instances in zip(data_samples, results_list):
        data_sample.pred_instances = pred_instances
    samplelist_boxtype2tensor(data_samples)
    return data_samples


def get_instancedata_resultlist(r_bboxes, labels, masks, scores):
    results = InstanceData()
    results.bboxes = RotatedBoxes(r_bboxes)
    # results.scores = qualities
    results.scores = scores
    results.labels = labels
    if RESULT_WITH_MASK:
        results.masks = masks.cpu().numpy()
    results_list = [results]
    return results_list
