import torch
from tqdm import tqdm
import numpy as np
import cv2
from mmrotate.utils import register_all_modules
from data import build_data_loader, build_evaluator, build_visualizer
from utils import show_box, show_mask
import matplotlib.pyplot as plt
from mmengine.structures import InstanceData
from segment_anything import sam_model_registry, SamPredictor
from mmrotate.structures import RotatedBoxes
from mmengine import ProgressBar
from mmdet.models.utils import samplelist_boxtype2tensor


register_all_modules(init_default_scope=True)

SHOW = False
FORMAT_ONLY = False
MERGE_PATCHES = False


if __name__ == '__main__':


    dataloader = build_data_loader('trainval_with_hbox')
    evaluator = build_evaluator(MERGE_PATCHES, FORMAT_ONLY)
    evaluator.dataset_meta = dataloader.dataset.metainfo

    sam_checkpoint = r"../segment-anything/checkpoints/sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    device = "cpu"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)

    sam = sam.to(device=device)

    predictor = SamPredictor(sam)

    for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):

        img = data['inputs'][0].permute(1, 2, 0).numpy()[:, :, ::-1]
        data_samples = data['data_samples']
        data_sample = data_samples[0]
        data_sample = data_sample.to(device=device)

        h_bboxes = data_sample.h_gt_bboxes.tensor.to(device=device)
        labels = data_sample.gt_instances.labels.to(device=device)

        r_bboxes = []
        if len(h_bboxes) == 0:
            qualities = h_bboxes[:, 0]
            masks = h_bboxes.new_tensor((0, *img.shape[:2]))
        else:
            predictor.set_image(img)
            transformed_boxes = predictor.transform.apply_boxes_torch(h_bboxes, img.shape[:2])
            masks, qualities, lr_logits = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False)
            masks = masks.squeeze(1)
            qualities = qualities.squeeze(-1)
            for mask in masks:
                y, x = np.nonzero(mask.cpu().numpy())
                points = np.stack([x, y], axis=-1)
                (cx, cy), (w, h), a = cv2.minAreaRect(points)
                r_bboxes.append(np.array([cx, cy, w, h, a/180*np.pi]))

        results = InstanceData()
        results.bboxes = RotatedBoxes(r_bboxes)
        results.scores = qualities
        results.labels = labels
        results.masks = masks.cpu().numpy()
        results_list = [results]

        # add_pred_to_datasample
        for data_sample, pred_instances in zip(data_samples, results_list):
            data_sample.pred_instances = pred_instances
        samplelist_boxtype2tensor(data_samples)

        evaluator.process(data_samples=data_samples, data_batch=data)

        if SHOW:
            plt.figure(figsize=(10, 10))
            plt.imshow(img)
            for mask in masks:
                show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
            for box in h_bboxes:
                show_box(box.cpu().numpy(), plt.gca())
            plt.axis('off')
            # plt.show()
            plt.savefig(f'./out_mask_{i}.png')

            # draw rbox with mmrotate
            visualizer = build_visualizer()
            visualizer.dataset_meta = dataloader.dataset.metainfo
            out_img = visualizer._draw_instances(
                img, results,
                dataloader.dataset.metainfo['classes'],
                dataloader.dataset.metainfo['palette'])
            # visualizer.show()
            cv2.imwrite(f'./out_rbox_{i}.png', out_img[:, :, ::-1])

    metrics = evaluator.evaluate(len(dataloader.dataset))
