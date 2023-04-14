# Copyright (c) OpenMMLab. All rights reserved.
# Refer from https://github.com/IDEA-Research/Grounded-Segment-Anything
import argparse
import os

import cv2
import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from mmengine import Registry
from mmengine.config import Config
from mmengine.utils import ProgressBar
from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples
from PIL import Image

# Grounding DINO
try:
    import groundingdino
    import groundingdino.datasets.transforms as T
    from groundingdino.models import build_model
    from groundingdino.util import get_tokenlizer
    from groundingdino.util.utils import (clean_state_dict,
                                          get_phrases_from_posmap)
    grounding_dino_transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
except ImportError:
    groundingdino = None

# GLIP
try:
    import maskrcnn_benchmark
    from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo
except ImportError:
    maskrcnn_benchmark = None

# mmdet
try:
    import mmdet
    from mmdet.apis import inference_detector, init_detector
except ImportError:
    mmdet = None

import sys

sys.path.append('../')
from mmpose_open_detection.utils import apply_exif_orientation  # noqa
from mmpose_open_detection.utils import get_file_list  # noqa


def parse_args():
    parser = argparse.ArgumentParser('mmpose grounding Demo', add_help=True)
    parser.add_argument('image', type=str, help='path to image file')
    parser.add_argument('det_config', type=str, help='path to det config file')
    parser.add_argument('det_weight', type=str, help='path to det weight file')
    parser.add_argument('pose_config', help='path to pose config file')
    parser.add_argument('pose_weight', help='path to pose weight file')
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
    parser.add_argument(
        '--apply-original-groudingdino',
        action='store_true',
        help='use original groudingdino label predict')

    # pose visualization param
    parser.add_argument(
        '--kpt-thr',
        type=float,
        default=0.3,
        help='Visualizing keypoint thresholds')
    parser.add_argument(
        '--skeleton-style',
        default='mmpose',
        type=str,
        choices=['mmpose', 'openpose'],
        help='Skeleton style selection')
    parser.add_argument(
        '--radius',
        type=int,
        default=3,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')
    parser.add_argument(
        '--alpha', type=float, default=0.8, help='The transparency of bboxes')
    parser.add_argument(
        '--draw-bbox', action='store_true', help='Draw bboxes of instances')

    return parser.parse_args()


def __reset_cls_layer_weight(model, weight):
    if isinstance(weight, str):
        print(f'Resetting cls_layer_weight from file: {weight}')
        zs_weight = torch.tensor(
            np.load(weight),
            dtype=torch.float32).permute(1, 0).contiguous()  # D x C
    else:
        zs_weight = weight
    zs_weight = torch.cat(
        [zs_weight, zs_weight.new_zeros(
            (zs_weight.shape[0], 1))], dim=1)  # D x (C + 1)
    zs_weight = F.normalize(zs_weight, p=2, dim=0)
    zs_weight = zs_weight.to(next(model.parameters()).device)
    num_classes = zs_weight.shape[-1]

    for bbox_head in model.roi_head.bbox_head:
        bbox_head.num_classes = num_classes
        del bbox_head.fc_cls.zs_weight
        bbox_head.fc_cls.zs_weight = zs_weight


def __build_grounding_dino_model(args):
    gdino_args = Config.fromfile(args.det_config)
    model = build_model(gdino_args)
    checkpoint = torch.load(args.det_weight, map_location='cpu')
    model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    model.eval()
    return model


def __build_glip_model(args):
    try:
        from maskrcnn_benchmark.config import cfg
    except ImportError:
        assert False, "'maskrcnn_benchmark' does not exist"
    cfg.merge_from_file(args.det_config)
    cfg.merge_from_list(['MODEL.WEIGHT', args.det_weight])
    cfg.merge_from_list(['MODEL.DEVICE', 'cpu'])
    model = GLIPDemo(
        cfg,
        min_image_size=800,
        confidence_threshold=args.box_thr,
        show_mask_heatmaps=False)
    return model


def build_detecter(args):
    if 'GroundingDINO' in args.det_config:
        detecter = __build_grounding_dino_model(args)
    elif 'glip' in args.det_config:
        detecter = __build_glip_model(args)
    else:
        config = Config.fromfile(args.det_config)
        if 'init_cfg' in config.model.backbone:
            config.model.backbone.init_cfg = None
        if 'Detic' in args.det_config:
            config.model.roi_head.mask_head = None
        detecter = init_detector(
            config, args.det_weight, device='cpu', cfg_options={})
    return detecter


def create_positive_dict(tokenized, tokens_positive, labels):
    """construct a dictionary such that positive_map[i] = j,
    if token i is mapped to j label"""

    positive_map_label_to_token = {}

    for j, tok_list in enumerate(tokens_positive):
        for (beg, end) in tok_list:
            beg_pos = tokenized.char_to_token(beg)
            end_pos = tokenized.char_to_token(end - 1)

            assert beg_pos is not None and end_pos is not None
            positive_map_label_to_token[labels[j]] = []
            for i in range(beg_pos, end_pos + 1):
                positive_map_label_to_token[labels[j]].append(i)

    return positive_map_label_to_token


def convert_grounding_to_od_logits(logits,
                                   num_classes,
                                   positive_map,
                                   score_agg='MEAN'):
    """
    logits: (num_query, max_seq_len)
    num_classes: 80 for COCO
    """
    assert logits.ndim == 2
    assert positive_map is not None
    scores = torch.zeros(logits.shape[0], num_classes).to(logits.device)
    # 256 -> 80, average for each class
    # score aggregation method
    if score_agg == 'MEAN':  # True
        for label_j in positive_map:
            scores[:, label_j] = logits[:,
                                        torch.LongTensor(positive_map[label_j]
                                                         )].mean(-1)
    else:
        raise NotImplementedError
    return scores


def run_detector(model, image_path, args):
    pred_dict = {}

    if 'GroundingDINO' in args.det_config:
        image_pil = Image.open(image_path).convert('RGB')  # load image
        image_pil = apply_exif_orientation(image_pil)
        image, _ = grounding_dino_transform(image_pil, None)  # 3, h, w

        text_prompt = args.text_prompt
        text_prompt = text_prompt.lower()
        text_prompt = text_prompt.strip()
        if not text_prompt.endswith('.'):
            text_prompt = text_prompt + '.'

        # Original GroundingDINO use text-thr to get class name,
        # the result will always result in categories that we don't want,
        # so we provide a category-restricted approach to address this

        if not args.apply_original_groudingdino:
            # custom label name
            custom_vocabulary = text_prompt[:-1].split('.')
            label_name = [c.strip() for c in custom_vocabulary]
            tokens_positive = []
            start_i = 0
            separation_tokens = ' . '
            for _index, label in enumerate(label_name):
                end_i = start_i + len(label)
                tokens_positive.append([(start_i, end_i)])
                if _index != len(label_name) - 1:
                    start_i = end_i + len(separation_tokens)
            tokenizer = get_tokenlizer.get_tokenlizer('bert-base-uncased')
            tokenized = tokenizer(
                args.text_prompt, padding='longest', return_tensors='pt')
            positive_map_label_to_token = create_positive_dict(
                tokenized, tokens_positive, list(range(len(label_name))))

        image = image.to(next(model.parameters()).device)

        with torch.no_grad():
            outputs = model(image[None], captions=[text_prompt])

        logits = outputs['pred_logits'].cpu().sigmoid()[0]  # (nq, 256)
        boxes = outputs['pred_boxes'].cpu()[0]  # (nq, 4)

        if not args.apply_original_groudingdino:
            logits = convert_grounding_to_od_logits(
                logits, len(label_name),
                positive_map_label_to_token)  # [N, num_classes]

        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > args.box_thr
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

        if args.apply_original_groudingdino:
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
        else:
            scores, pred_phrase_idxs = logits_filt.max(1)
            # build pred
            pred_labels = []
            pred_scores = []
            for score, pred_phrase_idx in zip(scores, pred_phrase_idxs):
                pred_labels.append(label_name[pred_phrase_idx])
                pred_scores.append(str(score.item())[:4])

        pred_dict['labels'] = pred_labels
        pred_dict['scores'] = pred_scores

        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]
        pred_dict['boxes'] = boxes_filt

    elif 'glip' in args.det_config:
        image = cv2.imread(image_path)
        text_prompt = args.text_prompt
        text_prompt = text_prompt.lower()
        text_prompt = text_prompt.strip()
        if not text_prompt.endswith('.'):
            text_prompt = text_prompt + '.'
        top_predictions = model.inference(image, text_prompt)
        scores = top_predictions.get_field('scores').tolist()
        labels = top_predictions.get_field('labels').tolist()
        new_labels = []
        if model.entities and model.plus:
            for i in labels:
                if i <= len(model.entities):
                    new_labels.append(model.entities[i - model.plus])
                else:
                    new_labels.append('object')
        else:
            new_labels = ['object' for i in labels]
        pred_dict['labels'] = new_labels
        pred_dict['scores'] = scores
        pred_dict['boxes'] = top_predictions.bbox
    else:
        with Registry('scope').switch_scope_and_registry('mmdet'):
            result = inference_detector(model, image_path)
        pred_instances = result.pred_instances[
            result.pred_instances.scores > args.box_thr]

        pred_dict['boxes'] = pred_instances.bboxes
        pred_dict['scores'] = pred_instances.scores.cpu().numpy().tolist()
        pred_dict['labels'] = [
            model.dataset_meta['classes'][label]
            for label in pred_instances.labels
        ]

    return model, pred_dict


def build_pose_estimator(args):
    pose_estimator = init_pose_estimator(
        args.pose_config,
        args.pose_weight,
        device=args.device,
        cfg_options=dict())
    return pose_estimator


def build_visualizer(pose_estimator, args):
    pose_estimator.cfg.visualizer.radius = args.radius
    pose_estimator.cfg.visualizer.alpha = args.alpha
    pose_estimator.cfg.visualizer.line_width = args.thickness

    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
    visualizer.set_dataset_meta(
        pose_estimator.dataset_meta, skeleton_style=args.skeleton_style)
    return visualizer


def run_pose_estimator(pose_estimator, image_path, det_results):
    bboxes = det_results['boxes'].cpu().numpy()
    pose_results = inference_topdown(pose_estimator, image_path, bboxes)
    data_samples = merge_data_samples(pose_results)
    return data_samples


def main():
    args = parse_args()

    if 'GroundingDINO' in args.det_config or 'glip' in args.det_config \
            or 'Detic' in args.det_config:
        assert args.text_prompt

    out_dir = args.out_dir

    det_model = build_detecter(args)
    if det_model is None:
        raise RuntimeError('no det model!')

    det_model = det_model.to(args.device)

    if 'Detic' in args.det_config:
        from projects.Detic.detic.utils import get_text_embeddings
        text_prompt = args.text_prompt
        text_prompt = text_prompt.lower()
        text_prompt = text_prompt.strip()
        if text_prompt.endswith('.'):
            text_prompt = text_prompt[:-1]
        custom_vocabulary = text_prompt.split('.')
        det_model.dataset_meta['classes'] = [
            c.strip() for c in custom_vocabulary
        ]
        embedding = get_text_embeddings(custom_vocabulary=custom_vocabulary)
        __reset_cls_layer_weight(det_model, embedding)

    pose_model = build_pose_estimator(args)
    visualizer = build_visualizer(pose_model, args)

    os.makedirs(out_dir, exist_ok=True)

    files, source_type = get_file_list(args.image)
    progress_bar = ProgressBar(len(files))
    for image_path in files:
        det_model, pred_dict = run_detector(det_model, image_path, args)

        if pred_dict['boxes'].shape[0] == 0:
            print('No objects detected !')
            continue

        data_samples = run_pose_estimator(pose_model, image_path, pred_dict)

        image = mmcv.imread(image_path, channel_order='rgb')
        save_path = os.path.join(args.out_dir, os.path.basename(image_path))
        visualizer.add_datasample(
            'result',
            image,
            data_sample=data_samples,
            draw_gt=False,
            draw_heatmap=False,
            draw_bbox=args.draw_bbox,
            show_kpt_idx=False,
            skeleton_style=args.skeleton_style,
            show=False,
            wait_time=0,
            out_file=save_path,
            kpt_thr=args.kpt_thr)

        progress_bar.update()


if __name__ == '__main__':
    main()
