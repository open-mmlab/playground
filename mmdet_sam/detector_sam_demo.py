# Copyright (c) OpenMMLab. All rights reserved.
# Refer from https://github.com/IDEA-Research/Grounded-Segment-Anything
import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

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

# mmdet
try:
    import mmdet
    from mmdet.apis import inference_detector, init_detector
except ImportError:
    mmdet = None

import sys

from mmengine.config import Config
from mmengine.utils import ProgressBar
from PIL import Image
# segment anything
from segment_anything import SamPredictor, sam_model_registry

sys.path.append('../')
from mmdet_sam.utils import apply_exif_orientation, get_file_list  # noqa

# GLIP
try:
    import maskrcnn_benchmark

    from mmdet_sam.predictor_glip import GLIPDemo
except ImportError:
    maskrcnn_benchmark = None


def parse_args():
    parser = argparse.ArgumentParser(
        'Detect-Segment-Anything Demo', add_help=True)
    parser.add_argument('image', type=str, help='path to image file')
    parser.add_argument('det_config', type=str, help='path to det config file')
    parser.add_argument('det_weight', type=str, help='path to det weight file')
    parser.add_argument('--only-det', action='store_true')
    parser.add_argument('--not-show-label', action='store_true')
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
        '--det-device',
        '-d',
        default='cuda:0',
        help='Device used for inference')
    parser.add_argument(
        '--sam-device',
        '-s',
        default='cuda:0',
        help='Device used for inference')
    parser.add_argument('--cpu-off-load', '-c', action='store_true')

    # Detic param
    parser.add_argument('--use-detic-mask', '-u', action='store_true')

    # GroundingDINO param
    parser.add_argument('--text-prompt', '-t', type=str, help='text prompt')
    parser.add_argument(
        '--text-thr', type=float, default=0.25, help='text threshold')
    parser.add_argument(
        '--apply-original-groudingdino',
        action='store_true',
        help='use original groudingdino label predict')

    # GLIP param
    parser.add_argument(
        '--apply-other-text',
        action='store_true',
        help='means use text prompt only conclude label name')
    return parser.parse_args()


def __build_grounding_dino_model(args):
    gdino_args = Config.fromfile(args.det_config)
    model = build_model(gdino_args)
    checkpoint = torch.load(args.det_weight, map_location='cpu')
    model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    model.eval()
    return model


def __build_glip_model(args):
    assert maskrcnn_benchmark is not None
    from maskrcnn_benchmark.config import cfg
    cfg.merge_from_file(args.det_config)
    cfg.merge_from_list(['MODEL.WEIGHT', args.det_weight])
    cfg.merge_from_list(['MODEL.DEVICE', 'cpu'])
    model = GLIPDemo(
        cfg,
        min_image_size=800,
        confidence_threshold=args.box_thr,
        show_mask_heatmaps=False)
    return model


def __reset_cls_layer_weight(model, weight):
    if type(weight) == str:
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


def build_detecter(args):
    if 'GroundingDINO' in args.det_config:
        detecter = __build_grounding_dino_model(args)
    elif 'glip' in args.det_config:
        detecter = __build_glip_model(args)
    else:
        config = Config.fromfile(args.det_config)
        if 'init_cfg' in config.model.backbone:
            config.model.backbone.init_cfg = None
        if 'detic' in args.det_config and not args.use_detic_mask:
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

    if args.cpu_off_load:
        if 'glip' in args.det_config:
            model.model = model.model.to(args.det_device)
            model.device = args.det_device
        else:
            model = model.to(args.det_device)

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
        # caption
        text_prompt = args.text_prompt
        text_prompt = text_prompt.lower()
        text_prompt = text_prompt.strip()
        if not text_prompt.endswith('.') and not args.apply_other_text:
            text_prompt = text_prompt + '.'

        custom_vocabulary = text_prompt[:-1].split('.')
        label_name = [c.strip() for c in custom_vocabulary]

        # top_predictions = model.inference(image, label_name)
        if args.apply_other_text:
            top_predictions = model.inference(
                image, args.text_prompt, use_other_text=True)
        else:
            top_predictions = model.inference(
                image, args.text_prompt, use_other_text=False)
        scores = top_predictions.get_field('scores').tolist()
        labels = top_predictions.get_field('labels').tolist()

        if args.apply_other_text:
            new_labels = []
            if model.entities and model.plus:
                for i in labels:
                    if i <= len(model.entities):
                        new_labels.append(model.entities[i - model.plus])
                    else:
                        new_labels.append('object')
            else:
                new_labels = ['object' for i in labels]
        else:
            new_labels = [label_name[i] for i in labels]

        pred_dict['labels'] = new_labels
        pred_dict['scores'] = scores
        pred_dict['boxes'] = top_predictions.bbox
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
        if args.use_detic_mask:
            pred_dict['masks'] = pred_instances.masks

    if args.cpu_off_load:
        if 'glip' in args.det_config:
            model.model = model.model.to('cpu')
            model.device = 'cpu'
        else:
            model = model.to('cpu')
    return model, pred_dict


def draw_and_save(image,
                  pred_dict,
                  save_path,
                  random_color=True,
                  show_label=True):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    with_mask = 'masks' in pred_dict
    labels = pred_dict['labels']
    scores = pred_dict['scores']

    bboxes = pred_dict['boxes'].cpu().numpy()
    for box, label, score in zip(bboxes, labels, scores):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        plt.gca().add_patch(
            plt.Rectangle((x0, y0),
                          w,
                          h,
                          edgecolor='green',
                          facecolor=(0, 0, 0, 0),
                          lw=2))

        if show_label:
            if isinstance(score, str):
                plt.gca().text(x0, y0, f'{label}|{score}', color='white')
            else:
                plt.gca().text(
                    x0, y0, f'{label}|{round(score,2)}', color='white')

    if with_mask:
        masks = pred_dict['masks'].cpu().numpy()
        for mask in masks:
            if random_color:
                color = np.concatenate(
                    [np.random.random(3), np.array([0.6])], axis=0)
            else:
                color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
            h, w = mask.shape[-2:]
            mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
            plt.gca().imshow(mask_image)

    plt.axis('off')
    plt.savefig(save_path)


def main():
    if groundingdino is None and maskrcnn_benchmark is None and mmdet is None:
        raise RuntimeError('detection model is not installed,\
                 please install it follow README')

    args = parse_args()
    if args.cpu_off_load is True:
        if 'cpu' in args.det_device and 'cpu ' in args.sam_device:
            raise RuntimeError(
                'args.cpu_off_load is an invalid parameter due to '
                'detection and sam model are on the cpu.')

    only_det = args.only_det
    cpu_off_load = args.cpu_off_load
    out_dir = args.out_dir

    if 'GroundingDINO' in args.det_config or 'glip' in args.det_config \
            or 'Detic' in args.det_config:
        assert args.text_prompt

    det_model = build_detecter(args)
    if not cpu_off_load:
        if 'glip' in args.det_config:
            det_model.model = det_model.model.to(args.det_device)
            det_model.device = args.det_device
        else:
            det_model = det_model.to(args.det_device)

    if args.use_detic_mask:
        only_det = True

    if not only_det:
        build_sam = sam_model_registry[args.sam_type]
        sam_model = SamPredictor(build_sam(checkpoint=args.sam_weight))
        if not cpu_off_load:
            sam_model.mode = sam_model.model.to(args.sam_device)

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

    os.makedirs(out_dir, exist_ok=True)

    files, source_type = get_file_list(args.image)
    progress_bar = ProgressBar(len(files))
    for image_path in files:
        save_path = os.path.join(out_dir, os.path.basename(image_path))
        det_model, pred_dict = run_detector(det_model, image_path, args)

        if pred_dict['boxes'].shape[0] == 0:
            print('No objects detected !')
            continue

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if not only_det:

            if cpu_off_load:
                sam_model.model = sam_model.model.to(args.sam_device)

            sam_model.set_image(image)

            transformed_boxes = sam_model.transform.apply_boxes_torch(
                pred_dict['boxes'], image.shape[:2])
            transformed_boxes = transformed_boxes.to(sam_model.model.device)

            masks, _, _ = sam_model.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False)
            pred_dict['masks'] = masks

            if cpu_off_load:
                sam_model.model = sam_model.model.to('cpu')

        draw_and_save(
            image, pred_dict, save_path, show_label=not args.not_show_label)
        progress_bar.update()


if __name__ == '__main__':
    main()
