import argparse
import os

import cv2
import matplotlib.pyplot as plt
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import openai
import re

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

from mmengine.config import Config
from PIL import Image

sys.path.append('../')
from mmdet_sam.utils import apply_exif_orientation, get_file_list  # noqa

system_prompt = """You must strictly answer the question step by step:

Step-1. based on the description and requirement provided by the user, find all objects related to the input from the description, and concisely explain why these objects meet the requirement.
Step-2. list out all related objects strictly as follows: <Therefore the answer is: [object_names]>.

If you did not complete all 2 steps as detailed as possible, you will be killed. You must finish the answer with complete sentences.

description: 
requirement: 
"""


def parse_args():
    parser = argparse.ArgumentParser(
        'Det GPT', add_help=True)
    parser.add_argument('image', type=str, help='path to image file')
    parser.add_argument('det_config', type=str, help='path to det config file')
    parser.add_argument('det_weight', type=str, help='path to det weight file')
    parser.add_argument('text_prompt', type=str, help='text prompt')
    parser.add_argument('--not-show-label', action='store_true')
    parser.add_argument(
        '--out-dir',
        '-o',
        type=str,
        default='outputs',
        help='output directory')
    parser.add_argument(
        '--box-thr', '-b', type=float, default=0.3, help='box threshold')
    parser.add_argument(
        '-det-device',
        '-d',
        default='cuda:0',
        help='Device used for inference')
    parser.add_argument(
        '-blip-device',
        '-p',
        default='cpu',
        help='Device used for inference')
    parser.add_argument(
        '--text-thr', type=float, default=0.25, help='text threshold')
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


def build_detecter(args):
    if 'GroundingDINO' in args.det_config:
        detecter = __build_grounding_dino_model(args)
    elif 'glip' in args.det_config:
        detecter = __build_glip_model(args)
    else:
        raise NotImplementedError()
    return detecter.to(args.det_device)


def build_blip(args):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    return model.to(args.blip_device), processor


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


def run(det_model, blip_model, blip_processor, args):
    raw_image = Image.open(args.image).convert('RGB')
    raw_image = apply_exif_orientation(raw_image)
    inputs = blip_processor(raw_image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    description = blip_processor.decode(out[0], skip_special_tokens=True)

    print(description)
    if len(str(description).strip()) == 0:
        print('exit!')
        return None

    content = system_prompt.replace('description: ', f'description: {description}')
    content = content.replace('requirement: ', f'requirement: {args.text_prompt}')

    prompt = [
        {
            'role': 'system',
            'content': content,
        }
    ]

    print(prompt)
    response = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=prompt, temperature=0.5, max_tokens=1000)
    text_prompt = response['choices'][0]['message']['content']
    print('pre match:', text_prompt)

    matches = re.findall(r'\[([^]]*)\]', text_prompt)

    if len(matches) == 0:
        print('exit!!')
        return None

    text_prompt = matches[0]
    print('post match:', text_prompt)

    pred_dict = {}
    if 'GroundingDINO' in args.det_config:
        image_pil = apply_exif_orientation(raw_image)
        image, _ = grounding_dino_transform(image_pil, None)  # 3, h, w

        text_prompt = text_prompt.lower()
        text_prompt = text_prompt.strip()
        text_prompt = text_prompt.replace(',', '.')

        if not text_prompt.endswith('. '):
            text_prompt = text_prompt + ' . '

        custom_vocabulary = text_prompt.split('.')
        label_name = [c.strip() for c in custom_vocabulary]
        label_name = list(filter(lambda x: len(x) > 0, label_name))

        tokens_positive = []
        separation_tokens = ' . '
        caption_string = ""

        for word in label_name:
            tokens_positive.append([[len(caption_string), len(caption_string) + len(word)]])
            caption_string += word
            caption_string += separation_tokens

        text_prompt = caption_string
        print('text_prompt:', text_prompt, 'label_name', label_name)

        tokenizer = get_tokenlizer.get_tokenlizer('bert-base-uncased')
        tokenized = tokenizer(
            text_prompt, padding='longest', return_tensors='pt')
        positive_map_label_to_token = create_positive_dict(
            tokenized, tokens_positive, list(range(len(label_name))))

        image = image.to(args.det_device)

        with torch.no_grad():
            outputs = det_model(image[None], captions=[text_prompt])

        logits = outputs['pred_logits'].cpu().sigmoid()[0]  # (nq, 256)
        boxes = outputs['pred_boxes'].cpu()[0]  # (nq, 4)

        logits = convert_grounding_to_od_logits(
            logits, len(label_name),
            positive_map_label_to_token)  # [N, num_classes]

        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > args.box_thr
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
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
        image = cv2.imread(args.image)
        text_prompt = args.text_prompt
        text_prompt = text_prompt.lower()
        text_prompt = text_prompt.strip()
        if not text_prompt.endswith('.'):
            text_prompt = text_prompt + '.'
        top_predictions = det_model.inference(image, text_prompt)
        scores = top_predictions.get_field('scores').tolist()
        labels = top_predictions.get_field('labels').tolist()
        new_labels = []
        if det_model.entities and det_model.plus:
            for i in labels:
                if i <= len(det_model.entities):
                    new_labels.append(det_model.entities[i - det_model.plus])
                else:
                    new_labels.append('object')
        else:
            new_labels = ['object' for i in labels]
        pred_dict['labels'] = new_labels
        pred_dict['scores'] = scores
        pred_dict['boxes'] = top_predictions.bbox

    return pred_dict


def draw_and_save(image_path,
                  pred_dict,
                  save_path,
                  show_label=True):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 10))
    plt.imshow(image)

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
                    x0, y0, f'{label}|{round(score, 2)}', color='white')

    plt.axis('off')
    plt.savefig(save_path)


def main():
    if groundingdino is None and maskrcnn_benchmark is None:
        raise RuntimeError('detection model is not installed,\
                 please install it follow README')

    args = parse_args()
    det_model = build_detecter(args)
    blip_model, blip_processor = build_blip(args)
    pred_dict = run(det_model, blip_model, blip_processor, args)

    os.makedirs(args.out_dir, exist_ok=True)
    save_path = os.path.join(args.out_dir, os.path.basename(args.image))
    draw_and_save(
        args.image, pred_dict, save_path, show_label=not args.not_show_label)


if __name__ == '__main__':
    main()

# processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
# model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
#
# # img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
# # raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
# img_url = '/home/PJLAB/huanghaian/yolo/DetGPT/examples/big_kitchen.jpg'
# raw_image = Image.open(img_url).convert('RGB')
# # conditional image captioning
# # text = "a photography of"
# # inputs = processor(raw_image, text, return_tensors="pt")
# #
# # out = model.generate(**inputs)
# # print(processor.decode(out[0], skip_special_tokens=True))
#
# # unconditional image captioning
# inputs = processor(raw_image, return_tensors="pt")
#
# out = model.generate(**inputs)
#
# description = processor.decode(out[0], skip_special_tokens=True)
# print(description)
#
# use_input = 'I want to have a cold beverage'
#
# system_prompt = """You must strictly answer the question step by step:
#
# Step-1. based on the description and requirement provided by the user, find all objects related to the input from the description, and concisely explain why these objects meet the requirement.
# Step-2. list out all related objects strictly as follows: <Therefore the answer is: [object_names]>.
#
# If you did not complete all 2 steps as detailed as possible, you will be killed. You must finish the answer with complete sentences.
#
# description:
# requirement:
# """
#
# content = system_prompt.replace('description: ', f'description: {description}')
# content = content.replace('requirement: ', f'requirement: {use_input}')
#
# prompt = [
#     {
#         'role': 'system',
#         'content': content,
#     }
# ]
#
# print(prompt)
# # response = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=prompt, temperature=0.5, max_tokens=1000)
# # reply = response['choices'][0]['message']['content']
# # print(reply)
#
# reply = 'stainless steel refrigerator'
#
# from groundingdino.util.inference import load_model, load_image, predict, annotate, Model
# import cv2
#
# model = load_model("GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "./groundingdino_swint_ogc.pth")
# IMAGE_PATH = "assets/demo1.jpg"
# TEXT_PROMPT = "bear."
# BOX_TRESHOLD = 0.35
# TEXT_TRESHOLD = 0.25
#
# image_source, image = load_image(IMAGE_PATH)
#
# boxes, logits, phrases = predict(
#     model=model,
#     image=image,
#     caption=TEXT_PROMPT,
#     box_threshold=BOX_TRESHOLD,
#     text_threshold=TEXT_TRESHOLD
# )
#
# annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
# cv2.imwrite("annotated_image.jpg", annotated_frame)
