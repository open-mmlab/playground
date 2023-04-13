# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import math
import os
import os.path as osp
from functools import partial

import mmcv
import mmengine

from mmocr.utils import dump_ocr_data


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate training and validation set of TextOCR '
        'by cropping box image.')
    parser.add_argument('root_path', help='Root dir path of TextOCR')
    parser.add_argument(
        'n_proc', default=1, type=int, help='Number of processes to run')
    args = parser.parse_args()
    return args


def process_img(args, src_image_root, dst_image_root):
    # Dirty hack for multi-processing
    img_idx, img_info, anns = args
    src_img = mmcv.imread(osp.join(src_image_root, img_info['file_name']))
    labels = []
    for ann_idx, ann in enumerate(anns):
        text_label = ann['utf8_string']

        # Ignore illegible or non-English words
        if text_label == '.':
            continue

        x, y, w, h = ann['bbox']
        x, y = max(0, math.floor(x)), max(0, math.floor(y))
        w, h = math.ceil(w), math.ceil(h)
        dst_img = src_img[y:y + h, x:x + w]
        dst_img_name = f'img_{img_idx}_{ann_idx}.jpg'
        dst_img_path = osp.join(dst_image_root, dst_img_name)
        mmcv.imwrite(dst_img, dst_img_path)
        labels.append({
            'file_name': dst_img_name,
            'anno_info': [{
                'text': text_label
            }]
        })
    return labels


def convert_textocr(root_path,
                    dst_image_path,
                    dst_label_filename,
                    annotation_filename,
                    img_start_idx=0,
                    nproc=1):

    annotation_path = osp.join(root_path, annotation_filename)
    if not osp.exists(annotation_path):
        raise Exception(
            f'{annotation_path} not exists, please check and try again.')
    src_image_root = root_path

    # outputs
    dst_label_file = osp.join(root_path, dst_label_filename)
    dst_image_root = osp.join(root_path, dst_image_path)
    os.makedirs(dst_image_root, exist_ok=True)

    annotation = mmengine.load(annotation_path)

    process_img_with_path = partial(
        process_img,
        src_image_root=src_image_root,
        dst_image_root=dst_image_root)
    tasks = []
    for img_idx, img_info in enumerate(annotation['imgs'].values()):
        ann_ids = annotation['imgToAnns'][img_info['id']]
        anns = [annotation['anns'][ann_id] for ann_id in ann_ids]
        tasks.append((img_idx + img_start_idx, img_info, anns))
    labels_list = mmengine.track_parallel_progress(
        process_img_with_path, tasks, keep_order=True, nproc=nproc)
    final_labels = []
    for label_list in labels_list:
        final_labels += label_list
    dump_ocr_data(final_labels, dst_label_file, 'textrecog')
    return len(annotation['imgs'])


def main():
    args = parse_args()
    root_path = args.root_path
    print('Processing training set...')
    num_train_imgs = convert_textocr(
        root_path=root_path,
        dst_image_path='image',
        dst_label_filename='train_label.json',
        annotation_filename='TextOCR_0.1_train.json',
        nproc=args.n_proc)
    print('Processing validation set...')
    convert_textocr(
        root_path=root_path,
        dst_image_path='image',
        dst_label_filename='val_label.json',
        annotation_filename='TextOCR_0.1_val.json',
        img_start_idx=num_train_imgs,
        nproc=args.n_proc)
    print('Finish')


if __name__ == '__main__':
    main()
