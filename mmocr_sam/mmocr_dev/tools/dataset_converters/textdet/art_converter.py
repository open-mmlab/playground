# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import math
import os.path as osp

import mmcv
import mmengine

from mmocr.utils import convert_annotations


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate training and validation set of ArT ')
    parser.add_argument('root_path', help='Root dir path of ArT')
    parser.add_argument(
        '--val-ratio', help='Split ratio for val set', default=0.0, type=float)
    args = parser.parse_args()
    return args


def collect_art_info(root_path, split, ratio, print_every=1000):
    """Collect the annotation information.

    The annotation format is as the following:
    {
        'gt_1726': # 'gt_1726' is file name
        [
            {
                'transcription': '燎申集团',
                'points': [
                    [141, 199],
                    [237, 201],
                    [313, 236],
                    [357, 283],
                    [359, 300],
                    [309, 261],
                    [233, 230],
                    [140, 231]
                ],
                'language': 'Chinese',
                'illegibility': False
            },
            ...
        ],
        ...
    }


    Args:
        root_path (str): Root path to the dataset
        split (str): Dataset split, which should be 'train' or 'val'
        ratio (float): Split ratio for val set
        print_every (int): Print log info per iteration

    Returns:
        img_info (dict): The dict of the img and annotation information
    """

    annotation_path = osp.join(root_path, 'annotations/train_labels.json')
    if not osp.exists(annotation_path):
        raise Exception(
            f'{annotation_path} not exists, please check and try again.')

    annotation = mmengine.load(annotation_path)
    img_prefixes = annotation.keys()

    trn_files, val_files = [], []
    if ratio > 0:
        for i, file in enumerate(img_prefixes):
            if i % math.floor(1 / ratio):
                trn_files.append(file)
            else:
                val_files.append(file)
    else:
        trn_files, val_files = img_prefixes, []
    print(f'training #{len(trn_files)}, val #{len(val_files)}')

    if split == 'train':
        img_prefixes = trn_files
    elif split == 'val':
        img_prefixes = val_files
    else:
        raise NotImplementedError

    img_infos = []
    for i, prefix in enumerate(img_prefixes):
        if i > 0 and i % print_every == 0:
            print(f'{i}/{len(img_prefixes)}')
        img_file = osp.join(root_path, 'imgs', prefix + '.jpg')
        # Skip not exist images
        if not osp.exists(img_file):
            continue
        img = mmcv.imread(img_file)

        img_info = dict(
            file_name=osp.join(osp.basename(img_file)),
            height=img.shape[0],
            width=img.shape[1],
            segm_file=osp.join(osp.basename(annotation_path)))

        anno_info = []
        for ann in annotation[prefix]:
            segmentation = []
            for x, y in ann['points']:
                segmentation.append(max(0, x))
                segmentation.append(max(0, y))
            xs, ys = segmentation[::2], segmentation[1::2]
            x, y = min(xs), min(ys)
            w, h = max(xs) - x, max(ys) - y
            bbox = [x, y, w, h]
            if ann['transcription'] == '###' or ann['illegibility']:
                iscrowd = 1
            else:
                iscrowd = 0
            anno = dict(
                iscrowd=iscrowd,
                category_id=1,
                bbox=bbox,
                area=w * h,
                segmentation=[segmentation])
            anno_info.append(anno)
        img_info.update(anno_info=anno_info)
        img_infos.append(img_info)

    return img_infos


def main():
    args = parse_args()
    root_path = args.root_path
    print('Processing training set...')
    training_infos = collect_art_info(root_path, 'train', args.val_ratio)
    convert_annotations(training_infos,
                        osp.join(root_path, 'instances_training.json'))
    if args.val_ratio > 0:
        print('Processing validation set...')
        val_infos = collect_art_info(root_path, 'val', args.val_ratio)
        convert_annotations(val_infos, osp.join(root_path,
                                                'instances_val.json'))
    print('Finish')


if __name__ == '__main__':
    main()
