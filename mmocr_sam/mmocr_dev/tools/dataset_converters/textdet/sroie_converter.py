# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

import mmcv
import mmengine
import numpy as np

from mmocr.utils import dump_ocr_data


def collect_files(img_dir, gt_dir):
    """Collect all images and their corresponding groundtruth files.

    Args:
        img_dir (str): The image directory
        gt_dir (str): The groundtruth directory

    Returns:
        files (list): The list of tuples (img_file, groundtruth_file)
    """

    assert isinstance(img_dir, str)
    assert img_dir
    assert isinstance(gt_dir, str)
    assert gt_dir

    ann_list, imgs_list = [], []
    for gt_file in os.listdir(gt_dir):
        # Filtering repeated and missing images
        if '(' in gt_file or gt_file == 'X51006619570.txt':
            continue
        ann_list.append(osp.join(gt_dir, gt_file))
        imgs_list.append(osp.join(img_dir, gt_file.replace('.txt', '.jpg')))

    files = list(zip(sorted(imgs_list), sorted(ann_list)))
    assert len(files), f'No images found in {img_dir}'

    print(f'Loaded {len(files)} images from {img_dir}')

    return files


def collect_annotations(files, nproc=1):
    """Collect the annotation information.

    Args:
        files (list): The list of tuples (image_file, groundtruth_file)
        nproc (int): The number of process to collect annotations

    Returns:
        images (list): The list of image information dicts
    """

    assert isinstance(files, list)
    assert isinstance(nproc, int)

    if nproc > 1:
        images = mmengine.track_parallel_progress(
            load_img_info, files, nproc=nproc)
    else:
        images = mmengine.track_progress(load_img_info, files)

    return images


def load_img_info(files):
    """Load the information of one image.

    Args:
        files (tuple): The tuple of (img_file, groundtruth_file)

    Returns:
        img_info (dict): The dict of the img and annotation information
    """

    assert isinstance(files, tuple)

    img_file, gt_file = files
    assert osp.basename(gt_file).split('.')[0] == osp.basename(img_file).split(
        '.')[0]
    # read imgs while ignoring orientations
    img = mmcv.imread(img_file, 'unchanged')

    img_info = dict(
        file_name=osp.join(osp.basename(img_file)),
        height=img.shape[0],
        width=img.shape[1],
        segm_file=osp.join(osp.basename(gt_file)))

    if osp.splitext(gt_file)[1] == '.txt':
        img_info = load_txt_info(gt_file, img_info)
    else:
        raise NotImplementedError

    return img_info


def load_txt_info(gt_file, img_info):
    """Collect the annotation information.

    Args:
        gt_file (list): The list of tuples (image_file, groundtruth_file)
        img_info (int): The dict of the img and annotation information

    Returns:
        img_info (list): The dict of the img and annotation information
    """

    with open(gt_file, encoding='unicode_escape') as f:
        anno_info = []
        for ann in f.readlines():

            # annotation format [x1, y1, x2, y2, x3, y3, x4, y4, transcript]
            try:
                ann_box = np.array(ann.split(',')[0:8]).astype(int).tolist()
            except ValueError:
                # skip invalid annotation line
                continue
            x = max(0, min(ann_box[0::2]))
            y = max(0, min(ann_box[1::2]))
            w, h = max(ann_box[0::2]) - x, max(ann_box[1::2]) - y
            bbox = [x, y, w, h]
            segmentation = ann_box

            anno = dict(
                iscrowd=0,
                category_id=1,
                bbox=bbox,
                area=w * h,
                segmentation=[segmentation])
            anno_info.append(anno)

    img_info.update(anno_info=anno_info)

    return img_info


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate training and test set of SROIE')
    parser.add_argument('root_path', help='Root dir path of SROIE')
    parser.add_argument(
        '--nproc', default=1, type=int, help='Number of process')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    root_path = args.root_path

    for split in ['training', 'test']:
        print(f'Processing {split} set...')
        with mmengine.Timer(
                print_tmpl='It takes {}s to convert SROIE annotation'):
            files = collect_files(
                osp.join(root_path, 'imgs', split),
                osp.join(root_path, 'annotations', split))
            image_infos = collect_annotations(files, nproc=args.nproc)
            dump_ocr_data(image_infos,
                          osp.join(root_path, 'instances_' + split + '.json'),
                          'textdet')


if __name__ == '__main__':
    main()
