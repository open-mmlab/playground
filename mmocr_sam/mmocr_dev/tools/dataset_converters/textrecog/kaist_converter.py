# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import math
import os
import os.path as osp
import xml.etree.ElementTree as ET

import mmcv
import mmengine

from mmocr.utils import crop_img, dump_ocr_data


def collect_files(img_dir, gt_dir, ratio):
    """Collect all images and their corresponding groundtruth files.

    Args:
        img_dir (str): The image directory
        gt_dir (str): The groundtruth directory
        ratio (float): Split ratio for val set

    Returns:
        files (list): The list of tuples (img_file, groundtruth_file)
    """
    assert isinstance(img_dir, str)
    assert img_dir
    assert isinstance(gt_dir, str)
    assert gt_dir
    assert isinstance(ratio, float)
    assert ratio < 1.0, 'val_ratio should be a float between 0.0 to 1.0'

    ann_list, imgs_list = [], []
    for img_file in os.listdir(img_dir):
        ann_list.append(osp.join(gt_dir, img_file.split('.')[0] + '.xml'))
        imgs_list.append(osp.join(img_dir, img_file))

    all_files = list(zip(sorted(imgs_list), sorted(ann_list)))
    assert len(all_files), f'No images found in {img_dir}'
    print(f'Loaded {len(all_files)} images from {img_dir}')

    trn_files, val_files = [], []
    if ratio > 0:
        for i, file in enumerate(all_files):
            if i % math.floor(1 / ratio):
                trn_files.append(file)
            else:
                val_files.append(file)
    else:
        trn_files, val_files = all_files, []

    print(f'training #{len(trn_files)}, val #{len(val_files)}')

    return trn_files, val_files


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

    if osp.splitext(gt_file)[1] == '.xml':
        img_info = load_xml_info(gt_file, img_info)
    else:
        raise NotImplementedError

    return img_info


def load_xml_info(gt_file, img_info):
    """Collect the annotation information.

    Annotation Format
    <image>
        <imageName>DSC02306.JPG</imageName>
        <resolution x="640" y="480" />
        <words>
            <word x="61" y="140" width="566" height="107">
                <character x="61" y="147" width="75" height="94" char="C" />
                <character x="173" y="147" width="77" height="93" char="L" />
                <character x="251" y="146" width="83" height="96" char="A" />
                <character x="335" y="146" width="75" height="97" char="V" />
                <character x="409" y="140" width="52" height="105" char="I" />
                <character x="464" y="147" width="76" height="96" char="T" />
                <character x="538" y="154" width="89" height="93" char="A" />
            </word>
        </words>
        <illumination>no</illumination>
        <difficulty>2</difficulty>
        <tag>
        </tag>
    </image>

    Args:
        gt_file (str): The path to ground-truth
        img_info (dict): The dict of the img and annotation information

    Returns:
        img_info (dict): The dict of the img and annotation information
    """

    obj = ET.parse(gt_file)
    root = obj.getroot()
    anno_info = []
    for word in root.iter('word'):
        x, y = max(0, int(word.attrib['x'])), max(0, int(word.attrib['y']))
        w, h = int(word.attrib['width']), int(word.attrib['height'])
        bbox = [x, y, x + w, y, x + w, y + h, x, y + h]
        chars = []
        for character in word.iter('character'):
            chars.append(character.attrib['char'])
        word = ''.join(chars)
        if len(word) == 0:
            continue
        anno = dict(bbox=bbox, word=word)
        anno_info.append(anno)

    img_info.update(anno_info=anno_info)

    return img_info


def generate_ann(root_path, split, image_infos, preserve_vertical):
    """Generate cropped annotations and label txt file.

    Args:
        root_path (str): The root path of the dataset
        split (str): The split of dataset. Namely: training or test
        image_infos (list[dict]): A list of dicts of the img and
            annotation information
        preserve_vertical (bool): Whether to preserve vertical texts
        format (str): Annotation format, should be either 'txt' or 'jsonl'
    """

    dst_image_root = osp.join(root_path, 'crops', split)
    ignore_image_root = osp.join(root_path, 'ignores', split)
    if split == 'training':
        dst_label_file = osp.join(root_path, 'train_label.json')
    elif split == 'val':
        dst_label_file = osp.join(root_path, 'val_label.json')
    mmengine.mkdir_or_exist(dst_image_root)
    mmengine.mkdir_or_exist(ignore_image_root)

    img_info = []
    for image_info in image_infos:
        index = 1
        src_img_path = osp.join(root_path, 'imgs', image_info['file_name'])
        image = mmcv.imread(src_img_path)
        src_img_root = image_info['file_name'].split('.')[0]

        for anno in image_info['anno_info']:
            word = anno['word']
            dst_img = crop_img(image, anno['bbox'], 0, 0)
            h, w, _ = dst_img.shape

            dst_img_name = f'{src_img_root}_{index}.png'
            index += 1
            # Skip invalid annotations
            if min(dst_img.shape) == 0:
                continue
            # Filter out vertical texts
            if not preserve_vertical and h / w > 2:
                dst_img_path = osp.join(ignore_image_root, dst_img_name)
                mmcv.imwrite(dst_img, dst_img_path)
                continue

            dst_img_path = osp.join(dst_image_root, dst_img_name)
            mmcv.imwrite(dst_img, dst_img_path)

            img_info.append({
                'file_name': dst_img_name,
                'anno_info': [{
                    'text': word
                }]
            })

    ensure_ascii = dict(ensure_ascii=False)
    dump_ocr_data(img_info, dst_label_file, 'textrecog', **ensure_ascii)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate training and val set of KAIST ')
    parser.add_argument('root_path', help='Root dir path of KAIST')
    parser.add_argument(
        '--val-ratio', help='Split ratio for val set', default=0.0, type=float)
    parser.add_argument(
        '--preserve-vertical',
        help='Preserve samples containing vertical texts',
        action='store_true')
    parser.add_argument(
        '--nproc', default=1, type=int, help='Number of process')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    root_path = args.root_path
    ratio = args.val_ratio

    trn_files, val_files = collect_files(
        osp.join(root_path, 'imgs'), osp.join(root_path, 'annotations'), ratio)

    # Train set
    trn_infos = collect_annotations(trn_files, nproc=args.nproc)
    with mmengine.Timer(
            print_tmpl='It takes {}s to convert KAIST Training annotation'):
        generate_ann(root_path, 'training', trn_infos, args.preserve_vertical)

    # Val set
    if len(val_files) > 0:
        val_infos = collect_annotations(val_files, nproc=args.nproc)
        with mmengine.Timer(
                print_tmpl='It takes {}s to convert KAIST Val annotation'):
            generate_ann(root_path, 'val', val_infos, args.preserve_vertical)


if __name__ == '__main__':
    main()
