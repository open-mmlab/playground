# Copyright (c) OpenMMLab. All rights reserved.
# Refer from https://github.com/Li-Qingyun/sam-mmrotate
import argparse
import os

from data_builder import build_data_loader, build_evaluator

from segment_anything import sam_model_registry, SamPredictor

from mmengine import Config
from mmengine.dist import (collect_results, get_dist_info, get_rank, init_dist,
                           is_distributed)
from mmengine.utils import ProgressBar

from mmrotate.utils import register_all_modules
from mmdet.apis import init_detector

from engine import single_sample_step


def parse_args():
    # sam_checkpoint = r"../segment-anything/checkpoints/sam_vit_b_01ec64.pth"
    # model_type = "vit_b"
    # device = "cuda"

    # ckpt_path = './rotated_fcos_sep_angle_r50_fpn_1x_dota_le90-0be71a0c.pth'
    # model_cfg_path = 'configs/rotated_fcos/rotated-fcos-hbox-le90_r50_fpn_1x_dota.py'

    parser = argparse.ArgumentParser(
        'Evaluation for Zero-shot Oriented Detector with Segment-Anything-'
        'Model Prompt by Predicted HBox of Horizontal Detector', add_help=True)
    # parser.add_argument(  # TODO: get data cfg from a file, instead of hard-code
    #     'data_config', type=str,
    #     help='path to config file contains `data` cfg of the oriented object '
    #          'detection data set for evaluation')
    parser.add_argument(
        'det_config', type=str,
        help='path to config file contains `model` cfg of the detector')
    parser.add_argument(
        'det_weight', type=str, help='path to detector weight file')
    parser.add_argument(
        '--sam-type', type=str, default='vit_h',
        choices=['vit_h', 'vit_l', 'vit_b'],
        help='sam type')
    parser.add_argument(
        '--sam-weight',
        type=str,
        default='../models/sam_vit_h_4b8939.pth',
        help='path to checkpoint file')
    parser.add_argument(
        '--device', '-d', default='cuda', help='device used for inference')
    parser.add_argument('--format-only', action='store_true')
    parser.add_argument('--merge-patches', action='store_true')
    parser.add_argument('--set-min-box', action='store_true')
    parser.add_argument('--result-with-mask', action='store_true')
    parser.add_argument(
        '--max-batch-num-pred', type=int, default=200,
        help='max prediction number of mask generation (avoid OOM)')
    parser.add_argument('--only-det', action='store_true')
    parser.add_argument(
        '--out-dir',
        '-o',
        type=str,
        default='outputs',
        help='output directory')
    parser.add_argument(
        '--box-thr', '-b', type=float, default=0.3, help='box threshold')
    parser.add_argument('--num-worker', '-n', type=int, default=2)

    # dist param
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


if __name__ == '__main__':
    args = parse_args()
    if args.launcher == 'none':
        _distributed = False
    else:
        _distributed = True
    if _distributed and not is_distributed():
        init_dist(args.launcher)

    register_all_modules(init_default_scope=True)

    # prepare the detector model
    detector_cfg = Config.fromfile(args.det_config)
    if 'init_cfg' in detector_cfg.model.backbone:
        detector_cfg.model.backbone.init_cfg = None
    if args.set_min_box:
        detector_cfg.model.test_cfg['min_bbox_size'] = 10
    det_model = init_detector(
        detector_cfg, args.det_weight, device='cpu', cfg_options={})
    det_model = det_model.to(args.device)

    # prepare the sam model
    build_sam = sam_model_registry[args.sam_type]
    sam_model = SamPredictor(build_sam(checkpoint=args.sam_weight))
    sam_model.model = sam_model.model.to(args.device)

    # prepare dataset
    # data_cfg = Config.fromfile(args.data_config)  # TODO: get data cfg from a file, instead of hard-code
    # dataloader = build_data_loader('test_without_hbox')
    dataloader = build_data_loader('trainval_with_hbox')
    evaluator = build_evaluator(args.merge_patches, args.format_only)
    evaluator.dataset_meta = dataloader.dataset.metainfo  # TODO: add assert to make sure the CLASSES in ckpt and in dataset are the same

    if get_rank() == 0:
        print('data len: ', len(dataloader.dataset),
              'num_word_size: ', get_dist_info()[1])

        progress_bar = ProgressBar(len(dataloader))

    det_model.eval()
    for i, data in enumerate(dataloader):

        evaluator = single_sample_step(data, det_model, sam_model, evaluator, args)
        if get_rank() == 0:
            progress_bar.update()

    metrics = evaluator.evaluate(len(dataloader.dataset))
