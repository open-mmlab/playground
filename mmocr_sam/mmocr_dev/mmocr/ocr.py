# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmocr.apis.inferencers import MMOCRInferencer


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--inputs',
        type=str,
        default=
        '/media/jiangqing/jqnas/projects/TextCLIP/data/LAION400M/data/part-00000-reordered/imgs',
        help='Input image file or folder path.')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='results/',
        help='Output directory of results.')
    parser.add_argument(
        '--det',
        type=str,
        default=
        'configs/textdet/dbnetpp/dbnetpp_swinv2_base_w16_in21k_LAION400M.py',
        help='Pretrained text detection algorithm. It\'s the path to the '
        'config file or the model name defined in metafile.')
    parser.add_argument(
        '--det-weights',
        type=str,
        default='checkpoints/db_swin_mix_pretrain.pth',
        help='Path to the custom checkpoint file of the selected det model. '
        'If it is not specified and "det" is a model name of metafile, the '
        'weights will be loaded from metafile.')
    parser.add_argument(
        '--rec',
        type=str,
        default='configs/textrecog/unirec/unirec_hiertext.py',
        help='Pretrained text recognition algorithm. It\'s the path to the '
        'config file or the model name defined in metafile.')
    parser.add_argument(
        '--rec-weights',
        type=str,
        default='checkpoints/unirec.pth',
        help='Path to the custom checkpoint file of the selected recog model. '
        'If it is not specified and "rec" is a model name of metafile, the '
        'weights will be loaded from metafile.')
    parser.add_argument(
        '--kie',
        type=str,
        default=None,
        help='Pretrained key information extraction algorithm. It\'s the path'
        'to the config file or the model name defined in metafile.')
    parser.add_argument(
        '--kie-weights',
        type=str,
        default=None,
        help='Path to the custom checkpoint file of the selected kie model. '
        'If it is not specified and "kie" is a model name of metafile, the '
        'weights will be loaded from metafile.')
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device used for inference. '
        'If not specified, the available device will be automatically used.')
    parser.add_argument(
        '--batch-size', type=int, default=1, help='Inference batch size.')
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display the image in a popup window.')
    parser.add_argument(
        '--print-result',
        action='store_true',
        help='Whether to print the results.')
    parser.add_argument(
        '--save_pred',
        action='store_true',
        default=True,
        help='Save the inference results to out_dir.')
    parser.add_argument(
        '--save_vis',
        action='store_true',
        default=True,
        help='Save the visualization results to out_dir.')

    call_args = vars(parser.parse_args())

    init_kws = [
        'det', 'det_weights', 'rec', 'rec_weights', 'kie', 'kie_weights',
        'device'
    ]
    init_args = {}
    for init_kw in init_kws:
        init_args[init_kw] = call_args.pop(init_kw)

    return init_args, call_args


def main():
    init_args, call_args = parse_args()
    ocr = MMOCRInferencer(**init_args)
    ocr(**call_args)


if __name__ == '__main__':
    main()
