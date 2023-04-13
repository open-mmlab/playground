# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json

import mmengine
from mmengine.config import Config, DictAction
from mmengine.evaluator import Evaluator
from mmengine.registry import init_default_scope


def parse_args():
    parser = argparse.ArgumentParser(description='Offline evaluation of the '
                                     'prediction saved in pkl format')
    parser.add_argument('config', help='Config of the model')
    parser.add_argument(
        'pkl_results', help='Path to the predictions in '
        'pickle format')
    parser.add_argument(
        'out_json', help='Path to save the json result')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get('default_scope', 'mmocr'))
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    predictions = mmengine.load(args.pkl_results)

    evaluator = Evaluator(cfg.test_evaluator)
    eval_results = evaluator.offline_evaluate(predictions)
    print(json.dumps(eval_results))
    mmengine.dump(eval_results, args.out_json)


if __name__ == '__main__':
    main()
