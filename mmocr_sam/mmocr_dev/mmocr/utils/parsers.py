# Copyright (c) OpenMMLab. All rights reserved.
import json
import warnings
from typing import Dict, Tuple

from mmocr.registry import TASK_UTILS
from mmocr.utils.string_utils import StringStripper


@TASK_UTILS.register_module()
class LineStrParser:
    """Parse string of one line in annotation file to dict format.

    Args:
        keys (list[str]): Keys in result dict. Defaults to
            ['filename', 'text'].
        keys_idx (list[int]): Value index in sub-string list for each key
            above. Defaults to [0, 1].
        separator (str): Separator to separate string to list of sub-string.
            Defaults to ' '.
    """

    def __init__(self,
                 keys: Tuple[str, str] = ['filename', 'text'],
                 keys_idx: Tuple[int, int] = [0, 1],
                 separator: str = ' ',
                 **kwargs):
        assert isinstance(keys, list)
        assert isinstance(keys_idx, list)
        assert isinstance(separator, str)
        assert len(keys) > 0
        assert len(keys) == len(keys_idx)
        self.keys = keys
        self.keys_idx = keys_idx
        self.separator = separator
        self.strip_cls = StringStripper(**kwargs)

    def __call__(self, in_str: str) -> Dict:
        line_str = self.strip_cls(in_str)
        if len(line_str.split(' ')) > 2:
            msg = 'More than two blank spaces were detected. '
            msg += 'Please use LineJsonParser to handle '
            msg += 'annotations with blanks. '
            msg += 'Check Doc '
            msg += 'https://mmocr.readthedocs.io/en/latest/'
            msg += 'tutorials/blank_recog.html '
            msg += 'for details.'
            warnings.warn(msg, UserWarning)
        line_str = line_str.split(self.separator)
        if len(line_str) <= max(self.keys_idx):
            raise ValueError(
                f'key index: {max(self.keys_idx)} out of range: {line_str}')

        line_info = {}
        for i, key in enumerate(self.keys):
            line_info[key] = line_str[self.keys_idx[i]]
        return line_info


@TASK_UTILS.register_module()
class LineJsonParser:
    """Parse json-string of one line in annotation file to dict format.

    Args:
        keys (list[str]): Keys in both json-string and result dict. Defaults
            to ['filename', 'text'].
    """

    def __init__(self, keys: Tuple[str, str] = ['filename', 'text']) -> None:
        assert isinstance(keys, list)
        assert len(keys) > 0
        self.keys = keys

    def __call__(self, in_str: str) -> Dict:
        line_json_obj = json.loads(in_str)
        line_info = {}
        for key in self.keys:
            if key not in line_json_obj:
                raise Exception(f'key {key} not in line json {line_json_obj}')
            line_info[key] = line_json_obj[key]

        return line_info
