# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import time

from mmengine import Registry
from mmengine.config import Config

DATA_OBTAINERS = Registry('data_obtainer')
DATA_CONVERTERS = Registry('data_converter')
DATA_PARSERS = Registry('data_parser')
DATA_DUMPERS = Registry('data_dumper')
CFG_GENERATORS = Registry('cfg_generator')


class DatasetPreparer:
    """Base class of dataset preparer.

    Dataset preparer is used to prepare dataset for MMOCR. It mainly consists
    of three steps:

      1. Obtain the dataset
            - Download
            - Extract
            - Move/Rename
      2. Process the dataset
            - Parse original annotations
            - Convert to mmocr format
            - Dump the annotation file
            - Clean useless files
      3. Generate the base config for this dataset

    After all these steps, the original datasets have been prepared for
    usage in MMOCR. Check out the dataset format used in MMOCR here:
    https://mmocr.readthedocs.io/en/dev-1.x/user_guides/dataset_prepare.html

    Args:
        cfg_path (str): Path to dataset config file.
        dataset_name (str): Dataset name.
        task (str): Task type. Options are 'textdet', 'textrecog',
            'textspotter', and 'kie'. Defaults to 'textdet'.
        nproc (int): Number of parallel processes. Defaults to 4.
        overwrite_cfg (bool): Whether to overwrite the dataset config file if
            it already exists. If False, Dataset Preparer will not generate new
            config for datasets whose configs are already in base.
    """

    def __init__(self,
                 cfg_path: str,
                 dataset_name: str,
                 task: str = 'textdet',
                 nproc: int = 4,
                 overwrite_cfg: bool = False) -> None:
        cfg_path = osp.join(cfg_path, dataset_name)
        self.nproc = nproc
        self.task = task
        self.dataset_name = dataset_name
        self.overwrite_cfg = overwrite_cfg
        self.parse_meta(cfg_path)
        self.parse_cfg(cfg_path)

    def __call__(self):
        """Prepare the dataset."""
        if self.with_obtainer:
            print('Obtaining Dataset...')
            self.data_obtainer()
        if self.with_converter:
            print('Converting Dataset...')
            self.data_converter()
        if self.with_config_generator:
            print('Generating base configs...')
            self.config_generator()

    def parse_meta(self, cfg_path: str) -> None:
        """Parse meta file.

        Args:
            cfg_path (str): Path to meta file.
        """
        try:
            meta = Config.fromfile(osp.join(cfg_path, 'metafile.yml'))
        except FileNotFoundError:
            return
        assert self.task in meta['Data']['Tasks'], \
            f'Task {self.task} not supported!'
        # License related
        if meta['Data']['License']['Type']:
            print(f"\033[1;33;40mDataset Name: {meta['Name']}")
            print(f"License Type: {meta['Data']['License']['Type']}")
            print(f"License Link: {meta['Data']['License']['Link']}")
            print(f"BibTeX: {meta['Paper']['BibTeX']}\033[0m")
            print(
                '\033[1;31;43mMMOCR does not own the dataset. Using this '
                'dataset you must accept the license provided by the owners, '
                'and cite the corresponding papers appropriately.')
            print('If you do not agree with the above license, please cancel '
                  'the progress immediately by pressing ctrl+c. Otherwise, '
                  'you are deemed to accept the terms and conditions.\033[0m')
            for i in range(5):
                print(f'{5-i}...')
                time.sleep(1)

    def parse_cfg(self, cfg_path: str) -> None:
        """Parse dataset config file.

        Args:
            cfg_path (str): Path to dataset config file.
        """
        cfg_path = osp.join(cfg_path, self.task + '.py')
        assert osp.exists(cfg_path), f'Config file {cfg_path} not found!'
        cfg = Config.fromfile(cfg_path)

        if 'data_obtainer' in cfg:
            cfg.data_obtainer.update(task=self.task)
            self.data_obtainer = DATA_OBTAINERS.build(cfg.data_obtainer)
        if 'data_converter' in cfg:
            cfg.data_converter.update(
                dict(nproc=self.nproc, dataset_name=self.dataset_name))
            self.data_converter = DATA_CONVERTERS.build(cfg.data_converter)
        if 'config_generator' in cfg:
            cfg.config_generator.update(
                dict(
                    dataset_name=self.dataset_name,
                    overwrite_cfg=self.overwrite_cfg))
            self.config_generator = CFG_GENERATORS.build(cfg.config_generator)

    @property
    def with_obtainer(self) -> bool:
        """bool: whether the data preparer has an obtainer"""
        return getattr(self, 'data_obtainer', None) is not None

    @property
    def with_converter(self) -> bool:
        """bool: whether the data preparer has an converter"""
        return getattr(self, 'data_converter', None) is not None

    @property
    def with_config_generator(self) -> bool:
        """bool: whether the data preparer has a config generator"""
        return getattr(self, 'config_generator', None) is not None
