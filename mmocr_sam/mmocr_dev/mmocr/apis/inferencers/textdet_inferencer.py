# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict

from mmocr.structures import TextDetDataSample
from .base_mmocr_inferencer import BaseMMOCRInferencer


class TextDetInferencer(BaseMMOCRInferencer):
    """Text Detection inferencer.

    Args:
        model (str, optional): Path to the config file or the model name
            defined in metafile. For example, it could be
            "dbnet_resnet18_fpnc_1200e_icdar2015" or
            "configs/textdet/dbnet/dbnet_resnet18_fpnc_1200e_icdar2015.py".
            If model is not specified, user must provide the
            `weights` saved by MMEngine which contains the config string.
            Defaults to None.
        weights (str, optional): Path to the checkpoint. If it is not specified
            and model is a model name of metafile, the weights will be loaded
            from metafile. Defaults to None.
        device (str, optional): Device to run inference. If None, the available
            device will be automatically used. Defaults to None.
        scope (str, optional): The scope of the model. Defaults to "mmocr".
    """

    def pred2dict(self, data_sample: TextDetDataSample) -> Dict:
        """Extract elements necessary to represent a prediction into a
        dictionary. It's better to contain only basic data elements such as
        strings and numbers in order to guarantee it's json-serializable.

        Args:
            data_sample (TextDetDataSample): The data sample to be converted.

        Returns:
            dict: The output dictionary.
        """
        result = {}
        pred_instances = data_sample.pred_instances
        if 'polygons' in pred_instances:
            result['polygons'] = self._array2list(pred_instances.polygons)
        if 'bboxes' in pred_instances:
            result['bboxes'] = self._array2list(pred_instances.bboxes)
        result['scores'] = self._array2list(pred_instances.scores)
        return result
