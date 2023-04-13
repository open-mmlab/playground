# Copyright (c) OpenMMLab. All rights reserved.
from mmocr.registry import MODELS
from .encoder_decoder_recognizer import EncoderDecoderRecognizer


@MODELS.register_module()
class ABINet(EncoderDecoderRecognizer):
    """Implementation of `Read Like Humans: Autonomous, Bidirectional and
    Iterative LanguageModeling for Scene Text Recognition.

    <https://arxiv.org/pdf/2103.06495.pdf>`_
    """
