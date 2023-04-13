# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Sequence, Union

import torch
import torch.nn as nn
from mmengine.model import Sequential
from mmocr.models.common.dictionary import Dictionary
from mmocr.models.textrecog.layers import BidirectionalLSTM
from mmocr.registry import MODELS
from mmocr.structures import TextRecogDataSample

from .base import BaseDecoder


@MODELS.register_module()
class CRNNDecoder(BaseDecoder):
    """Decoder for CRNN.

    Args:
        in_channels (int): Number of input channels.
        dictionary (dict or :obj:`Dictionary`): The config for `Dictionary` or
            the instance of `Dictionary`.
        rnn_flag (bool): Use RNN or CNN as the decoder. Defaults to False.
        module_loss (dict, optional): Config to build module_loss. Defaults
            to None.
        postprocessor (dict, optional): Config to build postprocessor.
            Defaults to None.
        init_cfg (dict or list[dict], optional): Initialization configs.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 dictionary: Union[Dictionary, Dict],
                 rnn_flag: bool = False,
                 module_loss: Dict = None,
                 postprocessor: Dict = None,
                 init_cfg=dict(type='Xavier', layer='Conv2d'),
                 **kwargs):
        super().__init__(
            init_cfg=init_cfg,
            dictionary=dictionary,
            module_loss=module_loss,
            postprocessor=postprocessor)
        self.rnn_flag = rnn_flag
        self.num_classes = self.dictionary.num_classes

        if rnn_flag:
            self.bilstm = nn.LSTM(
                in_channels,
                in_channels // 2,
                bidirectional=True,
                batch_first=True,
                num_layers=2)
        self.decoder = nn.Linear(in_channels, self.num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward_train(
        self,
        feat: torch.Tensor,
        out_enc: Optional[torch.Tensor] = None,
        data_samples: Optional[Sequence[TextRecogDataSample]] = None
    ) -> torch.Tensor:
        """
        Args:
            feat (Tensor): A Tensor of shape :math:`(N, C, 1, W)`.
            out_enc (torch.Tensor, optional): Encoder output. Defaults to None.
            data_samples (list[TextRecogDataSample], optional): Batch of
                TextRecogDataSample, containing gt_text information. Defaults
                to None.

        Returns:
            Tensor: The raw logit tensor. Shape :math:`(N, W, C)` where
            :math:`C` is ``num_classes``.
        """
        if feat.dim() == 4:
            assert feat.size(2) == 1, 'the height of input must be 1'
            feat = feat.squeeze(2)
            # N C W -> N W C
            feat = feat.permute(0, 2, 1)
        if self.rnn_flag:
            feat, _ = self.bilstm(feat)
        feat = self.decoder(feat)
        return feat

    def forward_test(
        self,
        feat: Optional[torch.Tensor] = None,
        out_enc: Optional[torch.Tensor] = None,
        data_samples: Optional[Sequence[TextRecogDataSample]] = None
    ) -> torch.Tensor:
        """
        Args:
            feat (Tensor): A Tensor of shape :math:`(N, C, 1, W)`.
            out_enc (torch.Tensor, optional): Encoder output. Defaults to None.
            data_samples (list[TextRecogDataSample]): Batch of
                TextRecogDataSample, containing ``gt_text`` information.
                Defaults to None.

        Returns:
            Tensor: Character probabilities. of shape
            :math:`(N, self.max_seq_len, C)` where :math:`C` is
            ``num_classes``.
        """
        return self.softmax(self.forward_train(feat, out_enc, data_samples))
