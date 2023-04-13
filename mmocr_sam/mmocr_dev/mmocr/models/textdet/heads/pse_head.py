# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

from mmocr.registry import MODELS
from . import PANHead


@MODELS.register_module()
class PSEHead(PANHead):
    """The class for PSENet head.

    Args:
        in_channels (list[int]): A list of numbers of input channels.
        hidden_dim (int): The hidden dimension of the first convolutional
            layer.
        out_channel (int): Number of output channels.
        module_loss (dict): Configuration dictionary for loss type. Supported
            loss types are "PANModuleLoss" and "PSEModuleLoss". Defaults to
            PSEModuleLoss.
        postprocessor (dict): Config of postprocessor for PSENet.
        init_cfg (dict or list[dict], optional): Initialization configs.
    """

    def __init__(self,
                 in_channels: List[int],
                 hidden_dim: int,
                 out_channel: int,
                 module_loss: Dict = dict(type='PSEModuleLoss'),
                 postprocessor: Dict = dict(
                     type='PSEPostprocessor', text_repr_type='poly'),
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None) -> None:

        super().__init__(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            out_channel=out_channel,
            module_loss=module_loss,
            postprocessor=postprocessor,
            init_cfg=init_cfg)
