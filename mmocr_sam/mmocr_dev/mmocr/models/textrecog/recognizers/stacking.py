from typing import Dict, Optional, Sequence, Union

import torch
import torch.nn as nn
from mmengine import Config
from mmengine.runner.checkpoint import _load_checkpoint_to_model
from mmocr.models.common.dictionary import Dictionary
from mmocr.models.textrecog.decoders import BaseDecoder
from mmocr.models.textrecog.recognizers import BaseRecognizer
from mmocr.registry import MODELS
from mmocr.structures import TextRecogDataSample
from mmocr.utils.typing_utils import (OptRecSampleList, RecForwardResults,
                                      RecSampleList)


@MODELS.register_module()
class StackingDecoder(BaseDecoder):

    def __init__(
            self,
            num_recognizers,
            dictionary: Union[Dictionary, Dict],
            module_loss: Dict = None,
            postprocessor: Dict = None,
            max_seq_len: int = 48,
            init_cfg=dict(type='Xavier', layer='Conv2d'),
    ):
        super().__init__(
            init_cfg=init_cfg,
            dictionary=dictionary,
            module_loss=module_loss,
            max_seq_len=max_seq_len,
            postprocessor=postprocessor)
        self.num_classes = self.dictionary.num_classes
        self.stacking_fc = nn.Linear(num_recognizers * self.num_classes,
                                     self.num_classes)
        self.stacking_fc.weight.data.fill_(1 / num_recognizers)
        self.stacking_fc.bias.data.fill_(0)
        self.softmax = nn.Softmax(dim=-1)

    def forward_train(
        self,
        stacking_logits: torch.Tensor = None,
        out_enc: Optional[torch.Tensor] = None,
        data_samples: Optional[Sequence[TextRecogDataSample]] = None
    ) -> torch.Tensor:
        input = torch.cat(stacking_logits, dim=-1)
        logits = self.stacking_fc(input)
        return logits

    def forward_test(
        self,
        stacking_logits: torch.Tensor = None,
        out_enc: Optional[torch.Tensor] = None,
        data_samples: Optional[Sequence[TextRecogDataSample]] = None
    ) -> torch.Tensor:
        return self.softmax(
            self.forward_train(stacking_logits, out_enc, data_samples))


@MODELS.register_module()
class StackingRecognizer(BaseRecognizer):
    """Stacking multiple recognizers with ensemble strategy.

    Args:
        model_cfg_paths (list[str]): Paths to model config files.
        decoder (dict): Config of stacking decoder.
    """

    def __init__(self,
                 model_cfg_paths=None,
                 model_ckpts_paths=None,
                 decoder=None,
                 data_preprocessor=None):
        super().__init__(data_preprocessor=data_preprocessor)
        self.stacking_models = nn.ModuleList(
            MODELS.build(Config.fromfile(model_cfg_path).model)
            for model_cfg_path in model_cfg_paths)

        # load checkpoint for every model
        for model, model_ckpt_path in zip(self.stacking_models,
                                          model_ckpts_paths):
            checkpoint = torch.load(model_ckpt_path, map_location='cpu')
            _load_checkpoint_to_model(model, checkpoint)

        # close the grad of stacking models
        for model in self.stacking_models:
            for param in model.parameters():
                param.requires_grad = False

        # get every model to Module

        self.decoder = MODELS.build(decoder)

    def extract_feat(self, inputs: torch.Tensor) -> torch.Tensor:
        pass

    def loss(self, inputs: torch.Tensor, data_samples: RecSampleList,
             **kwargs) -> Dict:
        """Calculate losses from a batch of inputs and data samples.
        Args:
            inputs (tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            data_samples (list[TextRecogDataSample]): A list of N
                datasamples, containing meta information and gold
                annotations for each of the images.

        Returns:
            dict[str, tensor]: A dictionary of loss components.
        """
        # concat inputs at dimension 0
        stacking_logits = [
            model._forward(inputs, data_samples)
            for model in self.stacking_models
        ]
        return self.decoder.loss(stacking_logits, stacking_logits,
                                 data_samples)

    def predict(self, inputs: torch.Tensor, data_samples: RecSampleList,
                **kwargs) -> RecSampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (torch.Tensor): Image input tensor.
            data_samples (list[TextRecogDataSample]): A list of N datasamples,
                containing meta information and gold annotations for each of
                the images.

        Returns:
            list[TextRecogDataSample]:  A list of N datasamples of prediction
            results. Results are stored in ``pred_text``.
        """
        stacking_logits = [
            model._forward(inputs, data_samples)
            for model in self.stacking_models
        ]
        return self.decoder.predict(stacking_logits, stacking_logits,
                                    data_samples)

    def _forward(self,
                 inputs: torch.Tensor,
                 data_samples: OptRecSampleList = None,
                 **kwargs) -> RecForwardResults:
        """Network forward process. Usually includes backbone, encoder and
        decoder forward without any post-processing.

         Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (list[TextRecogDataSample]): A list of N
                datasamples, containing meta information and gold
                annotations for each of the images.

        Returns:
            Tensor: A tuple of features from ``decoder`` forward.
        """
        stacking_logits = [
            model._forward(inputs, data_samples)
            for model in self.stacking_models
        ]
        return self.decoder(stacking_logits, stacking_logits, data_samples)
