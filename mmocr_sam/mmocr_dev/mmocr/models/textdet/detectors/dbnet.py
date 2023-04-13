# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Sequence

import torch
from mmocr.registry import MODELS
from mmocr.structures import TextDetDataSample
from torchvision.ops import RoIAlign

from .single_stage_text_detector import SingleStageTextDetector


@MODELS.register_module()
class DBNet(SingleStageTextDetector):
    """The class for implementing DBNet text detector: Real-time Scene Text
    Detection with Differentiable Binarization.

    [https://arxiv.org/abs/1911.08947].
    """

    def predict_new(
        self,
        inputs: torch.Tensor,
        data_samples: Sequence[TextDetDataSample],
        get_roi_features=False,
        roi_stride=4,
    ) -> Sequence[TextDetDataSample]:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (torch.Tensor): Images of shape (N, C, H, W).
            data_samples (list[TextDetDataSample]): A list of N
                datasamples, containing meta information and gold annotations
                for each of the images.
            get_roi_features (bool): Whether to get roi features.

        Returns:
            list[TextDetDataSample]: A list of N datasamples of prediction
            results.  Each DetDataSample usually contain
            'pred_instances'. And the ``pred_instances`` usually
            contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - polygons (list[np.ndarray]): The length is num_instances.
                    Each element represents the polygon of the
                    instance, in (xn, yn) order.
        """
        x = self.extract_feat(inputs)
        if get_roi_features:
            return self.roi_feature_extraction(x, data_samples, roi_stride)
        else:
            return self.det_head.predict(x, data_samples)

    def roi_feature_extraction(self, feature_maps, data_samples, roi_stride=4):
        """
        Extracts RoI features from the given feature maps for the given
        ground-truth instances.

        Args:
            feature_maps: torch.Tensor of shape (N, C, H, W) representing the
                input feature maps.
            data_samples: list[TextDetDataSample] representing the ground-truth
                instances.
            roi_stride: int representing the stride of the RoI pooling layer.
        Returns:
            roi_outs (list[Dict]): list of roi features. The length of the list
            equals to the batch size. Each element is a dict with the following
            keys:
                - img_path (str): Name of image.
                - roi_features (list[dict]): List of roi features with the
                    following keys:
                    - bbox (list[float]): Bounding box of roi.
                    - roi_feature (torch.Tensor): Feature of roi. The dimension
                        should be C, C is the number of channels (Equal to 256)
        """
        roi_outs = []
        roi_align = RoIAlign(
            output_size=1, spatial_scale=1 / roi_stride, sampling_ratio=2)
        H, W = feature_maps.shape[-2:]
        rois = []
        for batch_idx, data_sample in enumerate(data_samples):
            gt_instances = data_sample.gt_instances
            roi_single_image = self.construct_rois(gt_instances, roi_stride, H,
                                                   W, batch_idx)
            rois.append(roi_single_image)
        rois = torch.cat(rois, dim=0)
        roi_features = roi_align(feature_maps, rois)
        roi_features = roi_features.squeeze(dim=3).squeeze(dim=2)

        roi_idx_count = 0
        for data_sample in data_samples:
            roi_out_current_image = dict()
            roi_out_current_image['img_path'] = data_sample.img_path
            roi_out_current_image['roi_features'] = []
            gt_instances = data_sample.gt_instances
            for gt_instance in gt_instances:
                roi_out_current_image['roi_features'].append(
                    dict(
                        bbox=gt_instance.bboxes,
                        roi_feature=roi_features[roi_idx_count]))
                roi_idx_count += 1
            roi_outs.append(roi_out_current_image)
        return roi_outs

    def construct_rois(self, gt_instances, roi_stride, H, W, batch_idx):
        """
        Constructs RoIs from the given ground-truth instances.

        Args:
            gt_instances: list[TextInstance] representing the ground-truth
                instances.
            roi_stride: int representing the stride of the RoI pooling layer.
            H: int representing the height of the feature map.
            W: int representing the width of the feature map.
            batch_idx: int representing the batch index of the RoIs.

        Returns:
            rois: torch.Tensor of shape (num_rois, 5), where each row is
                [batch_index, x1, y1, x2, y2] representing the RoI coordinates 
                in the feature map space.
        """
        rois = []
        for gt_instance in gt_instances:
            bbox = gt_instance.bboxes
            x1, y1, x2, y2 = bbox.squeeze(dim=0)
            x1 = x1 / roi_stride
            y1 = y1 / roi_stride
            x2 = x2 / roi_stride
            y2 = y2 / roi_stride
            x1 = torch.clamp(x1, 0, W - 1)
            y1 = torch.clamp(y1, 0, H - 1)
            x2 = torch.clamp(x2, 0, W - 1)
            y2 = torch.clamp(y2, 0, H - 1)
            batch_idx = torch.tensor(batch_idx).type_as(x1)
            rois.append(torch.tensor([batch_idx, x1, y1, x2, y2]))
        rois = torch.stack(rois, dim=0)
        return rois
