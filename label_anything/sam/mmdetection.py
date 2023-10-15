# Copyright (c) OpenMMLab. All rights reserved.
import io
import json
import logging
import os
import random
import string
from urllib.parse import urlparse

import boto3
import cv2
import numpy as np
import torch
from botocore.exceptions import ClientError
from label_studio_converter import brush
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import (DATA_UNDEFINED_NAME, get_image_size,
                                   get_single_tag_keys)
from label_studio_tools.core.utils.io import get_data_dir
from filter_poly import NearNeighborRemover
import pdb
# from mmdet.apis import inference_detector, init_detector

logger = logging.getLogger(__name__)


def load_my_model(
        model_name="sam_hq",
        device="cuda:0",
        sam_config="vit_b",
        sam_checkpoint_file="sam_hq_vit_b.pth"):
    """
    Loads the Segment Anything model on initializing Label studio, so if you call it outside MyModel it doesn't load every time you try to make a prediction
    Returns the predictor object. For more, look at Facebook's SAM docs
    """
    if model_name == "sam":
        try:
            from segment_anything import SamPredictor, sam_model_registry
        except:
            raise ModuleNotFoundError(
                "segment_anything is not installed, run `pip install segment_anything` to install")
        sam = sam_model_registry[sam_config](checkpoint=sam_checkpoint_file)
        sam.to(device=device)
        predictor = SamPredictor(sam)
        return predictor
    elif model_name == "sam_hq":
        from segment_anything_hq import sam_model_registry, SamPredictor
        sam = sam_model_registry[sam_config](checkpoint=sam_checkpoint_file)
        sam.to(device=device)
        predictor = SamPredictor(sam)
        return predictor
    elif model_name == "mobile_sam":
        from models.mobile_sam import SamPredictor, sam_model_registry
        sam = sam_model_registry[sam_config](checkpoint=sam_checkpoint_file)
        sam.to(device=device)
        predictor = SamPredictor(sam)
        return predictor
    else:
        raise NotImplementedError


class MMDetection(LabelStudioMLBase):
    """Object detector based on https://github.com/open-mmlab/mmdetection."""

    def __init__(self,
                 model_name="sam_hq",
                 config_file=None,
                 checkpoint_file=None,
                 sam_config='vit_b',
                 sam_checkpoint_file=None,
                 image_dir=None,
                 labels_file=None,
                 out_mask=True,
                 out_bbox=False,
                 out_poly=False,
                 score_threshold=0.5,
                 device='cpu',
                 **kwargs):

        super(MMDetection, self).__init__(**kwargs)

        PREDICTOR = load_my_model(
            model_name, device, sam_config, sam_checkpoint_file)
        self.PREDICTOR = PREDICTOR

        self.out_mask = out_mask
        self.out_bbox = out_bbox
        self.out_poly = out_poly
        # config_file = config_file or os.environ['config_file']
        # checkpoint_file = checkpoint_file or os.environ['checkpoint_file']
        # self.config_file = config_file
        # self.checkpoint_file = checkpoint_file
        self.labels_file = labels_file
        # default Label Studio image upload folder
        upload_dir = os.path.join(get_data_dir(), 'media', 'upload')
        self.image_dir = image_dir or upload_dir
        logger.debug(
            f'{self.__class__.__name__} reads images from {self.image_dir}')
        if self.labels_file and os.path.exists(self.labels_file):
            self.label_map = json_load(self.labels_file)
        else:
            self.label_map = {}

        # self.from_name, self.to_name, self.value, self.labels_in_config = get_single_tag_keys(  # noqa E501
        #     self.parsed_label_config, 'RectangleLabels', 'Image')

        self.labels_in_config = dict(
            label=self.parsed_label_config['KeyPointLabels']
        )

        if 'RectangleLabels' in self.parsed_label_config and self.out_bbox:

            self.parsed_label_config_RectangleLabels = {
                'RectangleLabels': self.parsed_label_config['RectangleLabels']
            }
            self.from_name_RectangleLabels, self.to_name_RectangleLabels, self.value_RectangleLabels, self.labels_in_config_RectangleLabels = get_single_tag_keys(  # noqa E501
                self.parsed_label_config_RectangleLabels, 'RectangleLabels', 'Image')

        if 'BrushLabels' in self.parsed_label_config:

            self.parsed_label_config_BrushLabels = {
                'BrushLabels': self.parsed_label_config['BrushLabels']
            }
            self.from_name, self.to_name, self.value, self.labels_in_config = get_single_tag_keys(  # noqa E501
                self.parsed_label_config_BrushLabels, 'BrushLabels', 'Image')

        if 'BrushLabels' in self.parsed_label_config:

            self.parsed_label_config_BrushLabels = {
                'BrushLabels': self.parsed_label_config['BrushLabels']
            }
            self.from_name_BrushLabels, self.to_name_BrushLabels, self.value_BrushLabels, self.labels_in_config_BrushLabels = get_single_tag_keys(  # noqa E501
                self.parsed_label_config_BrushLabels, 'BrushLabels', 'Image')

        if 'PolygonLabels' in self.parsed_label_config and self.out_poly:

            self.parsed_label_config_PolygonLabels = {
                'PolygonLabels': self.parsed_label_config['PolygonLabels']
            }
            self.from_name_PolygonLabels, self.to_name_PolygonLabels, self.value_PolygonLabels, self.labels_in_config_PolygonLabels = get_single_tag_keys(  # noqa E501
                self.parsed_label_config_PolygonLabels, 'PolygonLabels', 'Image')

        schema = list(self.parsed_label_config.values())[0]
        self.labels_in_config = set(self.labels_in_config)

        # Collect label maps from `predicted_values="airplane,car"` attribute in <Label> tag # noqa E501
        self.labels_attrs = schema.get('labels_attrs')
        if self.labels_attrs:
            for label_name, label_attrs in self.labels_attrs.items():
                for predicted_value in label_attrs.get('predicted_values',
                                                       '').split(','):
                    self.label_map[predicted_value] = label_name

        # print('Load new model from: ', config_file, checkpoint_file)
        # self.model = init_detector(config_file, checkpoint_file, device=device)
        self.score_thresh = score_threshold

    def _get_image_url(self, task):
        image_url = task['data'].get(
            self.value) or task['data'].get(DATA_UNDEFINED_NAME)
        if image_url.startswith('s3://'):
            # presign s3 url
            r = urlparse(image_url, allow_fragments=False)
            bucket_name = r.netloc
            key = r.path.lstrip('/')
            client = boto3.client('s3')
            try:
                image_url = client.generate_presigned_url(
                    ClientMethod='get_object',
                    Params={
                        'Bucket': bucket_name,
                        'Key': key
                    })
            except ClientError as exc:
                logger.warning(
                    f'Can\'t generate presigned URL for {image_url}. Reason: {exc}'  # noqa E501
                )
        return image_url

    def predict(self, tasks, **kwargs):

        predictor = self.PREDICTOR

        results = []
        assert len(tasks) == 1
        task = tasks[0]
        image_url = self._get_image_url(task)
        image_path = self.get_local_path(image_url)

        if kwargs.get('context') is None:
            return []

        # image = cv2.imread(f"./{split}")
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)

        prompt_type = kwargs['context']['result'][0]['type']
        original_height = kwargs['context']['result'][0]['original_height']
        original_width = kwargs['context']['result'][0]['original_width']

        if prompt_type == 'keypointlabels':
            # getting x and y coordinates of the keypoint
            x = kwargs['context']['result'][0]['value']['x'] * \
                original_width / 100
            y = kwargs['context']['result'][0]['value']['y'] * \
                original_height / 100
            output_label = kwargs['context']['result'][0]['value']['labels'][0]

            masks, scores, logits = predictor.predict(
                point_coords=np.array([[x, y]]),
                # box=np.array([x.cpu() for x in bbox[:4]]),
                point_labels=np.array([1]),
                multimask_output=False,
            )

        if prompt_type == 'rectanglelabels':

            x = kwargs['context']['result'][0]['value']['x'] * \
                original_width / 100
            y = kwargs['context']['result'][0]['value']['y'] * \
                original_height / 100
            w = kwargs['context']['result'][0]['value']['width'] * \
                original_width / 100
            h = kwargs['context']['result'][0]['value']['height'] * \
                original_height / 100

            output_label = kwargs['context']['result'][0]['value']['rectanglelabels'][0]

            masks, scores, logits = predictor.predict(
                # point_coords=np.array([[x, y]]),
                box=np.array([x, y, x+w, y+h]),
                point_labels=np.array([1]),
                multimask_output=False,
            )
        mask = masks[0].astype(np.uint8)  # each mask has shape [H, W]
        # converting the mask from the model to RLE format which is usable in Label Studio

        # 找到轮廓
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 计算外接矩形

        if self.out_bbox:
            new_contours = []
            for contour in contours:
                new_contours.extend(list(contour))
            new_contours = np.array(new_contours)
            x, y, w, h = cv2.boundingRect(new_contours)
            print(x, y, w, h)
            results.append({
                'from_name': self.from_name_RectangleLabels,
                'to_name': self.to_name_RectangleLabels,
                'type': 'rectanglelabels',
                'value': {
                    'rectanglelabels': [output_label],
                    'x': float(x) / original_width * 100,
                    'y': float(y) / original_height * 100,
                    'width': float(w) / original_width * 100,
                    'height': float(h) / original_height * 100,
                },
                "id": ''.join(random.SystemRandom().choices(string.ascii_uppercase + string.ascii_lowercase + string.digits, k=6)), # creates a random ID for your label every time
            })

        if self.out_poly:

            points_list = []
            for contour in contours:
                points = []
                for point in contour:
                    x, y = point[0]
                    points.append([float(x)/original_width*100,
                                  float(y)/original_height * 100])
                points_list.extend(points)
            filterd_points=NearNeighborRemover(distance_threshold=0.4).remove_near_neighbors(points_list) # remove near neighbors (increase distance_threshold to reduce more points)
            # interval = points_list.__len__()//128

            # points_list = points_list[::points_list.__len__()//40]
            results.append({
                "from_name": self.from_name_PolygonLabels,
                "to_name": self.to_name_PolygonLabels,
                "original_width": original_width,
                "original_height": original_height,
                # "image_rotation": 0,
                "value": {
                    "points": filterd_points,
                    "polygonlabels": [output_label],
                },
                "type": "polygonlabels",
                "id": ''.join(random.SystemRandom().choices(string.ascii_uppercase + string.ascii_lowercase + string.digits, k=6)), # creates a random ID for your label every time
                "readonly": False,
            })

        if self.out_mask:
            mask = mask * 255
            rle = brush.mask2rle(mask)

            results.append({
                "from_name": self.from_name_BrushLabels,
                "to_name": self.to_name_BrushLabels,
                # "original_width": width,
                # "original_height": height,
                # "image_rotation": 0,
                "value": {
                    "format": "rle",
                    "rle": rle,
                    "brushlabels": [output_label],
                },
                "type": "brushlabels",
                "id": ''.join(random.SystemRandom().choices(string.ascii_uppercase + string.ascii_lowercase + string.digits, k=6)), # creates a random ID for your label every time
                "readonly": False,
            })

        return [{'result': results}]


def json_load(file, int_keys=False):
    with io.open(file, encoding='utf8') as f:
        data = json.load(f)
        if int_keys:
            return {int(k): v for k, v in data.items()}
        else:
            return data
