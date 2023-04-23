# Copyright (c) OpenMMLab. All rights reserved.
import io
import json
import logging
import os
from urllib.parse import urlparse
import numpy as np
from label_studio_converter import brush
import torch

import cv2

import boto3
from botocore.exceptions import ClientError
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import (DATA_UNDEFINED_NAME, get_image_size,
                                   get_single_tag_keys)
from label_studio_tools.core.utils.io import get_data_dir

from mmdet.apis import inference_detector, init_detector
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
import random
import string
logger = logging.getLogger(__name__)

def load_my_model(device="cuda:0"):
        """
        Loads the Segment Anything model on initializing Label studio, so if you call it outside MyModel it doesn't load every time you try to make a prediction
        Returns the predictor object. For more, look at Facebook's SAM docs
        """
        # if you're not using CUDA, use "cpu" instead
        device = "cuda:0"

        # Note: YOU MUST HAVE THE MODEL SAVED IN THE SAME DIRECTORY AS YOUR BACKEND
        sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")

        sam.to(device=device)
        predictor = SamPredictor(sam)
        print(predictor)
        print("#######################################")
        return predictor

PREDICTOR=load_my_model()


class MMDetection(LabelStudioMLBase):
    """Object detector based on https://github.com/open-mmlab/mmdetection."""

    def __init__(self,
                 config_file=None,
                 checkpoint_file=None,
                 image_dir=None,
                 labels_file=None,
                 score_threshold=0.5,
                 device='cpu',
                 **kwargs):

        super(MMDetection, self).__init__(**kwargs)
        self.PREDICTOR = PREDICTOR

        config_file = config_file or os.environ['config_file']
        checkpoint_file = checkpoint_file or os.environ['checkpoint_file']
        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
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
 
        # if 'RectangleLabels' in self.parsed_label_config:

        #     self.parsed_label_config_RectangleLabels = {
        #         'RectangleLabels':self.parsed_label_config['RectangleLabels']
        #     }
        #     self.from_name, self.to_name, self.value, self.labels_in_config = get_single_tag_keys(  # noqa E501
        #         self.parsed_label_config_RectangleLabels, 'RectangleLabels', 'Image')

        if 'BrushLabels' in self.parsed_label_config:

            self.parsed_label_config_BrushLabels = {
                'BrushLabels':self.parsed_label_config['BrushLabels']
            }
            self.from_name, self.to_name, self.value, self.labels_in_config = get_single_tag_keys(  # noqa E501
                self.parsed_label_config_BrushLabels, 'BrushLabels', 'Image')
        
        if 'BrushLabels' in self.parsed_label_config:

            self.parsed_label_config_BrushLabels = {
                'BrushLabels':self.parsed_label_config['BrushLabels']
            }
            self.from_name_BrushLabels, self.to_name_BrushLabels, self.value_BrushLabels, self.labels_in_config_BrushLabels = get_single_tag_keys(  # noqa E501
                self.parsed_label_config_BrushLabels, 'BrushLabels', 'Image')


        schema = list(self.parsed_label_config.values())[0]
        self.labels_in_config = set(self.labels_in_config)

        # Collect label maps from `predicted_values="airplane,car"` attribute in <Label> tag # noqa E501
        self.labels_attrs = schema.get('labels_attrs')
        if self.labels_attrs:
            for label_name, label_attrs in self.labels_attrs.items():
                for predicted_value in label_attrs.get('predicted_values',
                                                       '').split(','):
                    self.label_map[predicted_value] = label_name

        print('Load new model from: ', config_file, checkpoint_file)
        self.model = init_detector(config_file, checkpoint_file, device=device)
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

        predictor = PREDICTOR

        results=[]

        assert len(tasks) == 1
        task = tasks[0]
        image_url = self._get_image_url(task)
        image_path = self.get_local_path(image_url)


        # getting the height and width of the image that you are annotating real-time 
        height = kwargs['context']['result'][0]['original_height']
        width = kwargs['context']['result'][0]['original_width']

        # getting x and y coordinates of the keypoint
        x = kwargs['context']['result'][0]['value']['x'] * width / 100
        y = kwargs['context']['result'][0]['value']['y'] * height / 100
        output_label = kwargs['context']['result'][0]['value']['labels'][0]



        # image = cv2.imread(f"./{split}")
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # retriving predictions from SAM. For more info, look at Facebook's SAM docs
        predictor.set_image(image)



        # transformed_boxes = predictor.transform.apply_boxes_torch(
        #     bbox, image.shape[:2])
        # transformed_boxes = transformed_boxes.to(predictor.model.device)

        # masks, _, _ = sam_model.predict_torch(
        #     point_coords=None,
        #     point_labels=None,
        #     boxes=transformed_boxes,
        #     multimask_output=False)


        masks, scores, logits = predictor.predict(
            point_coords=np.array([[x, y]]),
            # box=np.array([x.cpu() for x in bbox[:4]]),
            point_labels=np.array([1]),
            multimask_output=False,
        )
        mask = masks[0].astype(np.uint8) # each mask has shape [H, W]
        # converting the mask from the model to RLE format which is usable in Label Studio
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
            "id": ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.ascii_lowercase + string.digits)), # creates a random ID for your label every time
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