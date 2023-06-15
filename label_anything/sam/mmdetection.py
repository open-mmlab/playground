# Copyright (c) OpenMMLab. All rights reserved.
import io
import json
import logging
import os
from urllib.parse import urlparse
import numpy as np
from label_studio_converter import brush
import torch
from torch.nn import functional as F

import cv2

import boto3
from botocore.exceptions import ClientError
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import (DATA_UNDEFINED_NAME, get_image_size,
                                   get_single_tag_keys)
from label_studio_tools.core.utils.io import get_data_dir

# from mmdet.apis import inference_detector, init_detector
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
from segment_anything.utils.transforms import ResizeLongestSide
import random
import string
logger = logging.getLogger(__name__)


import onnxruntime
import time

def load_my_onnx(encoder_model_abs_path,decoder_model_abs_path):
    # !wget https://huggingface.co/visheratin/segment-anything-vit-b/resolve/main/encoder.onnx
    # !wget https://huggingface.co/visheratin/segment-anything-vit-b/resolve/main/decoder.onnx
    # if onnx_config == 'vit_b':
    #     encoder_model_abs_path = "models/segment_anything_vit_b_encoder_quant.onnx"
    #     decoder_model_abs_path = "models/segment_anything_vit_b_decoder_quant.onnx"
    # elif onnx_config == 'vit_l':
    #     encoder_model_abs_path = "models/segment_anything_vit_l_encoder_quant.onnx"
    #     decoder_model_abs_path = "models/segment_anything_vit_l_decoder_quant.onnx"
    # elif onnx_config == 'vit_h':
    #     encoder_model_abs_path = "models/segment_anything_vit_h_encoder_quant.onnx"
    #     decoder_model_abs_path = "models/segment_anything_vit_h_decoder_quant.onnx"

    providers = onnxruntime.get_available_providers()
    if providers:
        logging.info(
                "Available providers for ONNXRuntime: %s", ", ".join(providers)
            )
    else:
        logging.warning("No available providers for ONNXRuntime")
    encoder_session = onnxruntime.InferenceSession(
            encoder_model_abs_path, providers=providers
            )
    decoder_session = onnxruntime.InferenceSession(
            decoder_model_abs_path, providers=providers
        )

    return encoder_session,decoder_session
 

def load_my_model(device="cuda:0",sam_config="vit_b",sam_checkpoint_file="sam_vit_b_01ec64.pth"):
        """
        Loads the Segment Anything model on initializing Label studio, so if you call it outside MyModel it doesn't load every time you try to make a prediction
        Returns the predictor object. For more, look at Facebook's SAM docs
        """
        sam = sam_model_registry[sam_config](checkpoint=sam_checkpoint_file)
        sam.to(device=device)
        predictor = SamPredictor(sam)
        return predictor



class MMDetection(LabelStudioMLBase):
    """Object detector based on https://github.com/open-mmlab/mmdetection."""

    def __init__(self,
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
                 onnx=False,
                 onnx_encoder_file=None,
                 onnx_decoder_file=None,
                 **kwargs):

        super(MMDetection, self).__init__(**kwargs)

        self.onnx=onnx
        if self.onnx:
            PREDICTOR=load_my_onnx(onnx_encoder_file,onnx_decoder_file)
        else:
            PREDICTOR=load_my_model(device,sam_config)
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
                'RectangleLabels':self.parsed_label_config['RectangleLabels']
            }
            self.from_name_RectangleLabels, self.to_name_RectangleLabels, self.value_RectangleLabels, self.labels_in_config_RectangleLabels = get_single_tag_keys(  # noqa E501
                self.parsed_label_config_RectangleLabels, 'RectangleLabels', 'Image')

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

        if 'PolygonLabels' in self.parsed_label_config and self.out_poly:

            self.parsed_label_config_PolygonLabels = {
                'PolygonLabels':self.parsed_label_config['PolygonLabels']
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

####################################################################################################

    def pre_process(self, image):
        image_size = 1024
        transform = ResizeLongestSide(image_size)

        input_image = transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image, device="cpu")
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
        x = (input_image_torch - pixel_mean) / pixel_std
        h, w = x.shape[-2:]
        padh = image_size - h
        padw = image_size - w
        x = F.pad(x, (0, padw, 0, padh))
        x = x.numpy()

        encoder_inputs = {
            "x": x,
        }
        return encoder_inputs, image.shape[:2]

    def run_encoder(self, encoder_inputs):
        output = self.encoder_session.run(None, encoder_inputs)
        image_embedding = output[0]
        return image_embedding

    def run_decoder(
        self, image_embedding, input_prompt,img_size):
        (original_height,original_width)=img_size
        points=input_prompt['points']
        masks=input_prompt['mask']
        boxes=input_prompt['boxes']
        labels=input_prompt['label']

        image_size = 1024
        transform = ResizeLongestSide(image_size)
        if boxes is not None:
            onnx_box_coords = boxes.reshape(2, 2)
            input_labels = np.array([2,3])

            onnx_coord = np.concatenate([onnx_box_coords, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
            onnx_label = np.concatenate([input_labels, np.array([-1])], axis=0)[None, :].astype(np.float32)
        elif points is not None:
            input_point=points
            input_label = np.array([1])
            onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
            onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(np.float32)

        onnx_coord = transform.apply_coords(onnx_coord, img_size).astype(np.float32)

        onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        onnx_has_mask_input = np.zeros(1, dtype=np.float32)

        
        decoder_inputs = {
            "image_embeddings": image_embedding,
            "point_coords": onnx_coord,
            "point_labels": onnx_label,
            "mask_input": onnx_mask_input,
            "has_mask_input": onnx_has_mask_input,
            "orig_im_size": np.array(
                img_size, dtype=np.float32
            ),
        }
        masks, _, _ = self.decoder_session.run(None, decoder_inputs)
        # masks = masks[0, 0, :, :]  # Only get 1 mask
        masks = masks > 0.0
        # masks = masks.reshape(img_size)
        return masks
##########################################################################################

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
        #共用区域
        start = time.time()
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
        prompt_type = kwargs['context']['result'][0]['type']
        original_height = kwargs['context']['result'][0]['original_height']
        original_width = kwargs['context']['result'][0]['original_width']
        #############################################
        if self.onnx:
            self.encoder_session,self.decoder_session=self.PREDICTOR
            encoder_inputs,_ = self.pre_process(image)

            input_prompt={}

            input_prompt['boxes']=input_prompt['mask']=input_prompt['points']=input_prompt['label']=None
            if prompt_type == 'keypointlabels':
                # getting x and y coordinates of the keypoint
                x = kwargs['context']['result'][0]['value']['x'] * original_width / 100
                y = kwargs['context']['result'][0]['value']['y'] * original_height / 100
                output_label = kwargs['context']['result'][0]['value']['labels'][0]

                input_prompt['points']=np.array([[x, y]])
                input_prompt['label']=np.array([1])

            
            if prompt_type == 'rectanglelabels':

                x = kwargs['context']['result'][0]['value']['x'] * original_width / 100
                y = kwargs['context']['result'][0]['value']['y'] * original_height / 100
                w = kwargs['context']['result'][0]['value']['width'] * original_width / 100
                h = kwargs['context']['result'][0]['value']['height'] * original_height / 100

                output_label = kwargs['context']['result'][0]['value']['rectanglelabels'][0]
            
            
                input_prompt['boxes']=np.array([x, y, x+w, y+h])

                input_prompt['label'] = np.array([2,3])
            
            
            #encoder
            image_embedding = self.run_encoder(encoder_inputs)
            masks = self.run_decoder(image_embedding,input_prompt,\
                                     (original_height,original_width))
            masks = masks[0].astype(np.uint8)
            # mask = masks.astype(np.uint8)
            # shapes = self.post_process(masks, resized_ratio)

        else:
            predictor = self.PREDICTOR

            predictor.set_image(image)
            



            if prompt_type == 'keypointlabels':
                # getting x and y coordinates of the keypoint
                x = kwargs['context']['result'][0]['value']['x'] * original_width / 100
                y = kwargs['context']['result'][0]['value']['y'] * original_height / 100
                output_label = kwargs['context']['result'][0]['value']['labels'][0]


                masks, scores, logits = predictor.predict(
                    point_coords=np.array([[x, y]]),
                    # box=np.array([x.cpu() for x in bbox[:4]]),
                    point_labels=np.array([1]),
                    multimask_output=False,
                )


            if prompt_type == 'rectanglelabels':

                x = kwargs['context']['result'][0]['value']['x'] * original_width / 100
                y = kwargs['context']['result'][0]['value']['y'] * original_height / 100
                w = kwargs['context']['result'][0]['value']['width'] * original_width / 100
                h = kwargs['context']['result'][0]['value']['height'] * original_height / 100

                output_label = kwargs['context']['result'][0]['value']['rectanglelabels'][0]

                masks, scores, logits = predictor.predict(
                    # point_coords=np.array([[x, y]]),
                    box=np.array([x, y, x+w, y+h]),
                    point_labels=np.array([1]),
                    multimask_output=False,
                )


            

            # 找到轮廓
        mask = masks[0].astype(np.uint8) # each mask has shape [H, W]
        # converting the mask from the model to RLE format which is usable in Label Studio
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        end = time.time()
        print(end-start)
########################

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
                "id": ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.ascii_lowercase + string.digits)), # creates a random ID for your label every time
            })


        if self.out_poly:

            points_list = []
            for contour in contours:
                points = []
                for point in contour:
                    x, y = point[0]
                    points.append([float(x)/original_width*100, float(y)/original_height * 100])
                points_list.extend(points)

            # interval = points_list.__len__()//128

            # points_list = points_list[::points_list.__len__()//40]
            results.append({
                "from_name": self.from_name_PolygonLabels,
                "to_name": self.to_name_PolygonLabels,
                "original_width": original_width,
                "original_height": original_height,
                # "image_rotation": 0,
                "value": {
                    "points": points_list,
                    "polygonlabels": [output_label],
                },
                "type": "polygonlabels",
                "id": ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.ascii_lowercase + string.digits)), # creates a random ID for your label every time
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
