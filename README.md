<div align=center>
<img src="resources/playground-logo.png"/>
</div>

<div align="center">

English | [简体中文](README_zh-CN.md)

</div>

This project is used to collect interesting applications related to OpenMMLab and will be updated continuously. We are very welcome for community users to participate in these projects. Any interesting applications or demos related to OpenMMLab are welcome to contribute. The overview is shown below.

<div align=center>
<img src="https://user-images.githubusercontent.com/27466624/231665783-4a97e86c-6f89-4d63-b828-e7c414d1ff2b.png"/>
</div>

Currently, the following applications are included:

- `mmdet_sam`: Detection models with SAM
- `mmrotate_sam`: Rotated object detection models with SAM
- `mmpose_open_detection`: Pose estimation based on open detection
- `mmtracking_open_detection`: Object tracking based on open detection

The following is a detailed description.

## mmdet_sam

<div align=center>
<img src="https://user-images.githubusercontent.com/27466624/231659917-e3069822-2193-4261-b216-5f53baa64b53.PNG"/>
</div>

We provide a set of applications based on mmdet and sam. The features include:

1. Support all detection models (Closed-Set) included in MMDet, such as Faster R-CNN and DINO, by using SAM for automatic detection and instance segmentation annotation.
2. Support Open-Vocabulary detection models, such as Detic, by using SAM for automatic detection and instance segmentation annotation.
3. Support Grounding Object Detection models, such as Grounding DINO and GLIP, by using SAM for automatic detection and instance segmentation annotation.
4. All models support distributed detection and segmentation evaluation, and automatic COCO JSON export, making it easy for users to evaluate custom data.

Please see [README](mmdet_sam/README.md) for more information.

## mmrotate_sam

<div align=center>
<img src="https://user-images.githubusercontent.com/27466624/231659969-adf7dd4d-fcec-4677-9105-aa72b2ced00f.PNG"/>
</div>

We provide a set of applications based on mmrotate and sam. The features include:

1. Support Zero-shot Oriented Object Detection with SAM.
2. Perform SAM-based Zero-shot Oriented Object Detection inference on a single image.

Please see [README](mmrotate_sam/README.md) for more information.

## mmpose_open_detection

<div align=center>
<img src="https://user-images.githubusercontent.com/27466624/231660029-03166059-e8cf-4b17-8aa5-b42f3a52f12a.PNG"/>
</div>

We provide a set of applications based on mmpose and open detection. The features include:

1. Support open detection and pose estimation model inference for a single image or a folder of images.
2. Will soon support inputting different text prompts to achieve pose detection for different object categories in an image.

Please see [README](mmpose_open_detection/README.md) for more information.

## mmtracking_open_detection

<div align=center>
<img src="https://user-images.githubusercontent.com/27466624/231666666-4f4c5696-df73-45cd-af04-758ea3806a82.png"/>
</div>

We provide an approach based on open object detection and utilizing motion information (Kalman filter) for multi-object tracking.

Please see [README](mmtracking_open_detection/README.md) for more information.
