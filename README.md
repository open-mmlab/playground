<div align=center>
<img src="resources/playground-logo.png"/>
</div>

<div align="center">

English | [简体中文](README_zh-CN.md)

</div>

🥳 🚀 **With the principles of openness, transparency, and cooperation, we encourage community members to join in our projects and explore the edges of AI. Our Playground project is the perfect place to collect exciting and cutting-edge applications related to OpenMMLab, which we constantly update.**

🥳 🚀 **Join the playground now and unleash your creativity in the world of AI!**

The overview diagram is shown below.

<div align=center>
<img src="https://user-images.githubusercontent.com/27466624/231665783-4a97e86c-6f89-4d63-b828-e7c414d1ff2b.png"/>
</div>

Currently, the following applications are included:

|                                                                  |                                                                    Demo                                                                     | Description                      |
| :--------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------- |
|                 [mmdet_sam](mmdet_sam/README.md)                 | <img src="https://user-images.githubusercontent.com/17425982/231419108-bc5ef1ed-cb0b-496a-a19e-9b3b55479426.png" width="50%" height="10%"/> | Detection models + sam           |
|              [mmrotate_sam](mmrotate_sam/README.md)              | <img src="https://user-images.githubusercontent.com/79644233/231568599-58694ec9-a3b1-44a4-833f-74cfb4d4ca45.png" width="50%" height="10%"/> | Rotated object detection + sam   |
|     [mmpose_open_detection](mmpose_open_detection/README.md)     | <img src="https://user-images.githubusercontent.com/8425513/231439110-c0e7d6f8-5692-4bcb-b6cf-c3c243a990a5.jpg" width="50%" height="10%"/>  | Open object detection + mmpose   |
| [mmtracking_open_detection](mmtracking_open_detection/README.md) |                      <img src="https://github.com/zwhus/pictures/raw/main/demo%2B(1).gif" width="50%" height="10%" />                       | Open object detection + tracking |

The following is a detailed description.

## ✨ mmdet_sam

<div align=center>
<img src="https://user-images.githubusercontent.com/27466624/231659917-e3069822-2193-4261-b216-5f53baa64b53.PNG"/>
</div>

We provide a set of applications based on mmdet and sam. The features include:

1. Support all detection models (Closed-Set) included in MMDet, such as Faster R-CNN and DINO, by using SAM for automatic detection and instance segmentation annotation.
2. Support Open-Vocabulary detection models, such as Detic, by using SAM for automatic detection and instance segmentation annotation.
3. Support Grounding Object Detection models, such as Grounding DINO and GLIP, by using SAM for automatic detection and instance segmentation annotation.
4. All models support distributed detection and segmentation evaluation, and automatic COCO JSON export, making it easy for users to evaluate custom data.

Please see [README](mmdet_sam/README.md) for more information.

## ✨ mmrotate_sam

<div align=center>
<img src="https://user-images.githubusercontent.com/27466624/231659969-adf7dd4d-fcec-4677-9105-aa72b2ced00f.PNG"/>
</div>

We provide a set of applications based on mmrotate and sam. The features include:

1. Support Zero-shot Oriented Object Detection with SAM.
2. Perform SAM-based Zero-shot Oriented Object Detection inference on a single image.

Please see [README](mmrotate_sam/README.md) for more information.

## ✨ mmpose_open_detection

<div align=center>
<img src="https://user-images.githubusercontent.com/27466624/231660029-03166059-e8cf-4b17-8aa5-b42f3a52f12a.PNG"/>
</div>

We provide a set of applications based on mmpose and open detection. The features include:

1. Support open detection and pose estimation model inference for a single image or a folder of images.
2. Will soon support inputting different text prompts to achieve pose detection for different object categories in an image.

Please see [README](mmpose_open_detection/README.md) for more information.

## ✨ mmtracking_open_detection

<div align=center>
<img src="https://user-images.githubusercontent.com/27466624/231666666-4f4c5696-df73-45cd-af04-758ea3806a82.png"/>
</div>

We provide an approach based on open object detection and utilizing motion information (Kalman filter) for multi-object tracking.

Please see [README](mmtracking_open_detection/README.md) for more information.
