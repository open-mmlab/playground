<div align=center>
<img src="resources/playground-logo.png"/>
</div>

<div align="center">

English | [简体中文](README_zh-CN.md)

</div>

🥳 🚀 **With the principles of openness, transparency, and cooperation, we encourage community members to join in our projects and explore the edges of AI. Our Playground project is the perfect place to collect exciting and cutting-edge applications related to OpenMMLab, which we constantly update.**

🥳 🚀 **Join the playground now and unleash your creativity in the world of AI!**

<div align="center">
<br>
  <a href="https://openmmlab.medium.com/" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/219255827-67c1a27f-f8c5-46a9-811d-5e57448c61d1.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://discord.com/channels/1037617289144569886/1046608014234370059" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218347213-c080267f-cbb6-443e-8532-8e1ed9a58ea9.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://twitter.com/OpenMMLab" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218346637-d30c8a0f-3eba-4699-8131-512fb06d46db.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://www.youtube.com/openmmlab" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218346691-ceb2116a-465a-40af-8424-9f30d2348ca9.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://space.bilibili.com/1293512903" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/219026751-d7d14cce-a7c9-4e82-9942-8375fca65b99.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://www.zhihu.com/people/openmmlab" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/219026120-ba71e48b-6e94-4bd4-b4e9-b7d175b5e362.png" width="3%" alt="" /></a>
<br>
</div>

Currently, the following applications are included:

|                                                                  |                                                                    Demo                                                                     | Description                                                                                                                                                                                                               |
| :--------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
|                 [mmdet_sam](mmdet_sam/README.md)                 | <img src="https://user-images.githubusercontent.com/17425982/231419108-bc5ef1ed-cb0b-496a-a19e-9b3b55479426.png" width="70%" height="20%"/> | Detection models + SAM. Let's explore a new way of instance segmentation by combining Closed-Set Object Detection, Open-Vocabulary Object Detection, Grounding Object Detection, and SAM in a fun and exciting way!       |
|              [mmrotate_sam](mmrotate_sam/README.md)              | <img src="https://user-images.githubusercontent.com/79644233/231568599-58694ec9-a3b1-44a4-833f-74cfb4d4ca45.png" width="70%" height="20%"/> | Rotated object detection + SAM. Let's join SAM and weakly supervised horizontal box detection to achieve rotated box detection, and say goodbye to the tedious task of annotating rotated boxes from now on!              |
|     [mmpose_open_detection](mmpose_open_detection/README.md)     | <img src="https://user-images.githubusercontent.com/8425513/231439110-c0e7d6f8-5692-4bcb-b6cf-c3c243a990a5.jpg" width="70%" height="20%"/>  | Open object detection + mmpose. Let's explore the combination of open object detection and various pose estimation algorithms to achieve "Pose All Things" - the ability to estimate the pose of anything and everything! |
| [mmtracking_open_detection](mmtracking_open_detection/README.md) |                      <img src="https://github.com/zwhus/pictures/raw/main/demo%2B(1).gif" width="70%" height="10%" />                       | Open object detection + tracking. Let's combine open object detection with video tasks to easily achieve video tracking and segmentation for open categories!                                                             |

The following is a detailed description.

## ✨ mmdet_sam

<div align=center>
<img src="https://user-images.githubusercontent.com/27466624/231659917-e3069822-2193-4261-b216-5f53baa64b53.PNG"/>
</div>

We provide a set of applications based on MMDet and SAM. The features include:

1. Support all detection models (Closed-Set) included in MMDet, such as Faster R-CNN and DINO, by using SAM for automatic detection and instance segmentation annotation.
2. Support Open-Vocabulary detection models, such as Detic, by using SAM for automatic detection and instance segmentation annotation.
3. Support Grounding Object Detection models, such as Grounding DINO and GLIP, by using SAM for automatic detection and instance segmentation annotation.
4. All models support distributed detection and segmentation evaluation, and automatic COCO JSON export, making it easy for users to evaluate custom data.

Please see [README](mmdet_sam/README.md) for more information.

## ✨ mmrotate_sam

<div align=center>
<img src="https://user-images.githubusercontent.com/27466624/231659969-adf7dd4d-fcec-4677-9105-aa72b2ced00f.PNG"/>
</div>

We provide a set of applications based on MMRotate and SAM. The features include:

1. Support Zero-shot Oriented Object Detection with SAM.
2. Perform SAM-based Zero-shot Oriented Object Detection inference on a single image.

Please see [README](mmrotate_sam/README.md) for more information.

## ✨ mmpose_open_detection

<div align=center>
<img src="https://user-images.githubusercontent.com/27466624/231660029-03166059-e8cf-4b17-8aa5-b42f3a52f12a.PNG"/>
</div>

We provide a set of applications based on MMPose and open detection. The features include:

1. Support open detection and pose estimation model inference for a single image or a folder of images.
2. Will soon support inputting different text prompts to achieve pose detection for different object categories in an image.

Please see [README](mmpose_open_detection/README.md) for more information.

## ✨ mmtracking_open_detection

<div align=center>
<img src="https://user-images.githubusercontent.com/27466624/231666666-4f4c5696-df73-45cd-af04-758ea3806a82.png"/>
</div>

We provide an approach based on open object detection and utilizing motion information (Kalman filter) for multi-object tracking.

Please see [README](mmtracking_open_detection/README.md) for more information.
