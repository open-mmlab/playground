<div align=center>
<img src="resources/playground-banner.png"/>
</div>

<div align="center">

English | [简体中文](README_zh-CN.md)

</div>

🥳 🚀 **Welcome to <span style="color: blue"> *OpenMMLab Playground* </span>, an open-source initiative dedicated to gathering and showcasing amazing projects built with OpenMMLab. Our goal is to provide a central hub for the community to share their innovative solutions and explore the edge of AI technologies.**

🥳 🚀 **[OpenMMLab](https://github.com/open-mmlab) builds the most influential open-source computer vision algorithm system in the deep learning era, which provides high-performance and out-of-the-box algorithms for detection, segmentation, classification, pose estimation, video understanding, and AIGC. We believe that equipped with OpenMMLab, everyone can build exciting AI-empowered applications and push the limits of what's possible. All you need is a touch of creativity and a willingness to take action.**

🥳 🚀 **Join the <span style="color: blue"> *OpenMMLab Playground* </span> now and enjoy the power of AI!**

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
</div>
<br>

______________________________________________________________________

# Project List

|                                              |                                                                     Demo                                                                     |                                                                                     Description                                                                                     |
| :------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|           [MMDet-SAM](#-mmdet-sam)           | <img src="https://user-images.githubusercontent.com/17425982/231419108-bc5ef1ed-cb0b-496a-a19e-9b3b55479426.png" width="70%" height="20%"/>  | Explore a new way of instance segmentation by combining SAM (Segment Anything Model) with Closed-Set Object Detection, Open-Vocabulary Object Detection, Grounding Object Detection |
|        [MMRotate-SAM](#-mmrotate-sam)        | <img src="https://user-images.githubusercontent.com/79644233/231568599-58694ec9-a3b1-44a4-833f-74cfb4d4ca45.png" width="70%" height="20%"/>  |       Join SAM and weakly supervised horizontal box detection to achieve rotated box detection, and say goodbye to the tedious task of annotating rotated boxes from now on!        |
| [Open-Pose-Detection](#-open-pose-detection) |  <img src="https://user-images.githubusercontent.com/8425513/231439110-c0e7d6f8-5692-4bcb-b6cf-c3c243a990a5.jpg" width="70%" height="20%"/>  |         Integrate open object detection and various pose estimation algorithms to achieve "Pose All Things" - the ability to estimate the pose of anything and everything!          |
|       [Open-Tracking](#-open-tracking)       |                       <img src="https://github.com/zwhus/pictures/raw/main/demo%2B(1).gif" width="70%" height="20%" />                       |                                                Track and segment open categories in videos by marrying open object dtection and MOT.                                                |
|           [MMOCR-SAM](#-mmocr-sam)           | <img src="https://user-images.githubusercontent.com/65173622/231919274-a7ebc63f-8665-4324-89bf-f685e3b5161c.jpg" width="70%" height="20%" /> |   A solution of Text Detection/Recognition + SAM that segments every text character, with striking text removal and text inpainting demos driven by diffusion models and Gradio!    |
|      [MMEditing-SAM](#-mmediting-sam)      | <img src="https://user-images.githubusercontent.com/12782558/232716961-54b7e634-8f89-4a38-9353-4c962f9ce0cf.gif" width="70%" height="20%" /> |                                                  Join SAM and image generation to create awesome images and edit any part of them.                                                  |
| [Label-Studio-SAM](#-label-studio-sam) | <img src="https://user-images.githubusercontent.com/25839884/233835223-16abc0cb-09f0-407d-8be0-33e14cd86e1b.gif" width="70%" height="20%" /> | Combining Label-Studio and SAM to achieve semi-automated annotation. |






# Gallery

## ✨ MMDet-SAM

<div align=center>
<img src="https://user-images.githubusercontent.com/27466624/231659917-e3069822-2193-4261-b216-5f53baa64b53.PNG"/>
</div>

We provide a set of applications based on MMDet and SAM. The features include:

1. Support all detection models (Closed-Set) included in MMDet, such as Faster R-CNN and DINO, by using SAM for automatic detection and instance segmentation annotation.
2. Support Open-Vocabulary detection models, such as Detic, by using SAM for automatic detection and instance segmentation annotation.
3. Support Grounding Object Detection models, such as Grounding DINO and GLIP, by using SAM for automatic detection and instance segmentation annotation.
4. All models support distributed detection and segmentation evaluation, and automatic COCO JSON export, making it easy for users to evaluate custom data.

Please see [README](mmdet_sam/README.md) for more information.

## ✨ MMRotate-SAM

<div align=center>
<img src="https://user-images.githubusercontent.com/27466624/231659969-adf7dd4d-fcec-4677-9105-aa72b2ced00f.PNG"/>
</div>

We provide a set of applications based on MMRotate and SAM. The features include:

1. Support Zero-shot Oriented Object Detection with SAM.
2. Perform SAM-based Zero-shot Oriented Object Detection inference on a single image.

Please see [README](mmrotate_sam/README.md) for more information.

## ✨ Open-Pose-Detection

<div align=center>
<img src="https://user-images.githubusercontent.com/27466624/231660029-03166059-e8cf-4b17-8aa5-b42f3a52f12a.PNG"/>
</div>

We provide a set of applications based on MMPose and open detection. The features include:

1. Support open detection and pose estimation model inference for a single image or a folder of images.
2. Will soon support inputting different text prompts to achieve pose detection for different object categories in an image.

Please see [README](mmpose_open_detection/README.md) for more information.

## ✨ Open-Tracking

<div align=center>
<img src="https://user-images.githubusercontent.com/27466624/231666666-4f4c5696-df73-45cd-af04-758ea3806a82.png"/>
</div>

We provide an approach based on open object detection and utilizing motion information (Kalman filter) for multi-object tracking.

Please see [README](mmtracking_open_detection/README.md) for more information.

## ✨ MMOCR-SAM

<div align=center>
<img src="https://user-images.githubusercontent.com/65173622/231803460-495cf11f-8e2e-4c95-aa48-b163fc7fbbab.png"/>
</div>

The project is migrated from [OCR-SAM](https://github.com/yeungchenwa/OCR-SAM), which combines MMOCR with Segment Anything. We provide a set of applications based on MMOCR and SAM. The features include:

1. Support End-to-End Text Detection and Recognition, with the ability to segment every text character.
2. Striking text removal and text inpainting WebUI demos driven by diffusion models and Gradio.

Please see [README](mmocr_sam/README.md) for more information.

## ✨ MMEditing-SAM

<div align=center>
<img src="https://user-images.githubusercontent.com/12782558/232700025-a7bfe119-9eb5-46d2-b57c-ba7dc8c40d83.png"/>
</div>

We provide a set of applications based on MMEditing and SAM. The features include:

1. Generate images with MMEditing interface.
2. Combine the masks generated by SAM with the image editing capabilities of MMEditing to create new pictures.

Please see [README](mmediting_sam/README.md) for more information.

## ✨ Label-Studio-SAM

![](https://user-images.githubusercontent.com/25839884/233835223-16abc0cb-09f0-407d-8be0-33e14cd86e1b.gif)


The solution provides an integration of SAM with Label Studio. The specific features include:

1. Point2Label: Supports triggering SAM in Label-Studio to generate object masks and axis-aligned bounding box annotations by clicking a point within the object's area.
2. Bbox2Label: Supports triggering SAM in Label-Studio to generate object masks and axis-aligned bounding box annotations by annotating the object's bounding box.
3. Refine: Supports refining the annotations generated by SAM within Label-Studio.

详情见 [README](./label_anything/readme.md)。

