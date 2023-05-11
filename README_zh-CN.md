<div align=center>
<img src="resources/playground-banner.png"/>
</div>

<div align="center">

[English](README.md) | 简体中文

</div>

🥳 🚀 **欢迎来到 <span style="color: blue"> *OpenMMLab Playground* </span>, 一个用于收集和展示 OpenMMLab 相关前沿和有趣应用的项目，旨在为社区搭建分享创新技术方案、玩转 OpenMMLab 的平台。**

🥳 🚀 **AI 领域日新月异，[OpenMMLab](https://github.com/open-mmlab) 作为深度学习领域头部社区始终秉持着拥抱变化、拥抱社区的理念，致力于不断推动 AI 技术的发展和创新。 秉承着开放、透明、合作的原则，我们鼓励社区成员参与到项目中来。我们相信，基于 OpenMMLab 提供的丰富算法能力和强大技术社区，每一位开发者都可以参与到 AI 技术的边界探索和应用实践中来。**

🥳 🚀 **希望 <span style="color: blue"> *OpenMMLab Playground* </span> 可以成为广大社区成员的开源自留地，共同分享、碰撞灵感，AI 新乐园，有你也有我！**

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

# 更新

🥳 🚀 我们新增了一个基于 DetGPT 项目，其是一个无需训练的仿真版本，可以进行推理式目标检测。具体见 [DetGPT](det_gpt/README_zh-CN.md)


# 项目列表

|                                              |                                                                     示例                                                                     |                                     说明                                      |
|:--------------------------------------------:| :------------------------------------------------------------------------------------------------------------------------------------------: |:---------------------------------------------------------------------------:|
|           [MMDet-SAM](#-mmdet-sam)           | <img src="https://user-images.githubusercontent.com/17425982/231419108-bc5ef1ed-cb0b-496a-a19e-9b3b55479426.png" width="70%" height="20%"/>  |     目标检测检测相关模型 + SAM。将闭集目标检测、开放词汇目标检测、 Grounding 目标检测和 SAM 结合探索实例分割新玩法      |
|             [DetGPT](#-det-gpt)              | <img src="https://github.com/open-mmlab/playground/assets/17425982/c3145a82-7748-4a79-a187-bcb8d91f1dd3" width="70%" height="20%"/>  |     视觉语言多模态 + Grounding。将视觉语言多模态如 MiniGPT-4 和 Grounding 结合探索推理式目标检测新方向      |
|        [MMRotate-SAM](#-mmrotate-sam)        | <img src="https://user-images.githubusercontent.com/79644233/231568599-58694ec9-a3b1-44a4-833f-74cfb4d4ca45.png" width="70%" height="20%"/>  |           旋转框检测相关模型 + SAM。 将 SAM 和弱监督即水平框检测联合实现旋转框检测，从此省掉累人的旋转框标注           |
| [Open-Pose-Detection](#-open-pose-detection) |  <img src="https://user-images.githubusercontent.com/8425513/231439110-c0e7d6f8-5692-4bcb-b6cf-c3c243a990a5.jpg" width="70%" height="20%"/>  |               开放目标检测 + mmpose。探索开放目标检测和各类姿态估计算法结合实现万物皆可摆 Pose               |
|       [Open-Tracking](#-open-tracking)       |                       <img src="https://github.com/zwhus/pictures/raw/main/demo%2B(1).gif" width="70%" height="20%" />                       |             开放目标检测 + tracking。探索开放目标检测和视频任务相结合，轻松实现开放类别的视频跟踪和分割             |
|           [MMOCR-SAM](#-mmocr-sam)           | <img src="https://user-images.githubusercontent.com/65173622/231919274-a7ebc63f-8665-4324-89bf-f685e3b5161c.jpg" width="70%" height="20%" /> | 端到端文字检测识别 + SAM，将每一个字符都进行分割。使用基于 Gradio 的 Web UI 探索有趣的 OCR 下游任务，包括文本擦除、文本编辑 |
|       [MMEditing-SAM](#-mmediting-sam)       | <img src="https://user-images.githubusercontent.com/12782558/232716961-54b7e634-8f89-4a38-9353-4c962f9ce0cf.gif" width="70%" height="20%" /> |                       将 SAM 和图像生成结合起来从而对图像进行任意位置的编辑修改                       |
|    [Label-Studio-SAM](#-label-studio-sam)    | <img src="https://user-images.githubusercontent.com/25839884/233835223-16abc0cb-09f0-407d-8be0-33e14cd86e1b.gif" width="70%" height="20%" /> |                       将 Label-Studio 和 SAM 结合实现半自动化标注                       |


下面详细说明。

# 项目展示

## ✨ MMDet-SAM

<div align=center>
<img src="https://user-images.githubusercontent.com/27466624/231659917-e3069822-2193-4261-b216-5f53baa64b53.PNG"/>
</div>

提供了和 MMDet 相关的结合 SAM 的应用。具体特性包括：

1. 支持 MMDet 中包括的所有检测模型 (Closed-Set)，典型的如 Faster R-CNN 和 DINO 等串联 SAM 模型进行自动检测和实例分割标注
2. 支持 Open-Vocabulary 检测模型，典型的如 Detic 串联 SAM 模型进行自动检测和实例分割标注
3. 支持 Grounding Object Detection 模型，典型的如 Grounding DINO 和 GLIP 串联 SAM 模型进行自动检测和实例分割标注
4. 所有模型均支持分布式检测和分割评估和自动 COCO JSON 导出，方便用户对自定义数据进行评估

详情见 [README](mmdet_sam/README_zh-CN.md)

## ✨ Det-GPT

<div align=center>
<img src="https://github.com/open-mmlab/playground/assets/17425982/c3145a82-7748-4a79-a187-bcb8d91f1dd3"/>
</div>

基于 DetGPT 原理，提供了一个无需训练的仿真版本：

1. 提供了 DetGPT 原理说明
2. 基于 MiniGPT-4 简单探索了使用无需专门微调的多模态算法进行推理式目标检测的可能性
3. 基于 ChatGPT3 实现了推理式目标检测，Grounding 检测算法支持 Grounding DINO 和 GLIP

详情见 [README](det_gpt/README_zh-CN.md)

## ✨ MMRotate-SAM

<div align=center>
<img src="https://user-images.githubusercontent.com/27466624/231659969-adf7dd4d-fcec-4677-9105-aa72b2ced00f.PNG"/>
</div>

提供了和 MMRotate 相关的结合 SAM 的应用。具体特性包括：

1. 支持 SAM 的 Zero-shot Oriented Object Detection
2. 对单张图片进行 SAM 的 Zero-shot Oriented Object Detection 推理

详情见 [README](mmrotate_sam/README_zh-CN.md)

## ✨ Open-Pose-Detection

<div align=center>
<img src="https://user-images.githubusercontent.com/27466624/231660029-03166059-e8cf-4b17-8aa5-b42f3a52f12a.PNG"/>
</div>

提供了和 MMPose 相关的结合开放检测的应用。具体特性包括：

1. 支持单张图片或者文件夹的开放检测和姿态估计模型推理
2. 即将支持输入不同的 text prompt 实现对图片中不同类别物体的姿态检测

详情见 [README](mmpose_open_detection/README_zh-CN.md)

## ✨ Open-Tracking

<div align=center>
<img src="https://user-images.githubusercontent.com/27466624/231666666-4f4c5696-df73-45cd-af04-758ea3806a82.png"/>
</div>

提供了基于开放目标检测，并利用运动信息（卡尔曼滤波器）来进行多目标跟踪。

详情见 [README](mmtracking_open_detection/README_zh-CN.md)

## ✨ MMOCR-SAM

<div align=center>
<img src="https://user-images.githubusercontent.com/65173622/231803460-495cf11f-8e2e-4c95-aa48-b163fc7fbbab.png"/>
</div>

该仓库搬运自 [OCR-SAM](https://github.com/yeungchenwa/OCR-SAM)。我们将 MMOCR 与 SAM 结合，并提供了以下功能。

1. 支持端到端的文字检测识别，并可以将每一个文本字符都进行分割。
2. 提供基于 diffusion 模型以及 Gradio 的 Web UI，可以探索有趣的 OCR 下游任务，包括文本擦除、文本编辑等。

详情见 [README](mmocr_sam/README_zh-CN.md)。

## ✨ MMEditing-SAM

<div align=center>
<img src="https://user-images.githubusercontent.com/12782558/232700025-a7bfe119-9eb5-46d2-b57c-ba7dc8c40d83.png"/>
</div>

提供了和 MMEditing 相关的结合 SAM 的应用。具体特性包括：

1. 使用 MMEDiting 的接口生成图片。
2. 结合 SAM 生成的 mask 与 MMEditing 的图像编辑能力创造新的图片。

详情见 [README](mmediting_sam/README_zh-CN.md)。


## ✨ Label-Studio-SAM

![](https://user-images.githubusercontent.com/25839884/233835223-16abc0cb-09f0-407d-8be0-33e14cd86e1b.gif)

提供了和 Label Studio 相关的结合 SAM 的应用。具体特性包括：

1. Point2Label：支持在 Label-Studio 通过点击物体区域的一点来触发 SAM 生成物体的掩码和水平边界框标注生成。
2. Bbox2Label：支持在 Label-Studio 通过标注物体的边界框来触发 SAM 生成物体掩码和水平边界框标注生成。
3. Refine: 支持在 Label-Studio 上对 SAM 生成的标注进行修正。


详情见 [README](./label_anything/readme_zh.md)。

# ❤️ 欢迎加入 OpenMMLab 社区

扫描下方的二维码可关注 OpenMMLab 团队的 [知乎官方账号](https://www.zhihu.com/people/openmmlab)，加入 OpenMMLab 团队的 [官方交流 QQ 群](https://jq.qq.com/?_wv=1027&k=aCvMxdr3)

<div align="center">
<img src="resources/zhihu_qrcode.jpg" height="400" />  <img src="resources/qq_group_qrcode.jpg" height="400" />
</div>

我们会在 OpenMMLab 社区为大家

- 📢 分享 AI 框架的前沿核心技术
- 💻 解读 PyTorch 常用模块源码
- 📰 发布 OpenMMLab 的相关新闻
- 🚀 介绍 OpenMMLab 开发的前沿算法
- 🏃 获取更高效的问题答疑和意见反馈
- 🔥 提供与各行各业开发者充分交流的平台

干货满满 📘，等你来撩 💗，OpenMMLab 社区期待您的加入 👬
