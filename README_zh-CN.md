<div align=center>
<img src="resources/playground-logo.png"/>
</div>

<div align="center">

[English](README.md) | 简体中文

</div>

本工程用于收集 OpenMMLab 相关的有趣应用并且将不断更新。我们非常欢迎社区用户能参与进这些项目中来，任何和 OpenMMLab 相关的有趣应用或者 Demo 都欢迎来贡献。总览图如下所示

<div align=center>
<img src="https://user-images.githubusercontent.com/27466624/231665783-4a97e86c-6f89-4d63-b828-e7c414d1ff2b.png"/>
</div>

目前包括如下相关应用

- `mmdet_sam` 检测相关模型串联 sam
- `mmrotate_sam` 旋转目标检测模型串联 sam
- `mmpose_open_detection` 基于开放检测的姿态估计
- `mmtracking_open_detection` 基于开放检测的目标跟踪

下面详细说明。

## mmdet_sam

<div align=center>
<img src="https://user-images.githubusercontent.com/27466624/231659917-e3069822-2193-4261-b216-5f53baa64b53.PNG"/>
</div>

提供了和 mmdet 相关的结合 sam 的应用。具体特性包括：

1. 支持 MMDet 中包括的所有检测模型 (Closed-Set)，典型的如 Faster R-CNN 和 DINO 等串联 SAM 模型进行自动检测和实例分割标注
2. 支持 Open-Vocabulary 检测模型，典型的如 Detic 串联 SAM 模型进行自动检测和实例分割标注
3. 支持 Grounding Object Detection 模型，典型的如 Grounding DINO 和 GLIP 串联 SAM 模型进行自动检测和实例分割标注
4. 所有模型均支持分布式检测和分割评估和自动 COCO JSON 导出，方便用户对自定义数据进行评估

详情见 [README](mmdet_sam/README_zh-CN.md)

## mmrotate_sam

<div align=center>
<img src="https://user-images.githubusercontent.com/27466624/231659969-adf7dd4d-fcec-4677-9105-aa72b2ced00f.PNG"/>
</div>

提供了和 mmrotate 相关的结合 sam 的应用。具体特性包括：

1. 支持 sam 的 Zero-shot Oriented Object Detection
2. 对单张图片进行 sam 的 Zero-shot Oriented Object Detection 推理

详情见 [README](mmrotate_sam/README_zh-CN.md)

## mmpose_open_detection

<div align=center>
<img src="https://user-images.githubusercontent.com/27466624/231660029-03166059-e8cf-4b17-8aa5-b42f3a52f12a.PNG"/>
</div>

提供了和 mmpose 相关的结合开放检测的应用。具体特性包括：

1. 支持单张图片或者文件夹的开放词汇检测和姿态估计模型推理
2. 即将支持输入不同的 text prompt 实现对图片中不同类别物体的姿态检测

详情见 [README](mmpose_open_detection/README_zh-CN.md)

## mmtracking_open_detection

<div align=center>
<img src="https://user-images.githubusercontent.com/27466624/231666666-4f4c5696-df73-45cd-af04-758ea3806a82.png"/>
</div>

提供了基于开放目标检测，并利用运动信息（卡尔曼滤波器）来进行多目标跟踪。

详情见 [README](mmtracking_open_detection/README_zh-CN.md)

## ❤️ 欢迎加入 OpenMMLab 社区

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
