<div align=center>
<img src="resources/playground-logo.png"/>
</div>

<div align="center">

[English](README.md) | 简体中文

</div>

🥳 🚀 **AI 领域日新月异，OpenMMLab 作为深度学习领域头部社区始终秉持着拥抱变化、拥抱社区的理念，致力于不断推动 AI 领域的发展和创新。 秉承着开放、透明、合作的原则，我们鼓励社区成员参与到项目中来，共同探索 AI 边界。本项目将用于收集 OpenMMLab 相关的有趣&前沿应用并不断更新。**

🥳 🚀 **希望 Playground 可以成为广大社区成员的开源自留地，共同分享、碰撞灵感，AI 新乐园，有你也有我！**

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
目前包括如下相关应用

|                                                                        |                                                                   效果图                                                                    |                                                      说明                                                      |
| :--------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------: |
|                 [mmdet_sam](mmdet_sam/README_zh-CN.md)                 | <img src="https://user-images.githubusercontent.com/17425982/231419108-bc5ef1ed-cb0b-496a-a19e-9b3b55479426.png" width="70%" height="20%"/> | 目标检测检测相关模型 + SAM。将闭集目标检测、开放词汇目标检测、 Grounding 目标检测和 SAM 结合探索实例分割新玩法 |
|              [mmrotate_sam](mmrotate_sam/README_zh-CN.md)              | <img src="https://user-images.githubusercontent.com/79644233/231568599-58694ec9-a3b1-44a4-833f-74cfb4d4ca45.png" width="70%" height="20%"/> |       旋转框检测相关模型 + SAM。 将 SAM 和弱监督即水平框检测联合实现旋转框检测，从此省掉累人的旋转框标注       |
|     [mmpose_open_detection](mmpose_open_detection/README_zh-CN.md)     | <img src="https://user-images.githubusercontent.com/8425513/231439110-c0e7d6f8-5692-4bcb-b6cf-c3c243a990a5.jpg" width="70%" height="20%"/>  |                开放目标检测 + mmpose。探索开放目标检测和各类姿态估计算法结合实现万物皆可摆 Pose                |
| [mmtracking_open_detection](mmtracking_open_detection/README_zh-CN.md) |                      <img src="https://github.com/zwhus/pictures/raw/main/demo%2B(1).gif" width="70%" height="20%" />                       |          开放目标检测 + tracking。探索开放目标检测和视频任务相结合，轻松实现开放类别的视频跟踪和分割           |

下面详细说明。

## ✨ mmdet_sam

<div align=center>
<img src="https://user-images.githubusercontent.com/27466624/231659917-e3069822-2193-4261-b216-5f53baa64b53.PNG"/>
</div>

提供了和 MMDet 相关的结合 SAM 的应用。具体特性包括：

1. 支持 MMDet 中包括的所有检测模型 (Closed-Set)，典型的如 Faster R-CNN 和 DINO 等串联 SAM 模型进行自动检测和实例分割标注
2. 支持 Open-Vocabulary 检测模型，典型的如 Detic 串联 SAM 模型进行自动检测和实例分割标注
3. 支持 Grounding Object Detection 模型，典型的如 Grounding DINO 和 GLIP 串联 SAM 模型进行自动检测和实例分割标注
4. 所有模型均支持分布式检测和分割评估和自动 COCO JSON 导出，方便用户对自定义数据进行评估

详情见 [README](mmdet_sam/README_zh-CN.md)

## ✨ mmrotate_sam

<div align=center>
<img src="https://user-images.githubusercontent.com/27466624/231659969-adf7dd4d-fcec-4677-9105-aa72b2ced00f.PNG"/>
</div>

提供了和 MMRotate 相关的结合 SAM 的应用。具体特性包括：

1. 支持 SAM 的 Zero-shot Oriented Object Detection
2. 对单张图片进行 SAM 的 Zero-shot Oriented Object Detection 推理

详情见 [README](mmrotate_sam/README_zh-CN.md)

## ✨ mmpose_open_detection

<div align=center>
<img src="https://user-images.githubusercontent.com/27466624/231660029-03166059-e8cf-4b17-8aa5-b42f3a52f12a.PNG"/>
</div>

提供了和 MMPose 相关的结合开放检测的应用。具体特性包括：

1. 支持单张图片或者文件夹的开放检测和姿态估计模型推理
2. 即将支持输入不同的 text prompt 实现对图片中不同类别物体的姿态检测

详情见 [README](mmpose_open_detection/README_zh-CN.md)

## ✨ mmtracking_open_detection

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
