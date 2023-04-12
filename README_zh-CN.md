# Collection of Awesome OpenMMLab Applications

-- 这里放一张logo

[English](README.md) | 简体中文

本工程用于收集 OpenMMLab 相关的有趣应用并且将不断更新。我们非常欢迎社区用户能参与进这些项目中来，任何和 OpenMMLab 相关的有趣应用或者 Demo 都欢迎来贡献。目前包括如下相关应用

- `mmdet_sam` 检测相关模型串联 sam
- `mmrotate_sam` 旋转目标检测模型串联 sam
- `mmtracking_open_detection` 基于开发集检测的目标跟踪
- `mmpose_open_detection` 基于开发集检测的姿态估计

-- 这里放一个总表

下面详细说明。

## mmdet_sam

提供了和 mmdet 相关的结合 sam 的应用。具体特性包括：

1. 支持 MMDet 中包括的所有检测模型 (Closed-Set)，典型的如 Faster R-CNN 和 DINO 等串联 SAM 模型进行自动检测和实例分割标注
2. 支持 Open-Vocabulary 检测模型，典型的如 Detic 串联 SAM 模型进行自动检测和实例分割标注
3. 支持 Grounding Object Detection 模型，典型的如 Grounding DINO 和 GLIP 串联 SAM 模型进行自动检测和实例分割标注
4. 所有模型均支持分布式检测和分割评估和自动 COCO JSON 导出，方便用户对自定义数据进行评估

-- 贴一张图片

详情见 [README](mmdet_sam/README_zh-CN.md)

## mmrotate_sam

详情见 [README](mmrotate_sam/README_zh-CN.md)

## mmtracking_open_detection

详情见 [README](mmtracking_open_detection/README_zh-CN.md)

## mmpose_open_detection

详情见 [README](mmpose_open_detection/README_zh-CN.md)

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
