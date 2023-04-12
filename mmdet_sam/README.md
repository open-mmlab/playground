# MMDetection-SAM

-- Graphs here

The current research direction of general-purpose object detection is moving toward large multimodality models. In addition to image inputs, most recent research outcomes incorporate text modality to improve the performance. Once the text modality is added, some very good properties of generic detection algorithms start to emerge, such as:

1. a large amount of easily accessible text data can be leveraged for joint training.
2. easy implementation of open-set object detection, leading to genuinely universal detection tasks.
3. can be used in conjunction with the superb models in NLP to create some interesting and useful features.

Recently, Meta AI proposed [Segment Anything](https://github.com/facebookresearch/segment-anything) which claims to be able to segment any object. MMDet integrates many high-performance and easy-to-use detection models, so it can also try to do some interesting things based on the MMDet model and Segment Anything jointly.

From the current point of view, generic object detection can be divided into two main categories:

1. closed-set object detection Closed-Set Object Detection, which can only detect a fixed number of classes of objects that appear in the training set
2. open-set object detection, i.e., objects of categories outside the training set can be detected.

With the popularity of multimodal algorithms, open-category object detection has become a new research direction, in which there are three popular research directions:

1. Zero-Shot Object Detection, i.e., zero-sample object detection, which emphasizes that the test set category is not in the training set
2. Open-Vocabulary Object Detection, i.e., developing vocabulary object detection, which detects all objects given pictures and category vocabularies
3. Grounding Object Detection, i.e., given a picture and a text description, predicts the location of the object mentioned in the text in the image.

The three directions are not completely distinguishable but just different in common parlance. Based on the above description, in conjunction with Segment Anything, we provide inference and evaluation scripts for multiple models in tandem. Specifically, we include the following features:

1. Support classical detection models (Closed-Set) of MMDet models, typically such as Faster R-CNN and DINO and other tandem SAM models for automatic detection and instance segmentation annotation
2. Support Open-Vocabulary detection model, typically such as Detic tandem SAM model for automatic detection and instance segmentation annotation
3. Support Grounding Object Detection models, typically Grounding DINO and GLIP tandem SAM models for automatic detection and instance segmentation annotation
4. All models support distributed detection and segmentation evaluation and automatic COCO JSON export for user-friendly custom data assessment

## Files Introduction

## Setup Base Development Environment

## Features

### 1 Open-Vocabulary + SAM

#### Dependencies Installation

#### Demo

### 2 MMdet models + SAM

#### Dependencies Installation

#### Demonstration

### 3 Grounding models + SAM

#### Dependencies Installation

#### Demonstration

### 4 COCO JSON evaluation
