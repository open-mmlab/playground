# OpenMMLab PlayGround: Point2Label - Semi-Automated Annotation with Label-Studio and SAM

This article will introduce a semi-automated annotation approach that combines Label-Studio and SAM (Segment Anything), allowing users to obtain object masks and bounding box annotations by simply clicking a point within the object's area. Community users can learn from this method to improve their data annotation efficiency.

<br>

<div align=center>
    <img src="https://user-images.githubusercontent.com/25839884/233835223-16abc0cb-09f0-407d-8be0-33e14cd86e1b.gif" width="80%">
</div>

<br>

- SAM (Segment Anything) is a segmentation model launched by Meta AI, designed to segment everything.
- [Label Studio](https://github.com/heartexlabs/label-studio) is an excellent annotation software, covering dataset annotation functions in areas such as image classification, object detection, and segmentation.

This article will use images from the [Cat Dataset](https://download.openmmlab.com/mmyolo/data/cat_dataset.zip) for semi-automated annotation.

## Environment Setup

Create a virtual environment:

```shell
conda create -n rtmdet-sam python=3.9 -y
conda activate rtmdet-sam
```

Clone OpenMMLab PlayGround

```shell
git clone https://github.com/open-mmlab/playground
```

Install PyTorch

```shell
# Linux and Windows CUDA 11.3
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html


# Linux and Windows CPU only
pip install torch==1.10.1+cpu torchvision==0.11.2+cpu torchaudio==0.10.1 -f https://download.pytorch.org/whl/cpu/torch_stable.html

# OSX
pip install torch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1

```

Install SAM and download the pre-trained model:

```shell
cd path/to/playground/label_anything
pip install opencv-python pycocotools matplotlib onnxruntime onnx
pip install git+https://github.com/facebookresearch/segment-anything.git
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# For better segmentation results, use the sam_vit_h_4b8939.pth weights
# wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
# wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```


Install Label-Studio 和 label-studio-ml-backend

```shell
# sudo apt install libpq-dev python3-dev # Note: If using Label Studio 1.7.2 version, you need to install libpq-dev and python3-dev dependencies.

# Installing label-studio may take some time. If you cannot find the version, please use the official source.
pip install label-studio==1.7.3
pip install label-studio-ml==1.0.9
```

## Start the service:

Start the SAM backend inference service:

```shell
cd path/to/playground/label_anything

label-studio-ml start sam --port 8003 --with \
  sam_config=vit_b \
  sam_checkpoint_file=./sam_vit_b_01ec64.pth \
  out_mask=True \
  out_bbox=True \
  device=cuda:0
# device=cuda:0 is for using GPU inference. If you want to use CPU inference, replace cuda:0 with cpu.
# out_poly=True returns the annotation of the bounding polygon.
```

![image](https://user-images.githubusercontent.com/25839884/233821553-0030945a-8d83-4416-8edd-373ae9203a63.png)


At this point, the SAM backend inference service has started. Next, you can configure the http://localhost:8003 backend inference service in the Label-Studio Web system.

Now start the Label-Studio web service:

```shell
# If the inference backend being used is SAM's vit-h, due to the long model loading time, the following environment variable needs to be set.
# export ML_TIMEOUT_SETUP=40
label-studio start
```

![](https://cdn.vansin.top/picgo20230330132913.png)

Open your browser and visit [http://localhost:8080/](http://localhost:8080/) to see the Label-Studio interface.

![](https://cdn.vansin.top/picgo20230330133118.png)

We will register a user and then create an OpenMMLabPlayGround project.

![](https://cdn.vansin.top/picgo20230330133333.png)

We will download the example Meow Meow images using the method below, click on Data Import to import the cat images that need to be annotated, and then click Save to create the project.

```shell
cd path/to/playground/label_anything
mkdir data && cd data

wget https://download.openmmlab.com/mmyolo/data/cat_dataset.zip && unzip cat_dataset.zip
```

![](https://cdn.vansin.top/picgo20230330133628.png)

![](https://cdn.vansin.top/picgo20230330133715.png)


Configure Label-Studio keypoint, Mask, and other annotations in Settings/Labeling Interface.

```xml
<View>
  <Image name="image" value="$image" zoom="true"/>
  <KeyPointLabels name="KeyPointLabels" toName="image">
    <Label value="cat" smart="true" background="#e51515" showInline="true"/>
    <Label value="person" smart="true" background="#412cdd" showInline="true"/>
  </KeyPointLabels>
  <RectangleLabels name="RectangleLabels" toName="image">
  	<Label value="cat" background="#FF0000"/>
  	<Label value="person" background="#0d14d3"/>
  </RectangleLabels>
  <PolygonLabels name="PolygonLabels" toName="image">
  	<Label value="cat" background="#FF0000"/>
  	<Label value="person" background="#0d14d3"/>
  </PolygonLabels>
  <BrushLabels name="BrushLabels" toName="image">
  	<Label value="cat" background="#FF0000"/>
  	<Label value="person" background="#0d14d3"/>
  </BrushLabels>
</View>
```
In the above XML, we have configured the annotations, where KeyPointLabels are for keypoint annotations, BrushLabels are for Mask annotations, PolygonLabels are for bounding polygon annotations, and RectangleLabels are for rectangle annotations. 

This example uses two categories, cat and person. If community users want to add more categories, they need to add the corresponding categories in KeyPointLabels, BrushLabels, PolygonLabels, and RectangleLabels respectively.

Next, copy and add the above XML to Label-Studio, and then click Save.

![image](https://user-images.githubusercontent.com/25839884/233832662-02f856e5-48e7-4200-9011-17693fc2e916.png)


After that, go to Settings and click Add Model to add the OpenMMLabPlayGround backend inference service. Set the URL for the SAM backend inference service, enable Use for interactive preannotations, and click Validate and Save.

![image](https://user-images.githubusercontent.com/25839884/233836727-568d56e3-3b32-4599-b0a8-c20f18479a6a.png)

If you see "Connected" as shown below, it means that the backend inference service has been successfully added.

![image](https://user-images.githubusercontent.com/25839884/233832884-1b282d1f-1f43-474b-b41d-de41ad248476.png)

## Start semi-automated annotation.

Click on Label to start annotating.

![image](https://user-images.githubusercontent.com/25839884/233833125-fd372b0d-5f3b-49f4-bcf9-e89971639fd5.png)

You need to turn on the Auto-Annotation switch, and it is recommended to check Auto accept annotation suggestions. Then, click on the Smart tool on the right side, switch to Point, and select the object label that needs to be annotated below. Here, we select cat.

![image](https://user-images.githubusercontent.com/25839884/233833200-a44c9c5f-66a8-491a-b268-ecfb6acd5284.png)

As shown in the following gif animation, you only need to click a point on the object, and the SAM algorithm can segment and detect the entire object.

![SAM8](https://user-images.githubusercontent.com/25839884/233835410-29896554-963a-42c3-a523-3b1226de59b6.gif)


After submitting all the images, click on export to export the annotated dataset in COCO format, which will generate a compressed file of the annotated dataset. Note: only the bounding box annotations are exported here. If you want to export the instance segmentation annotations, you need to set out_poly=True when starting the SAM backend service.

![image](https://user-images.githubusercontent.com/25839884/233835852-b2f56cf1-1608-44c8-aa2d-d876f58e61f3.png)

You can use VS Code to open the extracted folder and see the annotated dataset, which includes the images and the annotated JSON files.

![](https://cdn.vansin.top/picgo20230330140321.png)

With the semi-automated annotation function of Label-Studio, users can complete object segmentation and detection by simply clicking the mouse during the annotation process, greatly improving the efficiency of annotation.

Some of the code was borrowed from Pull Request ID 253 of label-studio-ml-backend. Thank you to the author for their contribution.