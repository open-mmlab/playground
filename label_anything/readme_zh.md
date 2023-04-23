# OpenMMLab PlayGround：Label-Studio X SAM 半自动化标注之 Point2Mask（一）


OpenMMLab PlayGround：Label-Studio X SAM 半自动化标注是一个系列的专题，本节介绍的是结合 Label-Studio 和 SAM (Segment Anything) 通过点击物体的一个点就得到物体整个标注。
<br>

<div align=center>
    <img src="https://user-images.githubusercontent.com/25839884/233818746-542f2e07-47cb-46db-9638-5e09bc799d87.gif" width="80%">
</div>

<br>

标注数据是一个费时费力的任务，然后使用 Label-Studio 进行微调标注，社区用户可以参考此流程和方法，将其应用到其他领域。

- SAM (Segment Anything) 是 Fackbook 推出的分割一切的模型。
- [Label Studio](https://github.com/heartexlabs/label-studio) 是一款优秀的标注软件，覆盖图像分类、目标检测、分割等领域数据集标注的功能。


本文将使用[喵喵数据集](https://download.openmmlab.com/mmyolo/data/cat_dataset.zip)的图片，进行半自动化标注。

## 环境配置

首先需要创建一个虚拟环境，然后安装 PyTorch 和 SAM。
创建虚拟环境：

```shell
conda create -n rtmdet-sam python=3.9 -y
conda activate rtmdet-sam
```

克隆 OpenMMLab PlayGround

```shell
git clone https://github.com/open-mmlab/playground
```


安装 PyTorch

```shell

# Linux and Windows CUDA 11.3
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html


# Linux and Windows CPU only
pip install torch==1.10.1+cpu torchvision==0.11.2+cpu torchaudio==0.10.1 -f https://download.pytorch.org/whl/cpu/torch_stable.html

# OSX
pip install torch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1

```

安装 SAM 并下载预训练模型

```shell
cd path/to/playground/label_anything
pip install opencv-python pycocotools matplotlib onnxruntime onnx
pip install git+https://github.com/facebookresearch/segment-anything.git
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# 如果想要分割的效果好请使用 sam_vit_h_4b8939.pth 权重
# wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
# wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```


安装 Label-Studio 和 label-studio-ml-backend

```shell
# sudo apt install libpq-dev python3-dev # Note：如果使用 Label Studio 1.7.2 版本需要安装 `libpq-dev` 和 `python3-dev` 依赖。

# 安装 label-studio 需要一段时间,如果找不到版本请使用官方源
pip install label-studio==1.7.3
pip install label-studio-ml==1.0.9
```

## 启动服务

启动 SAM 后端推理服务：

```shell
cd path/to/playground/label_anything

label-studio-ml start sam --port 8003 --with \
sam_config=vit_b \
sam_checkpoint_file=./sam_vit_b_01ec64.pth \
device=cuda:0 \
# device=cuda:0 为使用 GPU 推理，如果使用 cpu 推理，将 cuda:0 替换为 cpu
```

![image](https://user-images.githubusercontent.com/25839884/233821553-0030945a-8d83-4416-8edd-373ae9203a63.png)


此时，RTMDet 后端推理服务已经启动，后续在 Label-Studio Web 系统中配置 http://localhost:8003 后端推理服务即可。

现在启动 Label-Studio 网页服务：

```shell
label-studio start
```

![](https://cdn.vansin.top/picgo20230330132913.png)

打开浏览器访问 [http://localhost:8080/](http://localhost:8080/) 即可看到 Label-Studio 的界面。

![](https://cdn.vansin.top/picgo20230330133118.png)

我们注册一个用户，然后创建一个 RTMDet-Semiautomatic-Label 项目。

![](https://cdn.vansin.top/picgo20230330133333.png)

我们通过下面的方式下载好示例的喵喵图片，点击 Data Import 导入需要标注的猫图片。

```shell
cd path/to/playground/label_anything
mkdir data && cd data

wget https://download.openmmlab.com/mmyolo/data/cat_dataset.zip && unzip cat_dataset.zip
```

![](https://cdn.vansin.top/picgo20230330133628.png)

![](https://cdn.vansin.top/picgo20230330133715.png)

然后选择 Object Detection With Bounding Boxes 模板

![](https://cdn.vansin.top/picgo20230330133807.png)


在`Settings/Labeling Interface` 中配置 Label-Studio 关键点和 Mask 标注

```shell
<View>
  <Image name="image" value="$image" zoom="true"/>
  <BrushLabels name="BrushLabels" toName="image">
  	<Label value="cat" background="#FF0000"/>
  	<Label value="person" background="#0d14d3"/>
  </BrushLabels>
  <KeyPointLabels name="KeyPointLabels" toName="image">
    <Label value="cat" smart="true" background="#e51515" showInline="true"/>
    <Label value="person" smart="true" background="#412cdd" showInline="true"/>
  </KeyPointLabels>
</View>
```

然后将上述类别复制添加到 Label-Studio，然后点击 Save。

![image](https://user-images.githubusercontent.com/25839884/233820036-d895e723-0694-449e-b6f2-723bdbde28fb.png)

然后在设置中点击 Add Model 添加 RTMDet 后端推理服务。

![](https://cdn.vansin.top/picgo20230330134320.png)

点击 Validate and Save，然后点击 Start Labeling。

![](https://cdn.vansin.top/picgo20230330134424.png)

看到如下 Connected 就说明后端推理服务添加成功。

![](https://cdn.vansin.top/picgo20230330134554.png)

## 开始半自动化标注

点击 Label 开始标注

![](https://cdn.vansin.top/picgo20230330134804.png)

我们可以看到 RTMDet 后端推理服务已经成功返回了预测结果并显示在图片上，我们可以发现这个喵喵预测的框有点大。

![](https://cdn.vansin.top/picgo20230403104419.png)

我们手工拖动框，修正一下框的位置，得到以下修正过后的标注，然后点击 Submit，本张图片就标注完毕了。

![](https://cdn.vansin.top/picgo/20230403105923.png)

我们 submit 完毕所有图片后，点击 exprot 导出 COCO 格式的数据集，就能把标注好的数据集的压缩包导出来了。

![](https://cdn.vansin.top/picgo20230330135921.png)

用 vscode 打开解压后的文件夹，可以看到标注好的数据集，包含了图片和 json 格式的标注文件。

![](https://cdn.vansin.top/picgo20230330140321.png)

到此半自动化标注就完成了，我们可以用这个数据集在 MMDetection 训练精度更高的模型了，训练出更好的模型，然后再用这个模型继续半自动化标注新采集的图片，这样就可以不断迭代，扩充高质量数据集，提高模型的精度。

## 结论

即将支持利用 RTMDet 生成水平边界框作为 SAM 的提示词，并使用 SAM 生成 Mask Demo。
