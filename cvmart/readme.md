# 极市打榜-安全帽识别-新手任务（白银榜）

## 打榜链接：https://cvmart.net/topList/10163?tab=RealTime


## 操作流程

1、点击开发者工作台

<div align=center>
<img src="https://github.com/open-mmlab/playground/assets/105597268/97b88ef8-e4e8-4e31-a244-4e7eb82abafa"/>
</div>

2、创建实例

<div align=center>
<img src="https://github.com/open-mmlab/playground/assets/105597268/d312dee1-fc72-4f6b-b886-da73f86e8bf0"/>
</div>

3、选择训练套件

<div align=center>
<img src="https://github.com/open-mmlab/playground/assets/105597268/e1bee667-d801-433a-a301-4bd404bb97a9"/>
</div>

解释：由于没法通过git clone 下载相应的MMLAB的软件包，我们接下来采用极市平台上手动下载的方式。

<div align=center>
<img src="https://github.com/open-mmlab/playground/assets/105597268/405d49cd-fc05-4708-b5e0-1a5560956875"/>
</div>


4、下载open-mmlab/playground的软件包，拖进资产管理/我的文件里面

下载地址：https://github.com/open-mmlab/playground

<div align=center>
<img src="https://github.com/open-mmlab/playground/assets/105597268/76685f15-fe9b-4dff-9e95-99c5f09e1b24"/>
</div>

<div align=center>
<img src="https://github.com/open-mmlab/playground/assets/105597268/c94b552e-4deb-4edb-ad3f-2dd834da9b01"/>
</div>

<div align=center>
<img src="https://github.com/open-mmlab/playground/assets/105597268/bf42132c-c918-4cf5-aafd-4df1a4c12bd9"/>
</div>

5、复制 playground 软件包的地址，接着在VSCode终端，采用 <wget 地址> 的方式下载 playground 的软件包

<div align=center>
<img src="https://github.com/open-mmlab/playground/assets/105597268/0c5db527-0b17-4ccf-8fd2-8fad3bd805a8"/>
</div>


6、点击在线编码->选择VSCode->点击确定，进入 VSCode 编辑页面

<div align=center>
<img src="https://github.com/open-mmlab/playground/assets/105597268/6b2a5cad-0cfa-435c-a4ba-f7e9b89fb525"/>
</div>

7、打开VSCode终端，输入下面代码指令。
```linux
wget https://extremevision-js-userfile.oss-cn-hangzhou.aliyuncs.com/user-35679-files/9a3f2b18-a4e8-4470-8a13-3415f8bc3e41/playground-main.zip（该地址即为刚才复制的playground的软件包地址）
```
<div align=center>
<img src="https://github.com/open-mmlab/playground/assets/105597268/5b1ceddb-9ff3-496c-8365-82135f392fe8"/>
</div>

<div align=center>
<img src="https://github.com/open-mmlab/playground/assets/105597268/cd09957c-f99d-4166-becc-978d94bd2df3"/>
</div>

<div align=center>
<img src="https://github.com/open-mmlab/playground/assets/105597268/074fb109-0f73-45fd-adba-9c96a667c3f1"/>
</div>

8、同上面下载 playground 软件包操作一样，将 mmyolo 的软件包，也下载到极市的VSCode平台里面

下载地址：https://github.com/open-mmlab/mmyolo

<div align=center>
<img src="https://github.com/open-mmlab/playground/assets/105597268/82ef43f7-0578-4f4c-90f0-1536a5ac16e6"/>
</div>

<div align=center>
<img src="https://github.com/open-mmlab/playground/assets/105597268/fdaa05ca-68c7-404b-b0fb-2fd78b532f42"/>
</div>

```linux
wget https://extremevision-js-userfile.oss-cn-hangzhou.aliyuncs.com/user-35679-files/ca0f112a-9f5e-444c-97e7-433e7e2e3f56/mmyolo-main.zip（该地址即为刚才复制的 mmyolo 的软件包地址）
```

<div align=center>
<img src="https://github.com/open-mmlab/playground/assets/105597268/64cf3641-5085-45e8-aa2f-e403998be644"/>
</div>

9、同上面下载软件包操作一样，将预训练权重，也下载到极市的VSCode平台里面

下载地址:https://download.openmmlab.com/mmyolo/v0/rtmdet/rtmdet_tiny_syncbn_fast_8xb32-300e_coco/rtmdet_tiny_syncbn_fast_8xb32-300e_coco_20230102_140117-dbb1dc83.pth

<div align=center>
<img src="https://github.com/open-mmlab/playground/assets/105597268/7eddca8a-4125-4a0c-bded-939c9e0478bf"/>
</div>

<div align=center>
<img src="https://github.com/open-mmlab/playground/assets/105597268/01ed5db0-b419-4c81-b27d-e7c51bba023b"/>
</div>

注意，由于文件全名不能超过50个字符，请将预训练权重的名称改为：rtmdet_tiny_syncbn_fast_8xb32-300e_coco.pth，然后再上传文件。

<div align=center>
<img src="https://github.com/open-mmlab/playground/assets/105597268/95b3bd17-5de6-41da-b61f-f8bff5b3b78a"/>
</div>


```linux
wget https://extremevision-js-userfile.oss-cn-hangzhou.aliyuncs.com/user-35679-files/f05a6660-d240-4a28-85fc-eef11d374038/rtmdet_tiny_syncbn_fast_8xb32-300e_coco.pth（该地址即为刚才复制的 mmyolo 的软件包地址）
```

<div align=center>
<img src="https://github.com/open-mmlab/playground/assets/105597268/a59da755-614f-4b0c-9fa2-627d12ca2a2a"/>
</div>

10、输入下面指令，将 playground-main 和 mmyolo-main 软件包解压，重命名 playground-main 文件为 playground ,重命名 mmyolo-main 文件为 mmyolo 并且复制放到 train/src_repo 路径下面。

```linux
unzip playground-main.zip
unzip mmyolo-main.zip
mv playground-main playground
mv mmyolo-main mmyolo
cp -r playground train/src_repo
cp -r mmyolo train/src_repo 
```

11、执行下面指令，创建所需文件夹，并将 playground 文件里面的 run.sh ji.py convert_to_coco.py 以及预训练权重复制相应的位置。

```linux
mkdir /project/ev_sdk/src
mkdir /project/train/src_repo/dataset
mkdir /project/train/src_repo/dataset/images
mkdir /project/train/src_repo/coco_annotations
touch /project/train/src_repo/coco_annotations/instances_train2014.json
mkdir /project/train/src_repo/mmyolo/tools/pth
cp -r train/src_repo/playground/cvmart/Helmet_identification_10163/ji.py /project/ev_sdk/src
cp -r train/src_repo/playground/cvmart/Helmet_identification_10163/run.sh /project/train/src_repo
cp -r train/src_repo/playground/cvmart/Helmet_identification_10163/convert_to_coco.py /project/train/src_repo
cp -r train/src_repo/playground/cvmart/Helmet_identification_10163/rtmdet_tiny_syncbn_fast_8xb32-300e_coco.py /project/train/src_repo/mmyolo/tools
cp -r rtmdet_tiny_syncbn_fast_8xb32-300e_coco.pth /project/train/src_repo/mmyolo/tools/pth/
```
12、安装所需要的安装包

```linux
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -U openmim
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple xml
mim install -i https://pypi.tuna.tsinghua.edu.cn/simple mmengine
mim install -i https://pypi.tuna.tsinghua.edu.cn/simple mmyolo
mim install -i https://pypi.tuna.tsinghua.edu.cn/simple mmcv
```

13、首先在VSCode终端执行一下训练程序，验证是否能跑起来

```linux
bash /project/train/src_repo/run.sh
```

如果运行之后，如下图效果所示，则配置成功

<div align=center>
<img src="https://github.com/open-mmlab/playground/assets/105597268/7e53fe22-fbc0-4aa8-b3e1-be48a2616d97"/>
</div>

14、用极市的官方平台执行训练任务。

<div align=center>
<img src="https://github.com/open-mmlab/playground/assets/105597268/018436ef-43b0-452d-8447-d594a402614e"/>
</div>

执行之后，等待它训练完成

<div align=center>
<img src="https://github.com/open-mmlab/playground/assets/105597268/431e6145-1964-4896-a6fe-661912d3f4b4"/>
</div>

15、测试训练得到的模型，

训练完成之后，可以在我们的模型列表里面查看我们训练得到的模型。

<div align=center>
<img src="https://github.com/open-mmlab/playground/assets/105597268/c603b1bd-365b-4acd-8647-2df0a12f22bb"/>
</div>

点击模型测试，发起模型测试，选择我们要测试的模型对应的训练任务ID，找到我们想要测试的模型。

<div align=center>
<img src="https://github.com/open-mmlab/playground/assets/105597268/b5f5e8a9-22b7-4fef-aa1c-0c72b81bbbf6"/>
</div>

注意，我们的模型是放在 /project/train/models/train/exp/weights 路径下面的，请在该路径下寻找 pth 文件

<div align=center>
<img src="https://github.com/open-mmlab/playground/assets/105597268/2130f93f-70ea-46dc-8bd5-4488a4ad271c"/>
</div>

注意，如果你要测试的是 /project/train/models/train/exp/weights/epoch_320.pth 模型，那么请保证你的 /project/ev_sdk/src/ji.py 文件里面对应的路径也是 /project/train/models/train/exp/weights/epoch_320.pth。一般来说，你每次测试的模型的名称不同，你都需要将 /project/ev_sdk/src/ji.py 文件对应的路径修改成对应的路径。

<div align=center>
<img src="https://github.com/open-mmlab/playground/assets/105597268/7805ccd3-7072-4d07-94e0-152e7f3c10b7"/>
</div>

16、查看测试的结果

<div align=center>
<img src="https://github.com/open-mmlab/playground/assets/105597268/fbfd11fa-0de1-4abf-a734-7bc7bad98446"/>
</div>