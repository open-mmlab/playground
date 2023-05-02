# 导入COCO数据进入Label Studio


本文将以coco2017的验证集为例，介绍如何将coco格式的数据集导入Label Studio。

### json格式转换

本小节需要将coco格式的标注json文件转换为Label Studio标准格式的json文件以便将其导入进Label Studio项目中,LABEL_ANYTHING项目提供了辅助脚本进行格式转换，目前仅支持检测框格式转换。

```shell
python tools/convert_to_ls_format.py --input-file instances_val2017.json \ 
                                    --output-file out.json  \  
                                    --image-root-url "/data/local-files/?d=coco2017/val2017" 
``` 
其中各个参数的含义如下:
input-file：需要转换的coco格式json文件
output-file: 需要保存的Label Studio格式的json文件
image-root-url：Label Studio读取图片的路径前缀。本文采用本地存储来保存图片,在label-studio中本地储存的路径为/data/local-files/?d=coco2017/val2017 具体设置规则可以参见第三小节
脚本转换完成后会在目标目录下生成out.json和out.label_config.xml两个文件。out.json为转换成功的标注文件；out.label_config.xml为项目配置文件。


### 创建Label Studio项目
接下来需要创建label-studio项目，本文采用本地储存方式来保存图片文件。根据[Local-storage](https://labelstud.io/guide/storage.html#Local-storage)，在启动Label Studio服务器前需要设置两个环境变量:
```
LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/home/user/label-studio/datasets 
```
其中LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT为数据图片保存根目录。
随后运行
```
label-studio start
```
启动label-studio服务器

进入Label Studio服务器ui,创建用户并登录:
![image](https://user-images.githubusercontent.com/42299757/235576000-2d7e2a3e-dde8-4aca-83fd-d003f4ba0170.png)
随后我们创建一个名为coco2017_val的项目:
![image](https://user-images.githubusercontent.com/42299757/235576168-1768b92c-1d6b-4ad1-8958-43abcf2231fa.png)
在labeling-setup中选择左下角的custom-template:
![49b69c270d2e1cb7cd56039134ab3b5](https://user-images.githubusercontent.com/42299757/235576464-c4236a23-23f9-4e9c-ab11-8b8dbddb2797.png)
将之前转换出的out.label_config.xml中的内容复制到template中:
![image](https://user-images.githubusercontent.com/42299757/235576648-3f763f39-986e-4a47-9276-4574642d59cd.png)
导入转换好的out.json文件:
![image](https://user-images.githubusercontent.com/42299757/235576793-9b01cc23-6bb7-4742-be9a-f1be25134060.png)
此时可以看到标注文件已经导入进项目中了，但是图片还无法读取：
![image](https://user-images.githubusercontent.com/42299757/235577852-8f8377da-12d4-4dcd-acad-0d46027a16ca.png)
此时我们需要将本地的图片文件同步进label-studio中，在project->setting->Cloud Storage选择Add Source Storage，选择local files下拉菜单：
![image](https://user-images.githubusercontent.com/42299757/235577703-27d47f54-48be-4bf3-9155-4b85337d2302.png)

其中: \
Storage Tile: storage的名称，可以自定义 \
Absolute lcoal path: 图片所在文件夹的绝对路径，需要包含前面设置的${LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT} \
随后点击check connection可以测试本地路径是否可用,最后点击Add Storage可以将本地储存添加到项目中。

此时我们再打开项目可以发现，所有的图片都被读取出来了，并且标注也被正确显示出来了。
![image](https://user-images.githubusercontent.com/42299757/235578802-c3b13152-76ea-4388-b3c9-0d3c9bee2c13.png)

### 关于图片路径和标注文件中路径的对应
在Label Studio中并不推荐采用本地路径来储存数据，而推荐采用url来读取图片。\
如果图片数据储存在云端，通过url读取的话，转换脚本中--image-root-url参数直接设置为图片路径前缀即可。\
如果采用本地路径来储存数据，label-studio读取图片的路径会是在"/data/local-files/?d=${path_relative_to_data_root}",其中${path_relative_to_data_root}是图片所在目录相对于服务器启动时候的环境变量${LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT}的相对路径，例如: 

图片绝对路径:/home/user/label-studio/datasets/coco2017/val2017/***.jpg \
LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT:/home/user/label-studio/datasets \
则path_relative_to_data_root:/data/local-files/?d=coco2017/val2017 


