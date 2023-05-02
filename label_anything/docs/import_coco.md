# Importing COCO data into Label Studio


This article will use the validation set of coco2017 as an example to demonstrate how to import a coco formatted dataset into Label Studio.

### JSON format conversion

This section requires converting a COCO-formatted annotation JSON file into a Label Studio standard formatted JSON file for importing it into a Label Studio project. The LABEL_ANYTHING project provides auxiliary scripts for format conversion, which currently only support detection box format conversion.

```shell
python tools/convert_to_ls_format.py --input-file instances_val2017.json \ 
                                    --output-file out.json  \  
                                    --image-root-url "/data/local-files/?d=coco2017/val2017" 
``` 
The meanings of each parameter are as follows:
`input-file`: The COCO format JSON file that needs to be converted.
`output-file`: Label studio format json file that needs to be saved.
`image-root-url`：Image prefix in Label Studio project. Specifically, this article uses local storage to save images. 

In Label Studio, the local storage path is `/data/local-files/?d=coco2017/val2017`. Specific settings can be found in the third section.
After the script conversion is complete, two files, `out.json` and `out.label_config.xml` will be generated in the target directory. `out.json` is the successfully converted annotation file, and `out.label_config.xml` is the project configuration file.


### Create a Label Studio project
Next, we need to create a label-studio project and use local storage to save image files. According to the .[Local-storage](https://labelstud.io/guide/storage.html#Local-storage) on the label-studio guide, we need to set two environment variables before starting the label-studio server:
```
LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/home/user/label-studio/datasets 
```
`LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT` is the root directory where images for data are saved.

Subsequently, run `label-studio start` to start the label-studio server.

Access the Label Studio server's user interface, create a user, and log in.
![image](https://user-images.githubusercontent.com/42299757/235576000-2d7e2a3e-dde8-4aca-83fd-d003f4ba0170.png)
Next, we create a project called coco2017_val.
![image](https://user-images.githubusercontent.com/42299757/235576168-1768b92c-1d6b-4ad1-8958-43abcf2231fa.png)
Select the custom-template in the bottom left corner of the labeling-setup.
![49b69c270d2e1cb7cd56039134ab3b5](https://user-images.githubusercontent.com/42299757/235576464-c4236a23-23f9-4e9c-ab11-8b8dbddb2797.png)
Copy the content of `out.label_config.xml` previously converted into the template.
![image](https://user-images.githubusercontent.com/42299757/235576648-3f763f39-986e-4a47-9276-4574642d59cd.png)
Import the converted `out.json` file.
![image](https://user-images.githubusercontent.com/42299757/235576793-9b01cc23-6bb7-4742-be9a-f1be25134060.png)
At this point, the annotation files have been imported into the project, but the images cannot yet be read.
![image](https://user-images.githubusercontent.com/42299757/235577852-8f8377da-12d4-4dcd-acad-0d46027a16ca.png)
At this point, we need to synchronize local image files with label-studio. Go to project->setting->Cloud Storage and select Add Source Storage. 
![image](https://user-images.githubusercontent.com/42299757/235577703-27d47f54-48be-4bf3-9155-4b85337d2302.png)

Among them: \
`Storage Tile`: The name of the storage can be customized. \
`Absolute lcoal path`: The absolute path of the folder where the image is located, should include `${LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT}` set earlier. \

Finally, click on "check connection" to test if the local path is working properly. Then click on "Add Storage" to add the local storage to the project.

At this point, if we reopen the project, we can see that all the images have been loaded and the annotations are correctly displayed.
![image](https://user-images.githubusercontent.com/42299757/235578802-c3b13152-76ea-4388-b3c9-0d3c9bee2c13.png)

### The correspondence between the image path and the path in the annotation file
Label-studio does not recommend using local paths to store data, and instead recommends using URLs to load images.\
If the image data is stored in the cloud, it can be accessed through a URL. In this case, the `--image-root-url` parameter in the conversion script can be directly set to the prefix of the image URL. \
If local paths are used to store data, the path that label-studio reads images from is `/data/local-files/?d=${path_relative_to_data_root}`, where `${path_relative_to_data_root}` is the relative path of the directory where the image is located to the environment variable `${LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT}` at the time the server starts. For example:

Absolute path of the image: `/home/user/label-studio/datasets/coco2017/val2017/***.jpg` \
LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT: `/home/user/label-studio/datasets` \
path_relative_to_data_root: `/data/local-files/?d=coco2017/val2017` 


