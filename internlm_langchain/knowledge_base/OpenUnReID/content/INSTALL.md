## Installation

### Requirements

+ Linux (Windows is not officially supported)
+ Python 3.5+
+ PyTorch 1.1 or higher
+ CUDA 9.0 or higher

We have tested the following versions of OS and softwares:

+ OS: Ubuntu 16.04
+ Python: 3.6/3.7
+ PyTorch: 1.1/1.5/1.6
+ CUDA: 9.0/11.0

### Install OpenUnReID

**a.** Create a conda virtual environment and activate it.
```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab
```

**b.** Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,
```shell
conda install pytorch torchvision -c pytorch
```

**c.** Clone the this repository.
```shell
git clone https://github.com/open-mmlab/OpenUnReID.git
cd OpenUnReID
```

**d.** Install the dependent libraries.
```shell
pip install -r requirements.txt
```

**e.** Install `openunreid` library.
```shell
python setup.py develop
```


### Prepare datasets

It is recommended to symlink your dataset root to `OpenUnReID/datasets`. If your folder structure is different, you may need to change the corresponding paths (namely `DATA_ROOT`) in config files.

Download the datasets from
+ DukeMTMC-reID: [[Google Drive]](https://drive.google.com/file/d/1jjE85dRCMOgRtvJ5RQV9-Afs-2_5dY3O/view) [[Baidu Pan]](https://pan.baidu.com/share/init?surl=jS0XM7Var5nQGcbf9xUztw) (password: bhbh)
+ Market-1501-v15.09.15: [[Google Drive]](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view) [[Baidu Pan]](https://pan.baidu.com/s/1ntIi2Op)
+ subset1 (PersonX): [[Google Drive]](https://drive.google.com/file/d/1hiHoDt3u7_GfeICMdEBt2Of8vXr1RF-U/view)
+ MSMT17_V1: [[Home Page]](https://www.pkuvmc.com/dataset.html) (request link by email the holder)
+ VehicleID_V1.0: [[Home Page]](https://www.pkuml.org/resources/pku-vehicleid.html) (request link by email the holder)
+ AIC20_ReID_Simulation (VehicleX): [[Home Page]](https://www.aicitychallenge.org/2020-track2-download/) (request password by email the holder)
+ VeRi_with_plate: [[Home Page]](https://github.com/JDAI-CV/VeRidataset#2-download) (request link by email the holder)

Save them under
```shell
OpenUnReID
└── datasets
    ├── dukemtmcreid
    │   └── DukeMTMC-reID
    ├── market1501
    │   └── Market-1501-v15.09.15
    ├── msmt17
    │   └── MSMT17_V1
    ├── personx
    │   └── subset1
    ├── vehicleid
    │   └── VehicleID_V1.0
    ├── vehiclex
    │   └── AIC20_ReID_Simulation
    └── veri
        └── VeRi_with_plate
```

<!-- ### Prepare pre-trained weights

If you want to use [ResNet-IBN](https://arxiv.org/abs/1807.09441) as the backbone, which may perform better than plain ResNet, you need to download the ImageNet pre-trained weights manually from [resnet50_ibn_a](https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet50_ibn_a-d9d0bb7b.pth), [resnet101_ibn_a](https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet101_ibn_a-59ea0ac6.pth). And put them under the folder of `~/.cache/torch/checkpoints/` like
```shell
~/.cache/torch
└── checkpoints
    ├── resnet50_ibn_a-d9d0bb7b.pth # manually downloaded and saved here
    ├── resnet101_ibn_a-59ea0ac6 # manually downloaded and saved here
    └── resnet50-19c8e357.pth # automatically downloaded by python script
``` -->
