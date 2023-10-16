# 准备 ChairsSDHom 数据集

<!-- [DATASET] -->

```bibtex
@InProceedings{IMKDB17,
  author    = "E. Ilg and N. Mayer and T. Saikia and M. Keuper and A. Dosovitskiy and T. Brox",
  title     = "FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks",
  booktitle = "IEEE Conference on Computer Vision and Pattern Recognition (CVPR)",
  month     = "Jul",
  year      = "2017",
  url       = "http://lmb.informatik.uni-freiburg.de//Publications/2017/IMKDB17"
}
```

```text
ChairsSDHom
|   |   ├── data
|   |   |    ├── train
|   |   |    |    |── flow
|   |   |    |    |      |── xxxxx.pfm
|   |   |    |    |── t0
|   |   |    |    |      |── xxxxx.png
|   |   |    |    |── t1
|   |   |    |    |      |── xxxxx.png
|   |   |    ├── test
|   |   |    |    |── flow
|   |   |    |    |      |── xxxxx.pfm
|   |   |    |    |── t0
|   |   |    |    |      |── xxxxx.png
|   |   |    |    |── t1
|   |   |    |    |      |── xxxxx.png
```

从数据集[官网](https://lmb.informatik.uni-freiburg.de/data/FlowNet2/ChairsSDHom/ChairsSDHom.tar.gz)下载文件压缩包后，解压文件并对照上方目录检查解压后的数据集目录。
