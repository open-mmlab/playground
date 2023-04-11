## Installation

```shell
pip install mmengine

cd mmsam
git clone git@github.com:facebookresearch/segment-anything.git
cd segment-anything; pip install -e .; cd ..

git clone git@github.com:IDEA-Research/GroundingDINO.git
cd GroundingDINO; pip install -e .; cd ..
```

## Demo

```shell
cd mmsam/mmdet_sam

python detector_sam_demo.py ../images/demo2.jpg ../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py ../models/groundingdino_swint_ogc.pth -t bear --sam-device cpu

python coco_style_eval.py /home/PJLAB/huanghaian/dataset/coco1 ../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py ../models/groundingdino_swint_ogc.pth -t coco_cls_name.txt --sam-device cpu
```
