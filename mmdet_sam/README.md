## Installation

```shell
pip install mmengine

cd mmsam
git clone https://github.com/facebookresearch/segment-anything.git
cd segment-anything; pip install -e .; cd ..

git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO; pip install -e .; cd ..
```

## Demo

```shell
cd mmsam/mmdet_sam

python detector_sam_demo.py ../images/demo2.jpg ../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py ../models/groundingdino_swint_ogc.pth -t bear --sam-device cpu

python coco_style_eval.py /home/PJLAB/huanghaian/dataset/coco1 ../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py ../models/groundingdino_swint_ogc.pth -t coco_cls_name.txt --sam-device cpu

python coco_style_eval.py /home/huanghaian/coco20 ../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py ../models/groundingdino_swint_ogc.pth -t coco_cls_name.txt

bash ./dist_coco_style_eval.sh 8 /home/huanghaian/coco20 ../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py ../models/groundingdino_swint_ogc.pth -t coco_cls_name.txt
```
