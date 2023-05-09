cp /home/data/831/*.jpg  /project/train/src_repo/dataset/images
python /project/train/src_repo/convert_to_coco.py 
#执行YOLOV5训练脚本
python /project/train/src_repo/mmyolo/tools/train.py /project/train/src_repo/mmyolo/tools/rtmdet_tiny_syncbn_fast_8xb32-300e_coco.py 