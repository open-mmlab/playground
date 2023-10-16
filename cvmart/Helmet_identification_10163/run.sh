rm -rf /project/train/models/train/exp/weights

mkdir /project/train/models/train/exp/weights
rm -rf /project/train/tensorboard

cp /home/data/831/*.jpg  /project/train/src_repo/dataset/images
python /project/train/src_repo/convert_to_coco.py 

python /project/train/src_repo/mmyolo/tools/train.py /project/train/src_repo/mmyolo/tools/rtmdet_tiny_syncbn_fast_8xb32-300e_coco.py 
