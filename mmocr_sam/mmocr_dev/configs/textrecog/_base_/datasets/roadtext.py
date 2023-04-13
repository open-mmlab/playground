roadtext_textrecog_data_root = '../../datasets/RoadText/recog_images'

roadtext_textrecog_train = dict(
    type='OCRDataset',
    data_prefix=dict(img_path=f'{roadtext_textrecog_data_root}/train/'),
    ann_file='../../datasets/RoadText/annotation/recog_annotations/train.json',
    pipeline=None)

roadtext_textrecog_test = dict(
    type='OCRDataset',
    data_prefix=dict(img_path=f'{roadtext_textrecog_data_root}/val/'),
    ann_file='../../datasets/RoadText/annotation/recog_annotations/val.json',
    test_mode=True,
    pipeline=None)
