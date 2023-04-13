dstext_textrecog_data_root = '/media/jiangqing/jqhard/DSText'
dstext_textrecog_train = dict(
    type='OCRDataset',
    data_root='/media/jiangqing/jqhard/DSText',
    data_prefix=dict(img_path='recog_images/train'),
    ann_file=
    '/media/jiangqing/jqhard/DSText/annotation/recog_annotations/train.json',
    pipeline=None)
dstext_textrecog_test = dict(
    type='OCRDataset',
    ann_file='work_dirs/LAION400M/part-00001/laion400m_pseudo_recog.json',
    test_mode=True,
    pipeline=None)
