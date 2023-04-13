laion400m_root = '/media/jiangqing/jqhard/Projects/UniText/data/LAION400M/data/'
laion400m_textdet_train = dict(
    type='OCRDataset',
    data_prefix=dict(img_path='data/DSText/det_images/train'),
    ann_file='data/DSText/annotation/det_annotations/new_train.json',
    filter_cfg=dict(filter_empty_gt=True, min_size=2),
    pipeline=None)
laion400m_textdet_test = dict(
    type='OCRDataset',
    data_prefix=dict(
        img_path=
        '/media/jiangqing/jqnas/projects/TextCLIP/data/LAION400M/data/part-00004-reordered/imgs'
    ),
    ann_file='work_dirs/LAION400M/part-00004/laion400m_pseudo_det.json',
    test_mode=True,
    pipeline=None)
