roadtext_root = '/media/jiangqing/jqhard/RoadText_OCR/RoadText/'

roadtext_textdet_train = dict(
    type='OCRDataset',
    data_prefix=dict(img_path=f'{roadtext_root}/det_images/train'),
    ann_file=
    '/media/jiangqing/jqhard/RoadText_OCR/RoadText/annotation/det_annotations/train_sample5.json',
    filter_cfg=dict(filter_empty_gt=True, min_size=8),
    pipeline=None)

roadtext_textdet_test = dict(
    type='OCRDataset',
    data_prefix=dict(img_path=f'{roadtext_root}/det_images/val'),
    ann_file=
    '/media/jiangqing/jqhard/RoadText_OCR/RoadText/annotation/det_annotations/val.json',
    test_mode=True,
    pipeline=None)
