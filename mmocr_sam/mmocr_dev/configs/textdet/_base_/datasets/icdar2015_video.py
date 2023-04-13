ic15_video_root = '/media/jiangqing/jqssd/ICDAR-2023/data/ICDAR15'

ic15_video_textdet_train = dict(
    type='OCRDataset',
    data_prefix=dict(img_path=f'{ic15_video_root}/det_images/train'),
    ann_file=
    '/media/jiangqing/jqssd/ICDAR-2023/data/ICDAR15/annotation/det_annotations/train_adjacent_sample_frames_2.json',
    filter_cfg=dict(filter_empty_gt=True, min_size=2),
    pipeline=None)

ic15_video_textdet_test = dict(
    type='OCRDataset',
    data_prefix=dict(img_path=f'{ic15_video_root}/det_images/test'),
    ann_file=
    '/media/jiangqing/jqssd/ICDAR-2023/data/ICDAR15/annotation/det_annotations/test.json',
    test_mode=True,
    pipeline=None)
