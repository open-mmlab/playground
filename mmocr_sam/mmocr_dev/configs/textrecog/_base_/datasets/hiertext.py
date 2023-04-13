hiertext_textrecog_data_root = '../data/hiertext'

hiertext_textrecog_train = dict(
    type='OCRDataset',
    data_root=hiertext_textrecog_data_root,
    data_prefix=dict(img_path='recog_images/'),
    ann_file='annotation/recog_annotations/train.json',
    pipeline=None)

# hiertext_textrecog_test = dict(
#     type='OCRDataset',
#     data_root=hiertext_textrecog_data_root,
#     data_prefix=dict(img_path='recog_images/'),
#     ann_file='annotation/recog_annotations/val.json',
#     test_mode=True,
#     pipeline=None)

hiertext_textrecog_test = dict(
    type='OCRDataset',
    ann_file=
    '/media/jiangqing/jqssd/ICDAR-2023/mmocr-dev-1.x@9b0f1da/work_dirs/dbnetpp_resnet50-dcnv2_fpnc_100e_hiertext_ms/stage_ms_85.78/recog_results.json',
    test_mode=True,
    pipeline=None)