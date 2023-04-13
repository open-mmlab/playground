model = dict(
    type='DBNet',
    backbone=dict(
        type='mmcls.SwinTransformerV2',
        arch='base',
        img_size=256,  # TODO
        out_indices=(0, 1, 2, 3),
        window_size=[16, 16, 16, 8],
        drop_path_rate=0.2,
        pretrained_window_sizes=[12, 12, 12, 6]),
    neck=dict(
        type='FPNC',
        in_channels=[128, 256, 512, 1024],  # TODO
        lateral_channels=256,
        asf_cfg=dict(attention_type='ScaleChannelSpatial')),
    det_head=dict(
        type='DBHead',
        in_channels=256,
        module_loss=dict(type='DBModuleLoss'),
        postprocessor=dict(
            type='DBPostprocessor',
            text_repr_type='quad',
            mask_thr=0.5,
            min_text_score=0.5,
            min_text_width=15,
            epsilon_ratio=0.002)),
    data_preprocessor=dict(
        type='TextDetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32))

train_pipeline = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='LoadOCRAnnotations',
        with_bbox=True,
        with_polygon=True,
        with_label=True,
    ),
    dict(type='FixInvalidPolygon'),
    dict(
        type='TorchVisionWrapper',
        op='ColorJitter',
        brightness=32.0 / 255,
        saturation=0.5),
    dict(
        type='ImgAugWrapper',
        args=[['Fliplr', 0.5],
              dict(cls='Affine', rotate=[-10, 10]), ['Resize', [0.5, 3.0]]]),
    dict(type='RandomCrop', min_side_ratio=0.1),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='Pad', size=(640, 640)),
    dict(
        type='PackTextDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape'))
]

test_pipeline = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(type='Resize', scale=(1600, 800), keep_ratio=True),
    dict(
        type='LoadOCRAnnotations',
        with_polygon=True,
        with_bbox=True,
        with_label=True,
    ),
    dict(
        type='PackTextDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor',
                   'instances'))
]
