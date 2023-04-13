_base_ = [
    'fcenet_resnet50-dcnv2_fpn_1500e_ctw1500.py',
]

load_from = None

_base_.model.backbone = dict(
    type='CLIPResNet',
    out_indices=(1, 2, 3),
    init_cfg=dict(
        type='Pretrained',
        checkpoint='https://download.openmmlab.com/'
        'mmocr/backbone/resnet50-oclip-7ba0c533.pth'))

_base_.train_dataloader.num_workers = 24
_base_.optim_wrapper.optimizer.lr = 0.0005
