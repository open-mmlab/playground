_base_ = [
    '_base_dbnetpp_swinv2_b_w16_in21k.py',
    '../_base_/default_runtime.py',
    '../_base_/datasets/laion400m.py',
    '../_base_/schedules/schedule_sgd_1200e.py'
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=40, val_interval=5)
param_scheduler = [dict(type='CosineAnnealingLR', T_max=40, eta_min=1e-7)]
load_from = 'checkpoints/swin/swin_hier_epoch_100.pth'
# dataset settings
train_list = [_base_.laion400m_textdet_train]
test_list = [_base_.laion400m_textdet_test]

train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=train_list,
        pipeline=_base_.train_pipeline))

val_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets=test_list,
        pipeline=_base_.test_pipeline))

test_dataloader = val_dataloader

auto_scale_lr = dict(base_batch_size=16)
