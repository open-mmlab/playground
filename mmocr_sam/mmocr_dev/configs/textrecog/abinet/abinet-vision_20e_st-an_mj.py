_base_ = [
    '../_base_/datasets/mjsynth.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_adam_base.py',
    '_base_abinet-vision.py',
]

optim_wrapper = dict(optimizer=dict(lr=1e-4))
train_cfg = dict(max_epochs=20)
# learning policy
param_scheduler = [
    dict(
        type='LinearLR', end=2, start_factor=0.001,
        convert_to_iter_based=True),
    dict(type='MultiStepLR', milestones=[16, 18], end=20),
]

# dataset settings
train_list = [
    _base_.mjsynth_textrecog_test
]
test_list = [
    _base_.mjsynth_textrecog_test
]

train_dataset = dict(
    type='ConcatDataset', datasets=train_list, pipeline=_base_.train_pipeline)
test_dataset = dict(
    type='ConcatDataset', datasets=test_list, pipeline=_base_.test_pipeline)

train_dataloader = dict(
    batch_size=192 * 4,
    num_workers=32,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset)

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=test_dataset)

val_dataloader = test_dataloader

val_evaluator = dict(
    dataset_prefixes=['CUTE80', 'IIIT5K', 'SVT', 'SVTP', 'IC13', 'IC15'])
test_evaluator = val_evaluator

auto_scale_lr = dict(base_batch_size=192 * 8)
