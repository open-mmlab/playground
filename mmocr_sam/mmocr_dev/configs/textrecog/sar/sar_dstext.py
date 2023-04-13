_base_ = [
    '../_base_/datasets/dstext.py',
    '../_base_/schedules/schedule_adamw_cos_10e.py',
    '_base_sar_resnet31_parallel-decoder.py',
    '../_base_/default_runtime.py',
]

dstext_textrecog_train = _base_.dstext_textrecog_train
dstext_textrecog_train.pipeline = _base_.train_pipeline
dstext_textrecog_test = _base_.dstext_textrecog_test
dstext_textrecog_test.pipeline = _base_.test_pipeline

default_hooks = dict(logger=dict(type='LoggerHook', interval=5))

train_dataloader = dict(
    batch_size=64,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dstext_textrecog_train)

test_dataloader = dict(
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dstext_textrecog_test)

val_dataloader = test_dataloader
