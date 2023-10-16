# Tutorial 1: Learn about Configs

We incorporate modular and inheritance design into our config system, which is convenient to conduct various experiments.
If you wish to inspect the config file, you may run `python tools/misc/print_config.py /PATH/TO/CONFIG` to see the complete config.
The classification part of mmfewshot is built upon the [mmcls](https://github.com/open-mmlab/mmclassification),
thus it is highly recommended learning the basic of mmcls.

## Modify config through script arguments

When submitting jobs using "tools/classification/train.py" or "tools/classification/test.py", you may specify `--cfg-options` to in-place modify the config.

- Update config keys of dict chains.

  The config options can be specified following the order of the dict keys in the original config.
  For example, `--cfg-options model.backbone.norm_eval=False` changes the all BN modules in model backbones to `train` mode.

- Update keys inside a list of configs.

  Some config dicts are composed as a list in your config. For example, the training pipeline `data.train.pipeline` is normally a list
  e.g. `[dict(type='LoadImageFromFile'), ...]`. If you want to change `'LoadImageFromFile'` to `'LoadImageFromWebcam'` in the pipeline,
  you may specify `--cfg-options data.train.pipeline.0.type=LoadImageFromWebcam`.

- Update values of list/tuples.

  If the value to be updated is a list or a tuple. For example, the config file normally sets `workflow=[('train', 1)]`. If you want to
  change this key, you may specify `--cfg-options workflow="[(train,1),(val,1)]"`. Note that the quotation mark " is necessary to
  support list/tuple data types, and that **NO** white space is allowed inside the quotation marks in the specified value.

## Config Name Style

We follow the below style to name config files. Contributors are advised to follow the same style.

```
{algorithm}_[algorithm setting]_{backbone}_{gpu x batch_per_gpu}_[misc]_{dataset}_{meta test setting}.py
```

`{xxx}` is required field and `[yyy]` is optional.

- `{algorithm}`: model type like `faster_rcnn`, `mask_rcnn`, etc.
- `[algorithm setting]`: specific setting for some model, like `without_semantic` for `htc`, `moment` for `reppoints`, etc.
- `{backbone}`: backbone type like `conv4`, `resnet12`.
- `[norm_setting]`: `bn` (Batch Normalization) is used unless specified, other norm layer type could be `gn` (Group Normalization), `syncbn` (Synchronized Batch Normalization).
  `gn-head`/`gn-neck` indicates GN is applied in head/neck only, while `gn-all` means GN is applied in the entire model, e.g. backbone, neck, head.
- `[gpu x batch_per_gpu]`: GPUs and samples per GPU. For episodic training methods we use the total number of images in one episode, i.e. n classes x (support images+query images).
- `[misc]`: miscellaneous setting/plugins of model.
- `{dataset}`: dataset like `cub`, `mini-imagenet` and `tiered-imagenet`.
- `{meta test setting}`: n way k shot setting like `5way_1shot` or `5way_5shot`.

## An Example of Baseline

To help the users have a basic idea of a complete config and the modules in a modern classification system,
we make brief comments on the config of Baseline for MiniImageNet in 5 way 1 shot setting as the following.
For more detailed usage and the corresponding alternative for each module, please refer to the API documentation.

```python
# config of model
model = dict(
    # classifier name
    type='Baseline',
    # config of backbone
    backbone=dict(type='Conv4'),
    # config of classifier head
    head=dict(type='LinearHead', num_classes=64, in_channels=1600),
    # config of classifier head used in meta test
    meta_test_head=dict(type='LinearHead', num_classes=5, in_channels=1600))

# data pipeline for training
train_pipeline = [
    # first pipeline to load images from file path
    dict(type='LoadImageFromFile'),
    # random resize crop
    dict(type='RandomResizedCrop', size=84),
    # random flip
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    # color jitter
    dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.4),
    dict(type='Normalize',  # normalization
         # mean values used to normalization
         mean=[123.675, 116.28, 103.53],
         # standard variance used to normalization
         std=[58.395, 57.12, 57.375],
         # whether to invert the color channel, rgb2bgr or bgr2rgb
         to_rgb=True),
    # convert img into torch.Tensor
    dict(type='ImageToTensor', keys=['img']),
    # convert gt_label into torch.Tensor
    dict(type='ToTensor', keys=['gt_label']),
    # pipeline that decides which keys in the data should be passed to the runner
    dict(type='Collect', keys=['img', 'gt_label'])
]

# data pipeline for testing
test_pipeline = [
    # first pipeline to load images from file path
    dict(type='LoadImageFromFile'),
    # resize image
    dict(type='Resize', size=(96, -1)),
    # center crop
    dict(type='CenterCrop', crop_size=84),
    dict(type='Normalize',  # normalization
         # mean values used to normalization
         mean=[123.675, 116.28, 103.53],
         # standard variance used to normalization
         std=[58.395, 57.12, 57.375],
         # whether to invert the color channel, rgb2bgr or bgr2rgb
         to_rgb=True),
    # convert img into torch.Tensor
    dict(type='ImageToTensor', keys=['img']),
    # pipeline that decides which keys in the data should be passed to the runner
    dict(type='Collect', keys=['img', 'gt_label'])
]

# config of fine-tuning using support set in Meta Test
meta_finetune_cfg = dict(
    # number of iterations in fine-tuning
    num_steps=150,
    # optimizer config in fine-tuning
    optimizer=dict(
        type='SGD',  # optimizer name
        lr=0.01,  # learning rate
        momentum=0.9,  # momentum
        dampening=0.9,  # dampening
        weight_decay=0.001)),  # weight decay

data = dict(
    # batch size of a single GPU
    samples_per_gpu=64,
    # worker to pre-fetch data for each single GPU
    workers_per_gpu=4,
    # config of training set
    train=dict(
        # name of dataset
        type='MiniImageNetDataset',
        # prefix of image
        data_prefix='data/mini_imagenet',
        # subset of dataset
        subset='train',
        # train pipeline
        pipeline=train_pipeline),
    # config of validation set
    val=dict(
        # dataset wrapper for Meta Test
        type='MetaTestDataset',
        # total number of test tasks
        num_episodes=100,
        num_ways=5,  # number of class in each task
        num_shots=1,  # number of support images in each task
        num_queries=15,  # number of query images in each task
        dataset=dict(  # config of dataset
            type='MiniImageNetDataset',  # dataset name
            subset='val',  # subset of dataset
            data_prefix='data/mini_imagenet',  # prefix of images
            pipeline=test_pipeline),
        meta_test_cfg=dict(  # config of Meta Test
            num_episodes=100,  # total number of test tasks
            num_ways=5,  # number of class in each task
            # whether to pre-compute features from backbone for acceleration
            fast_test=True,
            # dataloader setting for feature extraction of fast test
            test_set=dict(batch_size=16, num_workers=2),
            support=dict(  # support set setting in meta test
                batch_size=4,  # batch size for fine-tuning
                num_workers=0,  # number of worker set 0 since the only 5 images
                drop_last=True,  # drop last
                train=dict(  # config of fine-tuning
                    num_steps=150,  # number of steps in fine-tuning
                    optimizer=dict(  # optimizer config in fine-tuning
                        type='SGD',  # optimizer name
                        lr=0.01,  # learning rate
                        momentum=0.9,  # momentum
                        dampening=0.9,  # dampening
                        weight_decay=0.001))),  # weight decay
            # query set setting predict 75 images
            query=dict(batch_size=75, num_workers=0))),
    test=dict(  # used for model validation in Meta Test fashion
        type='MetaTestDataset',  # dataset wrapper for Meta Test
        num_episodes=2000,  # total number of test tasks
        num_ways=5,  # number of class in each task
        num_shots=1,  # number of support images in each task
        num_queries=15,  # number of query images in each task
        dataset=dict(  # config of dataset
            type='MiniImageNetDataset',  # dataset name
            subset='test',  # subset of dataset
            data_prefix='data/mini_imagenet',  # prefix of images
            pipeline=test_pipeline),
        meta_test_cfg=dict(  # config of Meta Test
            num_episodes=2000,  # total number of test tasks
            num_ways=5,  # number of class in each task
            # whether to pre-compute features from backbone for acceleration
            fast_test=True,
            # dataloader setting for feature extraction of fast test
            test_set=dict(batch_size=16, num_workers=2),
            support=dict(  # support set setting in meta test
                batch_size=4,  # batch size for fine-tuning
                num_workers=0,  # number of worker set 0 since the only 5 images
                drop_last=True,  # drop last
                train=dict(  # config of fine-tuning
                    num_steps=150,  # number of steps in fine-tuning
                    optimizer=dict(  # optimizer config in fine-tuning
                        type='SGD',  # optimizer name
                        lr=0.01,  # learning rate
                        momentum=0.9,  # momentum
                        dampening=0.9,  # dampening
                        weight_decay=0.001))),  # weight decay
            # query set setting predict 75 images
            query=dict(batch_size=75, num_workers=0))))
log_config = dict(
    interval=50,  # interval to print the log
    hooks=[dict(type='TextLoggerHook')])
checkpoint_config = dict(interval=20)  # interval to save a checkpoint
evaluation = dict(
    by_epoch=True,  # eval model by epoch
    metric='accuracy',  # Metrics used during evaluation
    interval=5)  # interval to eval model
# parameters to setup distributed training, the port can also be set.
dist_params = dict(backend='nccl')
log_level = 'INFO'  # the output level of the log.
load_from = None  # load a pre-train checkpoints
# resume checkpoints from a given path, the training will be resumed from
# the epoch when the checkpoint's is saved.
resume_from = None
# workflow for runner. [('train', 1)] means there is only one workflow and
# the workflow named 'train' is executed once.
workflow = [('train', 1)]
pin_memory = True  # whether to use pin memory
# whether to use infinite sampler; infinite sampler can accelerate training efficient
use_infinite_sampler = True
seed = 0  # random seed
runner = dict(type='EpochBasedRunner', max_epochs=200)  # runner type and epochs of training
optimizer = dict(  # the configuration file used to build the optimizer, support all optimizers in PyTorch.
    type='SGD',  # optimizer type
    lr=0.05,  # learning rat
    momentum=0.9,  # momentum
    weight_decay=0.0001)  # weight decay of SGD
optimizer_config = dict(grad_clip=None)  # most of the methods do not use gradient clip
lr_config = dict(
    # the policy of scheduler, also support CosineAnnealing, Cyclic, etc. Refer to details of supported LrUpdater
    # from https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py#L9.
    policy='step',
    warmup='linear',  # warmup type
    warmup_iters=3000,  # warmup iterations
    warmup_ratio=0.25,  # warmup ratio
    step=[60, 120])  # Steps to decay the learning rate
```

## FAQ

### Use intermediate variables in configs

Some intermediate variables are used in the configuration file. The intermediate variables make the configuration file clearer and easier to modify.

For example, `train_pipeline` / `test_pipeline` is the intermediate variable of the data pipeline. We first need to define `train_pipeline` / `test_pipeline`, and then pass them to `data`. If you want to modify the size of the input image during training and testing, you need to modify the intermediate variables of `train_pipeline` / `test_pipeline`.

```python
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=384, backend='pillow',),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=384, backend='pillow'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
```

### Ignore some fields in the base configs

Sometimes, you need to set `_delete_=True` to ignore some domain content in the basic configuration file. You can refer to [mmcv](https://mmcv.readthedocs.io/en/latest/understand_mmcv/config.html#inherit-from-base-config-with-ignored-fields) for more instructions.

The following is an example. If you want to use cosine schedule, just using inheritance and directly modify it will report `get unexcepected keyword'step'` error, because the `'step'` field of the basic config in `lr_config` domain information is reserved, and you need to add `_delete_ =True` to ignore the content of `lr_config` related fields in the basic configuration file:

```python
lr_config = dict(
    _delete_=True,
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    by_epoch=True,
    warmup_iters=5,
    warmup_ratio=0.1
)
```
