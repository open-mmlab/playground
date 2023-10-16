## Getting Started

This page provides basic tutorials about the usage of `OpenUnReID`. The training and testing scripts can be found in `OpenUnReID/tools`.

We use 4 GPUs for training and testing, which is considered as a default setting in the scripts. You can adjust it (e.g. `${GPUS}`, `${GPUS_PER_NODE}`) based on your own needs.

### Test

#### Testing commands

+ Distributed testing with multiple GPUs:
```shell
bash dist_test.sh ${RESUME} ${CONFIG} [optional arguments]
```
+ Distributed testing with multiple machines:
```shell
bash slurm_test.sh ${PARTITION} ${RESUME} ${CONFIG} [optional arguments]
```
+ Testing with a single GPU:
```shell
GPUS=1 bash dist_test.sh ${RESUME} ${CONFIG} [optional arguments]
```

#### Arguments

+ `${RESUME}`: model for testing, e.g. `../logs/MMT/market1501/model_best.pth`.
+ `${CONFIG}`: config file for the model, e.g. `MMT/config.yaml`. **Note** the config is required to match the model.
+ `[optional arguments]`: modify some key values from the loaded config file, e.g. `TEST.rerank True`. (it's also ok to make the modification directly in the config file)

#### Configs

+ Test with different datasets, e.g.
```shell
TEST:
  datasets: ['dukemtmcreid', 'market1501',] # arrange the names in a list
```
+ Add re-ranking post-processing, e.g.
```shell
TEST:
  rerank: True # default: False
```
+ Save GPU memory but with a lower speed,
```shell
TEST:
  dist_cuda: False # use CPU for computing distances, default: True
  search_type: 3 # use CPU for re-ranking, default: 0 (1/2 is also for GPU)
```
+ ... (TBD)

### Train

#### Training commands

+ Distributed training with multiple GPUs:
```shell
bash dist_train.sh ${METHOD} ${WORK_DIR} [optional arguments]
```
+ Distributed training with multiple machines:
```shell
bash slurm_train.sh ${PARTITION} ${JOB_NAME} ${METHOD} ${WORK_DIR} [optional arguments]
```
+ Training with a single GPU:
> Please add `TRAIN.LOADER.samples_per_gpu 64` in `[optional arguments]`.

```shell
GPUS=1 bash dist_train.sh ${METHOD} ${WORK_DIR} [optional arguments]
```

#### Arguments

+ `${METHOD}`: method for training, e.g. `source_pretrain`, `UDA_TP`, `MMT`, `SpCL`.
+ `${WORK_DIR}`: folder for saving logs and checkpoints, e.g. `MMT/market1501`, the absolute path will be `LOGS_ROOT/${WORK_DIR}` (`LOGS_ROOT` is defined in config files).
+ `[optional arguments]`: modify some key values from the loaded config file, e.g. `TRAIN.val_freq 10`. (it's also ok to make the modification directly in the config file)

#### Configs

+ Flexible architectures,
```shell
MODEL:
  backbone: 'resnet50' # or 'resnet101', 'resnet50_ibn_a', etc
  pooling: 'gem' # or 'avg', 'max', etc
  dsbn: True # domain-specific BNs, critical for domain adaptation performance
```
+ Ensure reproducibility (may cause a lower speed),
```shell
TRAIN:
  deterministic: True
```
+ Flexible datasets,

the conventional USL task, e.g. unsupervised market1501
```shell
TRAIN:
  # arrange the names in a dict, {DATASET_NAME: DATASET_SPLIT}
  datasets: {'market1501': 'trainval'}
  # define the unsupervised dataset indexes, here index=[0] means market1501 is unlabeled
  unsup_dataset_indexes: [0,]
  # val_set of 'market1501' will be used for validation
  val_dataset: 'market1501'
```
the USL task with multiple datasets, e.g. unsupervised market1501+dukemtmcreid
```shell
TRAIN:
  # you could use multiple datasets for training without the limitation of numbers
  datasets: {'market1501': 'trainval', 'dukemtmcreid': 'trainval'}
  unsup_dataset_indexes: [0,1]
  # you could only choose one dataset for validation (to select the best model)
  # however, you could use multiple datasets for testing (TEST.datasets)
  val_dataset: 'market1501'
```
the conventional UDA task, e.g. market1501 -> dukemtmcreid
```shell
TRAIN:
  datasets: {'market1501': 'trainval', 'dukemtmcreid': 'trainval'}
  unsup_dataset_indexes: [1,]
  val_dataset: 'dukemtmcreid'
```
the UDA task with multiple source-domain datasets, e.g. market1501+msmt17 -> dukemtmcreid
```shell
TRAIN:
  datasets: {'market1501': 'trainval', 'msmt17': 'trainval', 'dukemtmcreid': 'trainval'}
  unsup_dataset_indexes: [2,]
  val_dataset: 'dukemtmcreid'
```
the UDA task with multiple target-domain datasets, e.g. msmt17 -> market1501+dukemtmcreid
```shell
TRAIN:
  datasets: {'market1501': 'trainval', 'msmt17': 'trainval', 'dukemtmcreid': 'trainval'}
  unsup_dataset_indexes: [0,2]
  val_dataset: 'market1501'
```
+ Train with different losses, e.g.
```shell
TRAIN:
  losses: {'cross_entropy': 1., 'softmax_triplet': 1.} # {LOSS_NAME: LOSS_WEIGHT}
```
+ Save GPU memory but with a lower speed,
```shell
TRAIN:
  PSEUDO_LABELS:
    search_type: 3 # only for dbscan, use CPU for searching top-k, default: 0 (1/2 is also for GPU)
    dist_cuda: False # only for kmeans, use CPU for computing distances, default: True
```
+ Mixed precision training
```shell
TRAIN:
  amp: True # mixed precision training for PyTorch>=1.6
```
+ ... (TBD)
