# Tutorial 2: Adding New Dataset

## Customize datasets by reorganizing data

### Customize loading annotations

You can write a new Dataset class inherited from `BaseFewShotDataset`, and overwrite `load_annotations(self)`,
like [CUB](https://github.com/open-mmlab/mmfewshot/blob/main/mmfewshot/classification/datasets/cub.py) and [MiniImageNet](https://github.com/open-mmlab/mmfewshot/blob/main/mmfewshot/classification/datasets/mini_imagenet.py).
Typically, this function returns a list, where each sample is a dict, containing necessary data information, e.g., `img` and `gt_label`.

Assume we are going to implement a `Filelist` dataset, which takes filelists for both training and testing. The format of annotation list is as follows:

```
000001.jpg 0
000002.jpg 1
```

We can create a new dataset in `mmfewshot/classification/datasets/filelist.py` to load the data.

```python
import mmcv
import numpy as np

from mmcls.datasets.builder import DATASETS
from .base import BaseFewShotDataset


@DATASETS.register_module()
class Filelist(BaseFewShotDataset):

    def load_annotations(self):
        assert isinstance(self.ann_file, str)

        data_infos = []
        with open(self.ann_file) as f:
            samples = [x.strip().split(' ') for x in f.readlines()]
            for filename, gt_label in samples:
                info = {'img_prefix': self.data_prefix}
                info['img_info'] = {'filename': filename}
                info['gt_label'] = np.array(gt_label, dtype=np.int64)
                data_infos.append(info)
            return data_infos

```

And add this dataset class in `mmcls/datasets/__init__.py`

```python
from .base_dataset import BaseDataset
...
from .filelist import Filelist

__all__ = [
    'BaseDataset', ... ,'Filelist'
]
```

Then in the config, to use `Filelist` you can modify the config as the following

```python
train = dict(
    type='Filelist',
    ann_file = 'image_list.txt',
    pipeline=train_pipeline
)
```

### Customize different subsets

To support different subset, we first predefine the classes of different subsets.
Then we modify `get_classes` to handle different classes of subset.

```python
import mmcv
import numpy as np

from mmcls.datasets.builder import DATASETS
from .base import BaseFewShotDataset

@DATASETS.register_module()
class Filelist(BaseFewShotDataset):

    TRAIN_CLASSES = ['train_a', ...]
    VAL_CLASSES = ['val_a', ...]
    TEST_CLASSES = ['test_a', ...]

    def __init__(self, subset, *args, **kwargs):
        ...
        self.subset = subset
        super().__init__(*args, **kwargs)

    def get_classes(self):
        if self.subset == 'train':
            class_names = self.TRAIN_CLASSES
        ...
        return class_names
```

## Customize datasets sampling

### EpisodicDataset

We use `EpisodicDataset` as wrapper to perform N way K shot sampling.
For example, suppose the original dataset is Dataset_A, the config looks like the following

```python
dataset_A_train = dict(
        type='EpisodicDataset',
        num_episodes=100000, # number of total episodes = length of dataset wrapper
        # each call of `__getitem__` will return
        # {'support_data': [(num_ways * num_shots) images],
        #  'query_data': [(num_ways * num_queries) images]}
        num_ways=5, # number of way (different classes)
        num_shots=5, # number of support shots of each class
        num_queries=5, # number of query shots of each class
        dataset=dict(  # This is the original config of Dataset_A
            type='Dataset_A',
            ...
            pipeline=train_pipeline
        )
    )
```

### Customize sampling logic

An example of customizing data sampling logic for training:

#### Create a new dataset wrapper

We can create a new dataset wrapper in mmfewshot/classification/datasets/dataset_wrappers.py to customize sampling logic.

```python
class MyDatasetWrapper:
    def __init__(self, dataset, args_a, args_b, ...):
        self.dataset = dataset
        ...
        self.episode_idxes = self.generate_episodic_idxes()

    def generate_episodic_idxes(self):
        episode_idxes = []
        # sampling each episode
        for _ in range(self.num_episodes):
            episodic_a_idx, episodic_b_idx, episodic_c_idx= [], [], []
            # customize sampling logic
            # select the index of data_infos from original dataset
            ...
            episode_idxes.append({
                'a': episodic_a_idx,
                'b': episodic_b_idx,
                'c': episodic_c_idx,
            })
        return episode_idxes

    def __getitem__(self, idx):
        # the key can be any value, but it needs to modify the code
        # in the forward function of model.
        return {
            'a_data' : [self.dataset[i] for i in self.episode_idxes[idx]['a']],
            'b_data' : [self.dataset[i] for i in self.episode_idxes[idx]['b']],
            'c_data' : [self.dataset[i] for i in self.episode_idxes[idx]['c']]
        }

```

#### Update dataset builder

We need to add the build code in mmfewshot/classification/datasets/builder.py
for our customize dataset wrapper.

```python
def build_dataset(cfg, default_args=None):
    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    ...
    elif cfg['type'] == 'MyDatasetWrapper':
        dataset = MyDatasetWrapper(
            build_dataset(cfg['dataset'], default_args),
            # pass customize arguments
            args_a=cfg['args_a'],
            args_b=cfg['args_b'],
            ...)
    else:
        dataset = build_from_cfg(cfg, DATASETS, default_args)

    return dataset
```

#### Update the arguments in model

The argument names in forward function need to be consistent with the customize dataset wrapper.

```python
class MyClassifier(BaseFewShotClassifier):
    ...
    def forward(self, a_data=None, b_data=None, c_data=None, ...):
        # pass the modified arguments name.
        if mode == 'train':
            return self.forward_train(a_data=a_data, b_data=b_data, c_data=None, **kwargs)
        elif mode == 'query':
            return self.forward_query(img=img, **kwargs)
        elif mode == 'support':
            return self.forward_support(img=img, **kwargs)
        elif mode == 'extract_feat':
            return self.extract_feat(img=img)
        else:
            raise ValueError()
```

#### using customize dataset wrapper in config

Then in the config, to use `MyDatasetWrapper` you can modify the config as the following,

```python
dataset_A_train = dict(
        type='MyDatasetWrapper',
        args_a=None,
        args_b=None,
        dataset=dict(  # This is the original config of Dataset_A
            type='Dataset_A',
            ...
            pipeline=train_pipeline
        )
    )
```
