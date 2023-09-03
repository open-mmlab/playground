# Tutorial 3: Customize Models

### Add a new classifier

Here we show how to develop a new classifier with an example as follows

#### 1. Define a new classifier

Create a new file `mmfewshot/classification/models/classifiers/my_classifier.py`.

```python
from mmcls.models.builder import CLASSIFIERS
from .base import BaseFewShotClassifier

@CLASSIFIERS.register_module()
class MyClassifier(BaseFewShotClassifier):

    def __init__(self, arg1, arg2):
        pass

    # customize input for different mode
    # the input should keep consistent with the dataset
    def forward(self, img, mode='train',**kwargs):
        if mode == 'train':
            return self.forward_train(img=img, **kwargs)
        elif mode == 'query':
            return self.forward_query(img=img,  **kwargs)
        elif mode == 'support':
            return self.forward_support(img=img, **kwargs)
        elif mode == 'extract_feat':
            assert img is not None
            return self.extract_feat(img=img)
        else:
            raise ValueError()

    # customize forward function for training data
    def forward_train(self, img, gt_label, **kwargs):
        pass

    # customize forward function for meta testing support data
    def forward_support(self, img, gt_label, **kwargs):
        pass

    # customize forward function for meta testing query data
    def forward_query(self, img):
        pass

    # prepare meta testing
    def before_meta_test(self, meta_test_cfg, **kwargs):
        pass

    # prepare forward meta testing query images
    def before_forward_support(self, **kwargs):
        pass

    # prepare forward meta testing support images
    def before_forward_query(self, **kwargs):
        pass
```

#### 2. Import the module

You can either add the following line to `mmfewshot/classification/models/heads/__init__.py`

```python
from .my_classifier import MyClassifier
```

or alternatively add

```python
custom_imports = dict(
    imports=['mmfewshot.classification.models.classifier.my_classifier'],
    allow_failed_imports=False)
```

to the config file to avoid modifying the original code.

#### 3. Use the classifier in your config file

```python
model = dict(
    type="MyClassifier",
    ...
)
```

### Add a new backbone

Here we show how to develop a new backbone with an example as follows

#### 1. Define a new backbone

Create a new file `mmfewshot/classification/models/backbones/mynet.py`.

```python
import torch.nn as nn

from mmcls.models.builder import BACKBONES

@BACKBONES.register_module()
class MyNet(nn.Module):

    def __init__(self, arg1, arg2):
        pass

    def forward(self, x):  # should return a tensor
        pass
```

#### 2. Import the module

You can either add the following line to `mmfewshot/classification/models/backbones/__init__.py`

```python
from .mynet import MyNet
```

or alternatively add

```python
custom_imports = dict(
    imports=['mmfewshot.classification.models.backbones.mynet'],
    allow_failed_imports=False)
```

to the config file to avoid modifying the original code.

#### 3. Use the backbone in your config file

```python
model = dict(
    ...
    backbone=dict(
        type='MyNet',
        arg1=xxx,
        arg2=xxx),
    ...
```

### Add new heads

Here we show how to develop a new head with an example as follows

#### 1. Define a new head

Create a new file `mmfewshot/classification/models/heads/myhead.py`.

```python
from mmcls.models.builder import HEADS
from .base_head import BaseFewShotHead

@HEADS.register_module()
class MyHead(BaseFewShotHead):

    def __init__(self, arg1, arg2) -> None:
        pass

    def forward_train(self, x, gt_label, **kwargs):
        pass

    def forward_support(self, x, gt_label, **kwargs):
        pass

    def forward_query(self, x, **kwargs):
        pass

    def before_forward_support(self) -> None:
        pass

    def before_forward_query(self) -> None:
        pass
```

#### 2. Import the module

You can either add the following line to `mmfewshot/classification/models/heads/__init__.py`

```python
from .myhead import MyHead
```

or alternatively add

```python
custom_imports = dict(
    imports=['mmfewshot.classification.models.backbones.myhead'],
    allow_failed_imports=False)
```

to the config file to avoid modifying the original code.

#### 3. Use the head in your config file

```python
model = dict(
    ...
    head=dict(
        type='MyHead',
        arg1=xxx,
        arg2=xxx),
    ...
```

### Add new loss

To add a new loss function, the users need implement it in `mmfewshot/classification/models/losses/my_loss.py`.
The decorator `weighted_loss` enable the loss to be weighted for each element.

```python
import torch
import torch.nn as nn

from ..builder import LOSSES
from .utils import weighted_loss

@weighted_loss
def my_loss(pred, target):
    assert pred.size() == target.size() and target.numel() > 0
    loss = torch.abs(pred - target)
    return loss

@LOSSES.register_module()
class MyLoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(MyLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * my_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss_bbox
```

Then the users need to add it in the `mmfewshot/classification/models/losses/__init__.py`.

```python
from .my_loss import MyLoss, my_loss
```

Alternatively, you can add

```python
custom_imports=dict(
    imports=['mmfewshot.classification.models.losses.my_loss'])
```

to the config file and achieve the same goal.

To use it, modify the `loss_xxx` field.
Since MyLoss is for regression, you need to modify the `loss_bbox` field in the head.

```python
loss_bbox=dict(type='MyLoss', loss_weight=1.0))
```
