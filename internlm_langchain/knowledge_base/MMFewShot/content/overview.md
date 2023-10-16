# Tutorial 0: Overview of MMFewShot Classification

The main difference between general classification task and few shot classification task
is the data usage.
Therefore, the design of MMFewShot target at data sampling, meta test and models apis for few shot setting based on [mmcls](https://github.com/open-mmlab/mmclassification).
Additionally, the modules in [mmcls](https://github.com/open-mmlab/mmclassification) can be imported and reused in the code or config.

## Design of Data Sampling

In MMFewShot, we suggest customizing the data pipeline using a dataset wrapper and modify the arguments in forward
function when returning the dict with customize keys.

```python
class CustomizeDataset:

    def __init__(self, dataset, ...):
        self.dataset = dataset
        self.customize_list = generate_function(dataset)

    def generate_function(self, dataset):
        pass

    def __getitem__(self, idx):
        return {
            'support_data': [self.dataset[i] for i in self.customize_list],
            'query_data': [self.dataset[i] for i in self.customize_list]
        }
```

More details can refer to [Tutorial 2: Adding New Dataset](https://mmfewshot.readthedocs.io/en/latest/classification/customize_dataset.html)

## Design of Models API

Each model in MMFewShot should implement following functions to support meta testing.
More details can refer to [Tutorial 3: Customize Models](https://mmfewshot.readthedocs.io/en/latest/classification/customize_models.html)

```python
@CLASSIFIERS.register_module()
class BaseFewShotClassifier(BaseModule):

    def forward(self, mode, ...):
        if mode == 'train':
            return self.forward_train(...)
        elif mode == 'query':
            return self.forward_query(...)
        elif mode == 'support':
            return self.forward_support(...)
        ...

    def forward_train(self, **kwargs):
        pass

    # --------- for meta testing ----------
    def forward_support(self, **kwargs):
        pass

    def forward_query(self, **kwargs):
        pass

    def before_meta_test(self, meta_test_cfg, **kwargs):
        pass

    def before_forward_support(self, **kwargs):
        pass

    def before_forward_query(self, **kwargs):
        pass

```

## Design of Meta Testing

Meta testing performs prediction on random sampled tasks multiple times.
Each task contains support and query data.
More details can refer to `mmfewshot/classification/apis/test.py`.
Here is the basic pipeline for meta testing:

```text
# the model may from training phase and may generate or fine-tine weights
1. Copy model
# prepare for the meta test (generate or freeze weights)
2. Call model.before_meta_test()
# some methods with fixed backbone can pre-compute the features for acceleration
3. Extracting features of all images for acceleration(optional)
# test different random sampled tasks
4. Test tasks (loop)
    # make sure all the task share the same initial weight
    a. Copy model
    # prepare model for support data
    b. Call model.before_forward_support()
    # fine-tune or none fine-tune models with given support data
    c. Forward support data: model(*data, mode='support')
    # prepare model for query data
    d. Call model.before_forward_query()
    # predict results of query data
    e. Forward query data: model(*data, mode='query')
```

### meta testing on multiple gpus

In MMFewShot, we also support multi-gpu meta testing during
validation or testing phase.
In multi-gpu meta testing, the model will be copied and wrapped with `MetaTestParallel`, which will
send data to the device of model.
Thus, the original model will not be affected by the operations in Meta Testing.
More details can refer to `mmfewshot/classification/utils/meta_test_parallel.py`
Specifically, each gpu will be assigned with (num_test_tasks / world_size) task.
Here is the distributed logic for multi gpu meta testing:

```python
sub_num_test_tasks = num_test_tasks // world_size
sub_num_test_tasks += 1 if num_test_tasks % world_size != 0 else 0
for i in range(sub_num_test_tasks):
    task_id = (i * world_size + rank)
    if task_id >= num_test_tasks:
        continue
    # test task with task_id
    ...
```

If user want to customize the way to test a task, more details can refer to [Tutorial 4: Customize Runtime Settings](https://mmfewshot.readthedocs.io/en/latest/classification/customize_runtime.html)
