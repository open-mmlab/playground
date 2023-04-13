# install
```bash
conda create -n mmocr1.0 python=3.8
# instal torch and torchvision, e,g, for cuda 11.1
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
# install mmocr
pip install -U openmim
mim install mmengine
mim install 'mmcv>=2.0.0rc1'
pip install 'mmdet>=3.0.0rc5'
cd mmocr-dev-1.x@9b0f1ba
pip install -v -e .
```
