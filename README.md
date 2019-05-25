# tensorflow_project

## meta
- Python 3.6.8
- conda 4.6.14
- tensorflow-gpu 1.13.1
- CUDA 9.0
- NVIDIA cuDNN v7.5.0 (Feb 21, 2019), for CUDA 9.0

## create train dataset
image location specified in config.py
 - *PROJECT_LOCATION\\dataset\\cifar10\\image\\*

```sh
python utils/tfrecord_util.py
```

## train model
training option specified in config.py
```sh
python train.py
```