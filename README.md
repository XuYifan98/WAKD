# Weak-supervised Visual Geo-localization via Attention-based Knowledge Distillation


## Introduction
`WAKD` is a PyTorch implementation for our ICPR-2022 paper "Weak-supervised Visual Geo-localization via Attention-based Knowledge Distillation".


## Installation
We test this repo with Python 3.8, PyTorch 1.9.0, and CUDA 10.2. But it should be runnable with recent PyTorch versions (Pytorch >=1.0.0).
```shell
python setup.py develop
```


## Preparation
### Datasets

We test our models on three geo-localization benchmarks, [Pittsburgh](https://www.cv-foundation.org/openaccess/content_cvpr_2013/papers/Torii_Visual_Place_Recognition_2013_CVPR_paper.pdf), [Tokyo 24/7](https://www.di.ens.fr/~josef/publications/Torii15.pdf) and [Tokyo Time Machine](https://arxiv.org/abs/1511.07247) datasets. The three datasets can be downloaded at [here](https://www.di.ens.fr/willow/research/netvlad/).

The directory of datasets used is like
```shell
datasets/data
├── pitts
│   ├── raw
│   │   ├── pitts250k_test.mat
│   │   ├── pitts250k_train.mat
│   │   ├── pitts250k_val.mat
│   │   ├── pitts30k_test.mat
│   │   ├── pitts30k_train.mat
│   │   ├── pitts30k_val.mat
│   └── └── Pittsburgh
│           ├──images/
│           └──queries/
└── tokyo
    ├── raw
    │   ├── tokyo247
    │   │   ├──images/
    │   │   └──query/
    │   ├── tokyo247.mat
    │   ├── tokyoTM/images/
    │   ├── tokyoTM_train.mat
    └── └── tokyoTM_val.mat
```

### Pre-trained Weights

The file tree we used for storing the pre-trained weights is like
```shell
logs
├── vgg16_pretrained.pth.tar # refer to (1)
├── mbv3_large.pth.tar
└── vgg16_pitts_64_desc_cen.hdf5 # refer to (2)
└── mobilenetv3_large_pitts_64_desc_cen.hdf5
```

**(1) ImageNet-pretrained weights for CNNs backbone

The ImageNet-pretrained weights for CNNs backbone or the pretrained weights for the whole model.

**(2) initial cluster centers for VLAD layer**

Note that the VLAD layer cannot work with random initialization.
The original cluster centers provided by NetVLAD or self-computed cluster centers by running the scripts/cluster.sh.

```shell
./scripts/cluster.sh mobilenetv3_large
```
