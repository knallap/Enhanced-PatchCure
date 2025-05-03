## Overview

This document provides a detailed guide to reproduce all experimental results in the main body of our Enhanced PatchCURE paper.

## Setup

#### File directory
Below is an overview of relevant files. 

```shell
├── README.md                        # general purpose readme file 
├── reproducibility.md               # this file 
├── requirements.txt                 # required packages
├── example_cmds.sh                  # example commands to reproduce PatchCURE results reported in the paper (a subset of commands detailed in this doc)
| 
├── main.py                          # PatchCURE entry point.  
| 
├── utils
|   ├── builder.py                   # utils for building models and getting data loaders
|   ├── pcure.py                     # utils for PatchCURE inference algorithms 
|   ├── split.py                     # utils for splitting models into sub-models for PatchCURE construction
|   ├── bagnet.py                    # BagNet model; adapted from https://github.com/wielandbrendel/bag-of-local-features-models/blob/master/bagnets/pytorchnet.py
|   └── vit_srf.py                   # ViT-SRF model; based on timm/models/vision_transformer.py 
|
| 
├── data   
|   └── imagenet                     # data directory for imagenet # use torchvision.datasets.ImageFolder
|
└── checkpoints                      # directory for checkpoints
    ├── README.md                    # details of checkpoints
    └── ...                          # model checkpoints
```

#### Environment

1. Create a conda environment `conda create -n ae python=3.10`

2. Install PyTorch with GPU support. [1.13.1 version](https://pytorch.org/get-started/previous-versions/#v1131)  `conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia`

3. Install other packages `pip install -r requirement.txt`

**Note:** It might be some incompatibility issue with other versions of the library. It is recommended to follow the instructions above, which will install `python3.10` `torch==1.13.1` and `timm==0.9.16`.

#### Datasets
- [ImageNet](https://image-net.org/download.php) (ILSVRC2012) - requires manual download; also available on [Kaggle](https://www.kaggle.com/c/imagenet-object-localization-challenge/data)
  - we provide a script for downloading `val` data . run `bash get_imagenet_val.sh`
- [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) - will be downloaded automatically within our code (not the main focus of our experiments; only used in Table 4)
Move manually downloaded data to the directory `data/`.

#### Checkpoints
1. Download the following checkpoints from the Google Drive [link](https://drive.google.com/drive/folders/146Qy-FKgSKrzuaaSluafhm3jYDQzYERj?usp=sharing) if you want to run all experiments in the main body.

```
mae_masked.pth.tar
resnet50_masked.pth.tar
bagnet17_masked.pth.tar
bagnet33_masked.pth.tar
bagnet45_masked.pth.tar
vitsrf14x2_masked.pth.tar
vitsrf14x1_masked.pth.tar
vitsrf2x2_masked.pth.tar
vitsrf14x2_split11_masked.pth.tar
vitsrf14x2_split10_masked.pth.tar
vitsrf14x2_split9_masked.pth.tar
vitsrf14x2_split8_masked.pth.tar
vitsrf14x2_split7_masked.pth.tar
vitsrf14x2_split6_masked.pth.tar
vitsrf14x2_split5_masked.pth.tar
vitsrf14x2_split4_masked.pth.tar
vitsrf14x2_split3_masked.pth.tar
vitsrf14x2_split2_masked.pth.tar
vitsrf14x2_split1_masked.pth.tar
vitsrf14x2_split0_masked.pth.tar
vitsrf14x1_split0_masked.pth.tar
vitsrf2x2_split0_masked.pth.tar
vitlsrf14x2_masked.pth.tar
vitsrf14x2_split12_masked_cifar.pth.tar
vitsrf14x2_split6_masked_cifar.pth.tar
vitsrf2x2_split3_masked_cifar.pth.tar
```

2. Move downloaded weights to the directory `checkpoint/`.

## Experiments

In this section, we list all the commands used for getting experiment results for every table and figure in the main body.

1. Our evaluation metrics are **clean accuracy**, **certified robust accuracy**, **Inference throughput**, which will be outputted to the console. Below is the expected output after running `python main.py --model  vitsrf14x2_masked --patch-size 32 --mask-stride 1 --certify  --runtime --batch-size 4`. In this example, the clean accuracy of PCURE-ViT14x2-k12 defense is 79.4%, the certified robust accuracy is 64.7%, and its throughput with a batch size of 4 is 294.1 img/s. These numbers match the results reported in Table 2 (PCURE-ViT14x2-k12).


2.
   - Running experiments for the entire ImageNet dataset can take a long time. Please feel free to set ``--num_img`` to a small positive integer (e.g., 1000) to run experiments on a subset of the dataset. This will give an approximated evaluation result.
   - When ``--num_img`` is set to a negative integer, we will use the entire dataset for experiments; when it is set to a positive integer, we will use a random subset (with ``num_img`` images) for experiments.


#### Table 2: Main Results (as well as Figure 1 + Figure 6)

The following script is used for Table 2 (clean accuracy, certified robust accuracy, and throughput of Enhanced PatchCURE models).

```shell
## feel free to add --num-img 1000 to reduce runtime for approximated results from a 1000-image random subset
### SRF-only PatchCURE (split k=12 for ViT; k=50 for BagNet) ### takes ~10-30 mins for the entire ImageNet dataset. 
#EPCURE-ViT14x2-k12
python main.py --model  vitsrf14x2_masked --patch-size 32 --mask-stride 1 --certify  --runtime
#EPCURE-ViT14x1-k12
python main.py --model  vitsrf14x1_masked --patch-size 32 --mask-stride 1 --certify  --runtime
#EPCURE-ViT2x2-k12
python main.py --model  vitsrf2x2_masked --patch-size 32 --mask-stride 1 --certify  --runtime
#EPCURE-BagNet17k50
python main.py --model  bagnet45_masked --patch-size 32 --mask-stride 1 --certify  --runtime
#EPCURE-BagNet33k50
python main.py --model  bagnet33_masked --patch-size 32 --mask-stride 1 --certify  --runtime
#EPCURE-BagNet45k50
python main.py --model  bagnet17_masked --patch-size 32 --mask-stride 1 --certify  --runtime

### SRF+LRF PatchCURE # takes 0.5-2 hours for the entire ImageNet dataset
# EPCURE-ViT14x2-k11
python main.py --model  vitsrf14x2_split11_masked --patch-size 32 --mask-stride 1 --certify  --runtime
# EPCURE-ViT14x2-k10
python main.py --model  vitsrf14x2_split10_masked --patch-size 32 --mask-stride 1 --certify  --runtime
# EPCURE-ViT14x2-k9
python main.py --model  vitsrf14x2_split9_masked --patch-size 32 --mask-stride 1 --certify  --runtime
# EPCURE-ViT14x2-k6
python main.py --model  vitsrf14x2_split6_masked --patch-size 32 --mask-stride 1 --certify  --runtime
# EPCURE-ViT14x2-k3
python main.py --model  vitsrf14x2_split3_masked --patch-size 32 --mask-stride 1 --certify  --runtime

### LRF-only PatchCURE (k=0) # the certification can take a long time. you can add `--num-img 1000` to select a subset for evaluation.
# EPCURE-ViT14x2-k0
python main.py --model  vitsrf14x2_split0_masked --patch-size 32 --mask-stride 1 --certify   --runtime  # ~2 hours  
# EPCURE-ViT14x1-k0
python main.py --model  vitsrf14x1_split0_masked --patch-size 32 --mask-stride 1 --certify   --runtime  # ~8 hours
# EPCURE-ViT2x2-k0
python main.py --model  vitsrf2x2_split0_masked --patch-size 32 --mask-stride 1 --certify    --runtime  # ~2-3 days

```


#### Figure 4: Enhanced PatchCURE with different splitting location $k$

The following commands are for Figure 4 (Enhanced PatchCURE with different parameters $k$).


```shell
# each a few minutes to a few hours for the entire dataset
## feel free to add --num-img 1000 to reduce runtime for approximated results from a 1000-image random subset
#python main.py --model  vitsrf14x2_masked --patch-size 32 --mask-stride 1 --certify  --runtime # split 12
python main.py --model  vitsrf14x2_split11_masked --patch-size 32 --mask-stride 1 --certify  --runtime
python main.py --model  vitsrf14x2_split10_masked --patch-size 32 --mask-stride 1 --certify  --runtime
python main.py --model  vitsrf14x2_split9_masked --patch-size 32 --mask-stride 1 --certify  --runtime
python main.py --model  vitsrf14x2_split8_masked --patch-size 32 --mask-stride 1 --certify  --runtime
python main.py --model  vitsrf14x2_split7_masked --patch-size 32 --mask-stride 1 --certify  --runtime
python main.py --model  vitsrf14x2_split6_masked --patch-size 32 --mask-stride 1 --certify  --runtime
python main.py --model  vitsrf14x2_split5_masked --patch-size 32 --mask-stride 1 --certify  --runtime
python main.py --model  vitsrf14x2_split4_masked --patch-size 32 --mask-stride 1 --certify  --runtime
python main.py --model  vitsrf14x2_split3_masked --patch-size 32 --mask-stride 1 --certify  --runtime
python main.py --model  vitsrf14x2_split2_masked --patch-size 32 --mask-stride 1 --certify  --runtime
python main.py --model  vitsrf14x2_split1_masked --patch-size 32 --mask-stride 1 --certify  --runtime
```



#### Table 4: Results for CIFAR-10

The following commands are for Table 4 (different models for CIFAR-10).

```shell
# takes a few mins to run
# EPCURE-ViT14x2-k12
python main.py --model  vitsrf14x2_split12_masked_cifar --patch-size 32 --mask-stride 1 --certify  --runtime --dataset cifar
# EPCURE-ViT14x2-k6
python main.py --model  vitsrf14x2_split6_masked_cifar --patch-size 32 --mask-stride 1 --certify  --runtime --dataset cifar
# EPCURE-ViT2x2-k3
python main.py --model  vitsrf2x2_split3_masked_cifar --patch-size 32 --mask-stride 1 --certify  --runtime --dataset cifar

```






