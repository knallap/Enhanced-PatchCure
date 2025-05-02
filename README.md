# Enhancing PATCHCURE: Improving Robustness, Utility, and Efficiency in Adversarial Patch Defenses

Code for "[Enhancing PATCHCURE:
Improving Robustness, Utility, and Efficiency in Adversarial Patch Defenses]. 



### Overview

<img src="./assets/arch.png" align="center" width="90%" alt="defense overview pipeline" >

## Dependency

Tested with `torch==1.13.1` and `timm==0.9.16`. This repository should be compatible with newer version of packages. `requirements.txt` lists other required packages (with version numbers commented out).

## Files

```shell
├── README.md                        # this file 
├── requirements.txt                 # required packages
├── example_cmds.sh                  # command to reproduce PatchCURE results reported in the paper
├── reproducibility.md               # detailed guide for experiments. used for artifact evaluation at USENIX Security 
├── get_imagenet_val.sh              # script for downloading ImageNet val dataset.  
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
└── checkpoint                      # directory for checkpoint
    ├── README.md                    # details of checkpoint
    └── ...                          # model checkpoint
```

## Dataset

- [ImageNet](https://image-net.org/download.php) (ILSVRC2012). 

## Getting Started

1. See **Files** for details of each file. 
2. Download data in **Datasets** to `data/`.
3. Read [`checkpoint/README.md`](checkpoint/README.md) and download checkpoint from Google Drive [link](https://drive.google.com/drive/folders/146Qy-FKgSKrzuaaSluafhm3jYDQzYERj?usp=sharing) and move them to `checkpoint`.
4. See [`example_cmd.sh`](example_cmds.sh) for example commands for running the code and reproducing Enhanced PatchCURE results reported in the paper.
5. See `reproducibility.md` for a more detailed guide for running experiments. 

