
# DLplay

<p align="center">
  <img src="./images/dlplay-logo.png" alt="pySLAM - Stereo mapping example" height="150">
</p>

<!-- TOC -->

- [DLplay](#dlplay)
  - [Overview](#overview)
  - [Install](#install)
    - [With `pyenv`](#with-pyenv)
    - [With `pixi`](#with-pixi)
    - [Install `jax` with CUDA support](#install-jax-with-cuda-support)
  - [Usage](#usage)
  - [References](#references)
  - [Notes](#notes)

<!-- /TOC -->


## Overview 

```bash
├── apps/               # python scripts organized by domain
│   ├── classification/ # classification models and training
│   ├── detection/      # object detection scripts
│   ├── finetuning/     # model fine-tuning examples
│   ├── space3d/        # 3D point cloud processing
│   └── starting/       # beginner tutorials
├── data/               # datasets and sample data
│   ├── cv_data/        # computer vision sample images
│   ├── datasets/       # structured datasets
│   └── torch_data/     # PyTorch sample data
├── dlplay/             # core library modules
│   ├── core/           # core functionality
│   ├── datasets/       # dataset loaders and utilities
│   ├── detection/      # detection models
│   ├── geometry/       # geometric operations
│   ├── io/             # input/output utilities
│   ├── models/         # neural network models
│   ├── optimization/   # optimization algorithms
│   ├── pointcloud/     # point cloud processing
│   ├── segmentation/   # segmentation models
│   ├── utils/          # utility functions
│   └── viz/            # visualization tools
├── docs/               # documentation
├── notebooks/          # Jupyter notebooks
├── quickstart/         # tutorial examples
│   ├── cv/             # computer vision tutorials
│   ├── huggingface/    # Hugging Face examples
│   ├── jax/            # JAX tutorials
│   ├── matplotlib/     # plotting examples
│   ├── numpy/          # NumPy tutorials
│   ├── tensorflow/     # TensorFlow examples
│   └── torch/          # PyTorch tutorials
├── results/            # output folder for models and results
├── thirdparty/         # external dependencies
└── tools/              # bash tools and utilities
```

## Install 

### With `pyenv` 

First install: 
```bash
./tools/install_all.sh
```

Next, open a new terminal: 
```bash
pyenv activate dlplay
```
and you are ready to launch any script.

### With `pixi` 

First install 
```bash
sudo apt -y install curl 
curl -fsSL https://pixi.sh/install.sh | sh
source ~/.bashrc
pixi shell
pip install -e . 
./tools/install_torch_points3d.sh
```
Next, open a new terminal: 
```bash
pixi shell 
```
and you are ready to launch any script. 

### Install `jax` with CUDA support 

If you are under CUDA 12, run:
```bash
pip install -U "jax[cuda12]"
```

## Usage 

Explore the `apps` folder and select what you want to play with. 

## References

- opencv examples (see this [README.md](./quickstart/cv/README.md))
- pixi primer https://github.com/luigifreda/pyslam/blob/master/docs/PIXI.md 


## Notes

I used part of this repository for a course I previously taught, and I hope it may be of use to others.