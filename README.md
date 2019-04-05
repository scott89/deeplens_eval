## DeepLens: Shallow Depth-of-Field from a Single Image

### Introduction
This repo contains interactive evaluation code for our [SIGGRAPH Asia 2018 paper](https://arxiv.org/abs/1810.08100). 
More details can be found in our project webpage [English](https://scott89.github.io/deeplens/DeepLens.html) or [中文](https://scott89.github.io/DeepLens_zh/DeepLens.html).

### Usage

* Supported OS: the source code has been tested on 64-bit Ubuntu 14.04 and 16.04 OS with python 2.7, and it should also be executable in other linux distributions.

* Dependencies: 
 * Tensorflow >= 1.8 and all its dependencies. 
 * Cuda enabled GPUs

* How to run: 
 1. Download pretrained models from [here](http://pan.dlut.edu.cn/share?id=tuvjqqshhgyu) and place checkpoint files in 'model/'.
 2. Run `python eval.py`, the input image and the corresponding depth maps will be displayed.
 3. Select the focal point by clicking on the image and slide the slider to adjust aperture radius. Shallow depth of field effects will be rendered automatically.

    fad

### Citing Our Work


If you find DeepLens useful in your research, please consider to cite our paper:

    @ARTICLE{deeplens2018, 
    author={Wang Lijun and Shen Xiaohui and Zhang Jianming and Wang Oliver and Lin Zhe and Hsieh Chih-Yao and Kong Sarah and Lu Huchuan}, 
    title={DeepLens: Shallow Depth of Field from a Single Image}, 
    journal={ACM Trans. Graph. (Proc. SIGGRAPH Asia)}, 
    year={2018}, 
    pages = {6:1-6:11}, 
    volume = {37}, 
    number = {6} 
    }
