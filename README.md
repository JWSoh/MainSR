# MainSR

This repo introduces some basic codes for training super-resolution networks based on [Tensorflow](https://www.tensorflow.org).

## Contents
1. How to generate TFRecord dataset for training.
2. How to load the TFRecord dataset for training.
3. Introduces some of our works.

## Environments
- Ubuntu 18.04
- NVIDIA GPU & GPU Driver
- Python 3
- [Tensorflow](https://www.tensorflow.org) (>=1.4)
- CUDA & cuDNN

----------
## Data Preparation
**Download Training Images for Super-Resolution**

### DIV2K
DIV2K contains high-resolution 800 training images, 100 validation images, and 100 test images.

You may use pre-arranged Tensorflow Dataset API (1st) or you may download the images (2nd) to build your own TFRecord dataset.

- Tensorflow (>= 2.0) (tfds.image.div2k,Div2k) [[Dataset API](https://www.tensorflow.org/datasets/catalog/div2k)]
- Images [[Download](https://data.vision.ee.ethz.ch/cvl/DIV2K/)]

### Flickr2K
- Images [[Download](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar)]
	
### BSD500 (Berkeley Segmentation Dataset 500)
- Images [[Download](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html#bsds500)]

<br><br>
## Generate TFRecord Dataset

First, prepare your training dataset.
Second, run the code below.

```
python generate_TFRecord.py --scale [Scaling Factor] --labelpath [Path to HR images] --datapath [Path to LR images] --tfrecord [TFRecord file path]
```
**Example Code**
```
python generate_TFRecord.py --scale 2 --labelpath DIV2K/DIV2K/DIV2K_train_HR --datapath DIV2K/DIV2K/DIV2K_train_LR_bicubic --tfrecord train_SR_X2.tfrecord
```

**Important Notes**

This code design is for DIV2K settings.

- Line 38: You may change the format of "datapath." <br>
- Line 94: You may change the strides and patch sizes.<br>
- Line 63: You may remove the gradient option or change the threshold.<br>

<br><br>
## Verify TFRecord Dataset

After creating the TFRecord file, the code below can be used to verify your TFRecord file.
Also, you can modify and embed this code for training any other networks.

Navigate **dataLoader.py** file and modify it for other general usage of this code.

### Code Design
```
├─ config.py: All the configurations, including Argument Parser.
├─ dataLoader.py: Inherits "config" class to retrieve all the configuration values.
        ├──── __init__(): Constructor.
                ├────> "BUFFER_SIZE" for the size for shuffling.
                └────> "augmentation" a flag for data augmentation.
        ├──── _parse_function(): To parse the Tfrecord file.
        └──── load_tfrecord(): To load the tfrecord file to iterative .
└─ Visualize_TFRecord.py: Run code for visualization.
```

Run the code below.
```
python Visualize_TFRecord.py --gpu [GPU_number] --tfrecord [TFRecord File]
```
**Example Code** 
```
python Visualize_TFRecord.py --gpu 0 --tfrecord train_SR_X2.tfrecord
```

**Important Notes**

This code is for Tensorflow version 1.x.
- You don't need tf.Session() for Tensorflow 2.x versions.
- Requires slight modification of dataLoader.load_tfrecord() for Tensorflow 2.x versions.

## Introduction of Our Works

Natural and Realistic Single Image Super-Resolution with Explicit Natural Manifold Discrimination (CVPR, 2019)[Paper](http://openaccess.thecvf.com/content_CVPR_2019/html/Soh_Natural_and_Realistic_Single_Image_Super-Resolution_With_Explicit_Natural_Manifold_CVPR_2019_paper.html) [Repo](https://www.github.com/JWSoh/NatSR)

Deep Hierarchical Single Image Super-Resolution by Exploiting Controlled Diverse Context Features (ISM, 2019)[Paper](https://ieeexplore.ieee.org/abstract/document/8959052/) [Repo](https://www.github.com/JWSoh/DHSR)

Lightweight Single Image Super-Resolution With Multi-Scale Spatial Attention Networks (IEEE ACCESS, 2020)] [Paper](https://ieeexplore.ieee.org/abstract/document/9001090) [Repo](https://www.github.com/JWSoh/MSAN)

Meta-Transfer Learning for Zero-Shot Super-Resolution (CVPR, 2020) [Arxiv](https://arxiv.org/abs/2002.12213) [Repo](https://www.github.com/JWSoh/MZSR)

## References
Please cite our paper if this repo is helpful to your research.

```
@inproceedings{soh2019natural,
  title={Natural and realistic single image super-resolution with explicit natural manifold discrimination},
  author={Soh, Jae Woong and Park, Gu Yong and Jo, Junho and Cho, Nam Ik},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={8122--8131},
  year={2019}
}

@inproceedings{soh2019deep,
  title={Deep Hierarchical Single Image Super-Resolution by Exploiting Controlled Diverse Context Features},
  author={Soh, Jae Woong and Park, Gu Yong and Cho, Nam Ik},
  booktitle={2019 IEEE International Symposium on Multimedia (ISM)},
  pages={160--1608},
  year={2019},
  organization={IEEE}
}

@article{soh2020lightweight,
  title={Lightweight Single Image Super-Resolution With Multi-Scale Spatial Attention Networks},
  author={Soh, Jae Woong and Cho, Nam Ik},
  journal={IEEE Access},
  volume={8},
  pages={35383--35391},
  year={2020},
  publisher={IEEE}
}

@article{soh2020meta,
  title={Meta-Transfer Learning for Zero-Shot Super-Resolution},
  author={Soh, Jae Woong and Cho, Sunwoo and Cho, Nam Ik},
  journal={arXiv preprint arXiv:2002.12213},
  year={2020}
}

```
