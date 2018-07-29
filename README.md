# Semantic Soft Segmentation, ACM SIGGRAPH 2018

This repository includes the Tensorflow implementation of the semantic feature extraction part used in \[Semantic Soft Segmentation (Yağız et al., 2018)\] [Project page](http://people.inf.ethz.ch/aksoyy/sss/).
Remaining code can be found in blabla.



# Requirement
Python 3.6, TensorFlow >= 1.4 and other common packages listed in requirements.txt.
The code has been tested on {Linux Ubuntu 16.04, TensorFlow-GPU 1.4} and {Windows 10, TensorFlow-GPU 1.8}.

# Installation
1. Install dependencies
```
pip3 install -r requirements.txt
```
2. Clone or download this repository.
3. Download the [pre-trained](http://cvg.ethz.ch/research/semantic-soft-segmentation/SSS_model.zip) model.
4. Extract the model and put the extracted "model" folder into the folder where the repository is cloned.
   - e.g., If the repository is cloned at "/project/sss", then move the model to be "/project/sss/model")
5. Run "run_extract_feat.sh", which will process sample images in the "samples" folder. If you want to run your own images, notice that image files should be the PNG formats.


# Note
** Input image files should be PNG file formats at this point. **

 

# Citation
If you use this code, please cite our paper:

```
@ARTICLE{sss,
author={Ya\u{g}{\i}z Aksoy and Tae-Hyun Oh and Sylvain Paris and Marc Pollefeys and Wojciech Matusik},
title={Semantic Soft Segmentation},
journal={ACM Transactions on Graphics (Proc. SIGGRAPH)},
year={2018},
pages = {72:1-72:13},
volume = {37},
number = {4}
}
```

# Credit
The part of the base codes (the tools in the "deeplab_resnet" directory) are borrowed from [(Re-)implementation of DeepLab-ResNet-TensorFlow](https://github.com/DrSleep/tensorflow-deeplab-resnet#using-your-dataset)
Likewise, our code (the tools in "kaffe" directory) is benefited from [Caffe to TensorFlow](https://github.com/ethereon/caffe-tensorflow)

Also, our architecture is implemented on top of the base architecture, DeepLab-ResNet-101.

```
@article{CP2016Deeplab,
      title={DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs},
      author={Liang-Chieh Chen and George Papandreou and Iasonas Kokkinos and Kevin Murphy and Alan L Yuille},
      journal={arXiv:1606.00915},
      year={2016}
    }
```

