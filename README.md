### Point2Sequence: *Learning the Shape Representation of 3D Point Clouds with an Attention-based Sequence to Sequence Network*
Created by <a href="https://scholar.google.com/citations?user=vg2IvzsAAAAJ&hl=en" target="_blank">Xinhai Liu</a>, <a href="https://scholar.google.com/citations?user=RGNWczEAAAAJ&hl=en" target="_blank">Zhizhong Han</a>, <a href="http://cgcad.thss.tsinghua.edu.cn/liuyushen/" target="_blank">Yu-Shen Liu</a>, <a href="https://scholar.google.com/citations?user=KW0FmzgAAAAJ&hl=en" target="_blank">Matthias Zwicker</a>.

![prediction example](https://github.com/liuxinhai/Point2Sequence/blob/master/doc/architecture.jpg)

### Citation
If you find our work useful in your research, please consider citing:

        @inproceedings{liu2019point2sequence,
          title={Point2Sequence: Learning the Shape Representation of 3D Point Clouds with an Attention-based Sequence to Sequence Network},
          author={Liu, Xinhai and Han, Zhizhong and Liu, Yu-Shen and Zwicker, Matthias},
          booktitle={Thirty-Third AAAI Conference on Artificial Intelligence},
          year={2019}
        }

### Introduction
In Point2Sequence, we build the multi-scale areas in the local region of point sets by a sequential manner.
To explore the correlation between different scale areas, a RNN-based sequence model is employed to capture the contextual information inside local regions.
In addition, we also introduce an attention mechanism to highlight the importance different scale areas. 

In this repository we release code our Point2Sequence classification and segmentation networks as well as a few utility scripts for training, testing and data processing.

### Installation

Install <a href="https://www.tensorflow.org/install/">TensorFlow</a>. The code is tested under TF1.4 GPU version and Python 2.7 on Ubuntu 16.04. There are also some dependencies for a few Python libraries for data processing like `cv2`, `h5py` etc. It's highly recommended that you have access to GPUs.
Before running the code, you need to compile customized TF operators as described in <a href="https://github.com/charlesq34/pointnet2/">PointNet++</a>.
### Usage

#### Shape Classification

To train a Point2Sequence model to classify ModelNet40 shapes (using point clouds with XYZ coordinates):

        python train.py

To see all optional arguments for training:

        python train.py -h

In the training process, we also evaluate the performance the model.

#### Shape Part Segmentation

To train a model to segment object parts for ShapeNet models:

        cd part_seg
        python train.py
#### Prepare Your Own Data
Follow the dataset in PointNet++, you can refer to <a href="https://github.com/charlesq34/3dmodel_feature/blob/master/io/write_hdf5.py">here</a> on how to prepare your own HDF5 files for either classification or segmentation. Or you can refer to `modelnet_dataset.py` on how to read raw data files and prepare mini-batches from them.
### License
Our code is released under MIT License (see LICENSE file for details).

### Related Projects

* <a href="https://arxiv.org/abs/1706.02413" target="_blank">PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space</a> by Qi et al. (NIPS 2017)
* <a href="http://openaccess.thecvf.com/content_cvpr_2018/html/Xie_Attentional_ShapeContextNet_for_CVPR_2018_paper.html" target="_blank">Attentional ShapeContextNet for Point Cloud Recognition</a> by Xie et al. (CVPR 2018)
