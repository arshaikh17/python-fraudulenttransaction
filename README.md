

# Installation Guide

1. Keras Installation
Before installing Keras, please install one of its backend engines: TensorFlow, Theano, or CNTK. We recommend the TensorFlow backend.
    - [TensorFlow installation instructions.](https://www.tensorflow.org/install/)
    - [Theano installation instructions.](http://deeplearning.net/software/theano/install.html#install)
    - [CNTK installation instructions.](http://deeplearning.net/software/theano/install.html#install)

    You may also consider installing the following optional dependencies:
    
    - [cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/) (recommended if you plan on running Keras on GPU).
    - HDF5 and [h5py](http://docs.h5py.org/en/latest/build.html) (required if you plan on saving Keras models to disk).
    - [graphviz](https://graphviz.gitlab.io/download/) and [pydot](https://github.com/erocarrera/pydot) (used by [visualization utilities](https://keras.io/visualization/) to plot model graphs).
    
    There are two ways to install Keras:
    - Install Keras from PyPI (recommended):
        ```sh
        pip install keras
        ```
    - Alternatively, Install Keras from the GitHub source:
        ```sh
        git clone https://github.com/keras-team/keras.git
        ```
        Then move to the Keras folder and run to install command:
        ```sh
        cd keras
        python setup.py install
        ```
    
2. Other Python Libraries
We use Python pandas, numpy, sklearn libraries.
    ```sh
    pip install pandas
    pip install sklearn
    ```


 