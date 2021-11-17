# Learn2Assemble

This repository is the official implementation of [Learn2Assemble with Structured Representations and Search for Robotic Architectural Construction](https://openreview.net/forum?id=wBT0lZJAJ0V) by N. Funk, G. Chalvatzaki, B. Belousov and J. Peters, which has been accepted in the Conference on Robot Learning (CoRL) 2021.
Additional video material can accesssed [here](https://sites.google.com/view/learn2assemble).

If you use code or ideas from this work for your projects or research, please cite it.

```
@inproceedings{
funk2021learnassemble,
title={Learn2Assemble with Structured Representations and Search for Robotic Architectural Construction},
author={Niklas Funk and Georgia Chalvatzaki and Boris Belousov and Jan Peters},
booktitle={5th Annual Conference on Robot Learning },
year={2021},
url={https://openreview.net/forum?id=wBT0lZJAJ0V}
}
```

## General Information

This repository implements several assembly environments in pybullet. It provides all the
environments that have been presented in the paper 
[Learn2Assemble with Structured Representations and Search for Robotic Architectural Construction](https://sites.google.com/view/learn2assemble), in which agents have been trained
using Graph Neural Networks and Q-learning for which the implementation is provided 
[here](https://github.com/nifunk/GNNMushroomRL).

## Getting Started

To get started with this repository, clone it and setup a conda environment using the file 'conda_env.yml''
by running:

```
conda env create -f conda_env.yml
``` 

Then activate the conda environment and install the package via:

```
pip install -e .
```

## Examples

Run a simple example of one of the environments, simply navigate into the examples folder and run:

```
python test_env_3D_mushroom.py
```

## More advanced experiments

As the more advanced experiments that have been presented in the paper are based on graph neural networks, they are 
provided in the same place as their implementation, which is [here](https://github.com/nifunk/GNNMushroomRL).



