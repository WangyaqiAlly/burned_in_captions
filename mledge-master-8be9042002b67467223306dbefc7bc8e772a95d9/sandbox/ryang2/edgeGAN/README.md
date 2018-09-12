## Introduction
This is a simulation of edge-center GAN.
In this version, the edge `site_s` learn a model based on all the images
 under folder `time_t` and save its model record for each `time_t`.
For each `time_t`, all the edge nodes will learn the model in turn,
utilizing the whole GPU.

The center node could specify which `time_t` record it want to use.
The center will generate images with the corresponding generator for each edge.
Afterwards, the center would utilize all the generated images to train the classifier.

*Notice:* Current version only support mnist for test.

## Pre-request
Install pip package: coloredlogs
```
pip install coloredlogs
```

Optional operation:

Add framework path to the python environment.
```
echo 'export PYTHONPATH="${PYTHONPATH}:'$PWD'"' >> ~/.bashrc
source ~/.bashrc
```

The example code in under the folder `./example`

## How to run
#### Edges:
```
CUDA_VISIBLE_DEVICES='1' python labelgan_multiedge.py
```

#### Center
```
CUDA_VISIBLE_DEVICES='1' python test_center.py
```

## Essential/Possible modification for new network
1) Create a new subclass for edgenode

2) Visualization function in Utils.py

3) Generate_fake_images function in ImageGenerator class in Utils.py

4) Record directory (Need to change the default dir or add an option in command)