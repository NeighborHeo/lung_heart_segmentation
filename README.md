# Unet and Unet++: multiple classification using Pytorch

This repository contains code for a multiple classification image segmentation model based on [UNet](https://arxiv.org/pdf/1505.04597.pdf) and [UNet++](https://arxiv.org/abs/1807.10165)


## Usage

#### Note : Use Python 3

### Dataset
make sure to put the files as the following structure:
```
dataset/rayence
├── raw
|   ├── valid
|   |   ├── CXR
|   |   |   ├── 0001.jpg
|   |   |   ├── 0002.jpg
|   |   |   └── ...
|   |   └── Mask
|   |       ├── 0001.bmp
|   |       ├── 0002.bmp
|   |       └── ...
│   └── test
|       ├── CXR
|       |   ├── 0001.jpg
|       |   ├── 0002.jpg
|       |   └── ...
|       └── Mask
|           ├── 0001.bmp
|           ├── 0002.bmp
|           └── ...
└── processed
    ├── valid
    |   ├── CXR
    |   └── Mask
    └── test
        ├── CXR
        └── Mask
```
mask is a single-channel category index. For example, your dataset has three categories, mask should be 8-bit images with value 0,1,2 as the categorical value, this image looks black.

### Training
```bash
python train.py
```

### inference
```bash
python inference.py -m ./dataset/rayence/checkpoints/epoch_100.pth -i ./dataset/rayence/processed/test/CXR -o ./dataset/rayence/processed/test/pred
# python inference.py -m ./data/checkpoints/epoch_10.pth -i ./data/test/input -o ./data/test/output
```
If you want to highlight your mask with color, you can
```bash
python inference_color.py -m ./dataset/rayence/checkpoints/epoch_100.pth -i ./dataset/rayence/processed/test/CXR -o ./dataset/rayence/processed/test/pred
# python inference_color.py -m ./data/checkpoints/epoch_10.pth -i ./data/test/input -o ./data/test/output
```

## Tensorboard
You can visualize in real time the train and val losses, along with the model predictions with tensorboard:
```bash
tensorboard --logdir=runs
```

