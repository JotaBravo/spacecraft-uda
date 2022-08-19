# Spacecraft Pose Estimation Based on Unsupervised Domain Adaptation and on a 3D-Guided Loss Combination

Official PyTorch implementation of "Spacecraft Pose Estimation Based on Unsupervised Domain Adaptation and on a 3D-Guided Loss Combination" by Juan Ignacio Bravo Pérez-Villar, Álvaro García-Martín and Jesús Bescós.

<p align="center">
    <img src="https://user-images.githubusercontent.com/22771127/185179617-e77acf05-2f93-45dc-9d2d-a9d771e48d0b.png" alt="Deimos Space Logo" style="width:15%"/>
    <img src="https://user-images.githubusercontent.com/22771127/185183738-692554f7-548b-4192-a50f-9dd2af2d4b9d.png"  alt="VPU Lab Logo" style="width:15%"/>
</p>

If you find this work useful, please consider citing... 

## 1 - Summary
This paper presents the second ranking solution to the [Kelvins Pose Estimation 2021 Challenge](https://kelvins.esa.int/pose-estimation-2021). The proposed solution has ranked second in both Sunlamp and Lightbox categories, with the best total average error over the two datasets. 

The main contributions of the paper are:

- A spacecraft pose estimation algorithm that incorporates 3D structure information during training, providing robustness to intensity based domain-shift.
- An unsupervised domain adaptation scheme based on robust pseudo-label generation and self-training.

The proposed architecture with the losses incorporating the 3D information are depicted in the following figure:
<p align="center">
<img src="https://user-images.githubusercontent.com/22771127/185176722-d28f7e5b-082d-4145-919a-3e1d3198f064.png" width="640">
</p>

## 2. Setup

This section contains the instructions to execute the code. The repository has been tested in a system with:
- Ubuntu 18.04
- CUDA 11.2
- Conda 4.8.3

### 2.1. Download the datasets and generate the heatmaps

You can download the original [SPEED+](https://arxiv.org/abs/2110.03101) dataset from [Zenodo](https://zenodo.org/record/5588480#.Yv3x9S7P0aY). The dataset has the following structure:

<details>
  <summary>Dataset structure (click to open)</summary>
  
```
speedplus
│   LICENSE.md
│   camera.json  # Camera parameters 
│
└───synthetic
│   │   train.json
│   │   validation.json
│   │
│   └───images
│       │   img000001.jpg
│       │   img000002.jpg
│       │   ...
│   
└───sunlamp
│   │   test.json
│   │
│   └───images
│       │   img000001.jpg
│       │   img000002.jpg
│       │   ...
│   
└───lightbox
│   │   test.json
│   │
│   └───images
│       │   img000001.jpg
│       │   img000002.jpg
│       │   ...

```
</details>

SPEED+ provides the ground-truth information as pairs of images and poses (relative position and orientation of the spacecraft w.r.t the camera). Our method assumes the ground-truth is provided as key-point maps. We generate the key-point maps prior to the training to improve the speed. You can choose to download our computed key-points or create them manually.

#### **2.1.1. Download the heatmaps (recommended)**

Download and decompress the kptsmap.zip file. Place the kptsmap folder under the synthetic folder of the speedplus dataset.

- Download from [Mega](https://mega.nz/file/oa9B0IiI#gMe7gaU1-vs1ZrFvLGAL3SXnQn9T9IENNXhJqJSQmeY)
- Download from [OneDrive](https://dauam-my.sharepoint.com/:u:/g/personal/juanignacio_bravo_estudiante_uam_es/EafYSGNOw8VFqfpslgLAifIB3pX_gYJLsDld35t_ax4Z6Q?e=e2G9u0)


#### **2.1.2. Generate the heatmaps**

TBW 

### 2.2. Clone Repository and create a Conda environment
To clone the repository, type in your terminal:

```
git clone https://github.com/JotaBravo/spacecraft-uda.git
```

After [instaling conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) go to the spacecraft-uda folder and type in your terminal:

``` 
conda env create -f Config/Conda_Env.yml
conda activate spacecraft-uda
```

## 3. Training process

### 3.1 Train a baseline model

The training process is controlled with configuration files defined in .json. You can find example configuration files under the folder "configs/". 

To train a model simply modify the configuration file with your required values. **NOTE:** The current implementation only supports squared images.

<details>
  <summary>Configuration example (click to open)</summary>

```json
{
    "root_dir"         : "path to your datasets",
    "path_pretrain"    : "path to your pretrained weights",  # Put "" for no weight initalization
    "path_results"     : "./results",
    "device"           : "cuda",

    "start_epoch"      :0,      # Starting epoch
    "total_epochs"     :20,     # Number of total epochs (N-1)
    "save_tensorboard" :100,    # Number of steps to save to tensorboard
    "save_epoch"       :5,      # Save every number of epochs
    "save_optimizer"   :false,  # Flag to save or not the optimzer

    "mean"     :41.3050, # Mean value of the training dataset
    "std"      :37.0706, # Standard deviation of training the dataset   
    "mean_val" :41.1280, # Mean value of the validation dataset
    "std_val"  :36.9064, # Mean value of the validation dataset    

    "batch_size"      :8,  # Batch size to input the GPU during training
    "batch_size_test" :1,  # Batch size to input the GPU during test
    "num_stacks"      :2,  # Number of stacks of the hourglass network
    "lr"              :2.5e-4, # Learning rate

    "num_workers"   :8,    # Number of CPU workers (might fail in Windows)
    "pin_memory"    :true, 
    "rows"          :640,  # Resize input image rows (currently only supporting rows=cols)
    "cols"          :640,  # Resize input image cols (currently only supporting rows=cols)

    "alpha_heatmap":10, # Flag to activate pnp loss

    "activate_lpnp":true, # Flag to activate pnp loss
    "activate_l3d": true, # Flag to activate 3D loss
    "weigth_lpnp": 1e-1,  # Weight of the PnP loss
    "weigth_l3d": 1e-1,   # Weight of the 3D loss

    "isloop":false # Flag to true if training with pseudo-labels, false otherwise
}
```
</details>

Then, after properly modifying the configuration file under the repository folder type:

``` 
python main.py --cfg "configs/experiment.json"
``` 

### 3.2 Train a model with pseudo-labels

TBW

## 4. Use tensorboard to observe the training process



## Acknowledgment

## References
