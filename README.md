# Spacecraft Pose Estimation Based on Unsupervised Domain Adaptation and on a 3D-Guided Loss Combination

Official PyTorch implementation of [Spacecraft Pose Estimation Based on Unsupervised Domain Adaptation and on a 3D-Guided Loss Combination](https://arxiv.org/abs/2212.13415) by Juan Ignacio Bravo Pérez-Villar, Álvaro García-Martín and Jesús Bescós.

<p align="center">
    <img src="https://user-images.githubusercontent.com/22771127/185179617-e77acf05-2f93-45dc-9d2d-a9d771e48d0b.png" alt="Deimos Space Logo" style="width:15%"/>
    <img src="https://user-images.githubusercontent.com/22771127/185183738-692554f7-548b-4192-a50f-9dd2af2d4b9d.png"  alt="VPU Lab Logo" style="width:15%"/>
    <img src="https://user-images.githubusercontent.com/22771127/189942036-58e17f72-a385-4955-be07-f347e109eaba.png"  alt="VPU Lab Logo" style="width:15%"/>
</p>

If you find this work useful, please consider citing:
```
@inproceedings{perez2023spacecraft,
  title={Spacecraft Pose Estimation Based on Unsupervised Domain Adaptation and on a 3D-Guided Loss Combination},
  author={P{\'e}rez-Villar, Juan Ignacio Bravo and Garc{\'\i}a-Mart{\'\i}n, {\'A}lvaro and Besc{\'o}s, Jes{\'u}s},
  booktitle={Computer Vision--ECCV 2022 Workshops: Tel Aviv, Israel, October 23--27, 2022, Proceedings, Part I},
  pages={37--52},
  year={2023},
  organization={Springer}
}
```

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
- Download from [OneDrive](https://dauam-my.sharepoint.com/:u:/g/personal/juanignacio_bravo_estudiante_uam_es/EafYSGNOw8VFqfpslgLAifIB3pX_gYJLsDld35t_ax4Z6Q?e=YmlYcw)


#### **2.1.2. Generate the heatmaps**

TBW 

#### **2.1.3. Keypoints**
Place the keypoints file "kpts.mat" into the speed_root folder

### 2.2. Clone Repository and create a Conda environment
To clone the repository, type in your terminal:

```
git clone https://github.com/JotaBravo/spacecraft-uda.git
```

After [instaling conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) go to the spacecraft-uda folder and type in your terminal:

``` 
conda env create -f env.yml
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

The script will take the initial configuration file and the training weights associated to that training file to generate pseudo-labels and train a new model. Every iteration a new configuration file is generated automatically so the results are not overwritten.

#### 3.2.1 Create the config file

To train the pseudo-labelling loop you first need to configure the "main_loop.py" script by specifying the path to the folder where the configuration files will be stored, the initial configuration file and the number of iterations. In each iteration a new configuration file will be created in the BASE_CONFIG folder with an increased niter counter. For example you first create the folder "configs_loop_sunlamp_10_epoch" and place the config file "loop_sunlamp_niter_0000.json" under it. For the next iteration of the pseudolabelling a new configuration file loop_sunlamp_niter_0001.json will be created.

```python
NITERS      = 100
BASE_CONFIG = "configs_loop_sunlamp_10_epoch" # folder path
BASE_FILE   = "loop_sunlamp_niter_0000.json"
```

#### 3.2.2 Place the first checkpoint

After you have crated the configuration files, you will need to manually place the weights used for the first iteration of the pseudo-labelling process. Under the "results" folder create a folder with the BASE_CONFIG name, and then another subfolder with the BASE_FILE name. For example "results/configs_loop_sunlamp_10_epoch/loop_sunlamp_niter_0000.json". Under that folder place a new subfolder called "ckpt" containing a file of weights named "init.pth". The final result should look as "results/configs_loop_sunlamp_10_epoch/loop_sunlamp_niter_0000.json/ckpt/init.pth"

The init.pth should be the weights of the model trained over the synthetic domain. If you want to skip that training phase you can use our available weights in Section 5 of this page.

### 3.2.3 Create the Sunlamp and Lightbox train folders
Go to the folder where you have the dataset saved and duplicate the Sunlamp and Lightbox folders, renaming the new ones as "sunlamp_train" and "lightbox_train". In these folders the new pseudo-labels will be stored and generated.

## 4. Use tensorboard to observe the training process

You can monitor the training process via [TensorBoard](https://www.tensorflow.org/tensorboard) by typing in the command line:

```
tensorboard --logdir="path to your logs folder"
```
<p align="center">
    <img src="https://user-images.githubusercontent.com/22771127/186684570-d866c48f-d5f8-4f51-9407-dd9b20c248c8.png" alt="TensorBoard output"  width="640""/>
</p>

## 5. Training weights available at:
- [GoogleDrive](https://drive.google.com/drive/folders/1HWrCEsd4K5-9_6M-7jK6yC9TjgfB79Rg?usp=share_link)


## Acknowledgment
This work is supported by Comunidad Autónoma de Madrid (Spain) under the Grant IND2020/TIC-17515
