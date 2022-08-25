from models import large_hourglass
from src import utils

import argparse
import kornia
import os
import pathlib
import roma
import time
import torch

import torch.nn.functional as F
import kornia.augmentation as K

from collections import OrderedDict
from tqdm import tqdm
from torchvision import transforms
from torch.utils import tensorboard

from BPnP import BPnP
from loaders import speedplus_segmentation_precomputed as speedplus
from models import large_hourglass
from src import utils

import warnings
warnings.filterwarnings("ignore", category=UserWarning) 
#to remove
import numpy as np
import cv2
np.set_printoptions(suppress=True)

parser = argparse.ArgumentParser(description='Pose Estimation')
parser.add_argument('--cfg', '--config', metavar='DIR', help='Path to the configuration', required=True)


args = parser.parse_args()
config = utils.load_config(args.cfg)

path_checkpoints = os.path.join(config["path_results"],args.cfg,"ckpt", "init.pth")
path_json =  os.path.join(config["path_results"],args.cfg,"train.json")
config["path_pretrain"] = path_checkpoints

# ----------------------------- #
#           Networks            #
# ----------------------------- #
device = config["device"]
# Two heads, one for key-points the other for depth
heads      = {'hm_c':11, 'depth': 11}    
hourglass  = large_hourglass.get_large_hourglass_net(heads, config["num_stacks"]).to(device)



print("\n--------------  Generating Label Started  -------------------\n")
print("  -- Using config from:", args.cfg)
print("  -- Using weights from:", config["path_pretrain"])
print("\n-----------------------------------------------------\n")


if config["path_pretrain"] is not None:

    model_dict = torch.load(config["path_pretrain"])
    #model_dict = model_dict["state_dict"]
    #model_dict = OrderedDict((k.split("module.")[1], v) for k, v in model_dict.items())    
    hourglass.load_state_dict(model_dict,strict=True)


# ----------------------------- #
#     Datasets  & Transforms    #
# ----------------------------- #

# These are applied in the data loader
tforms = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Resize((config["rows"], config["cols"]))
    ])


aug_intensity_test = K.AugmentationSequential(
     K.Normalize(mean=config["mean"]/255.0,std=config["std"]/255.0),
     K.RandomGaussianBlur(kernel_size=[15,15],sigma=[0.8,0.8],p=0.1),
     K.RandomGaussianNoise(p=1,std=0.005)
)


test_dataset = speedplus.PyTorchSatellitePoseEstimationDataset(split=config["split_submission"], speed_root=config["root_dir"], transform_input=tforms, config=config)
test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=config["batch_size_test"], shuffle=False,  num_workers=config["num_workers"], drop_last=False,  pin_memory=True)


# ----------------------------- #
#        Optimizer  & Losses    #
# ----------------------------- #


# Intrinsics matrices
k_mat_input = utils.get_kmat_scaled(config, device)
kpts_world  = utils.get_world_kpts(config,device)

# Distortion coefficients
dist_coefs  = utils.get_coefs(config, device)

# Normalization factor
# TODO: Change the normalization factor for more image sizes (not only squared ones)
norm_factor = (float(config["rows"]) - 1)

# Instatiate Backpropagatable PnP
pnp_fast   = BPnP.BPnP_fast.apply

# dictionary used to update writer
dict_writer = dict()
dict_writer["kpts_world"]  = kpts_world[0]
dict_writer["k_mat_input"] = k_mat_input[0]

with torch.no_grad():
    hourglass.eval()
    utils.generate_json_loop(test_loader, aug_intensity_test, hourglass, kpts_world, k_mat_input, device, config, path_json)
