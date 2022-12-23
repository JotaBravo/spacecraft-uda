import argparse
import os
import pathlib
import warnings

import kornia
import kornia.augmentation as K
import numpy as np
import roma
import torch
import torch.nn.functional as F

from kornia.geometry.conversions import QuaternionCoeffOrder
from torch.utils import tensorboard
from torchvision import transforms
from tqdm import tqdm

from BPnP import BPnP
from loaders import speedplus_segmentation_precomputed as speedplus
from models import large_hourglass
from src import utils


np.set_printoptions(suppress=True)
warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------------- #
#     Initialization            #
# ----------------------------- #

parser = argparse.ArgumentParser(description="Spacecraft Pose Estimation Based on Unsupervised Domain Adaptation and on a 3D-Guided Loss Combination")
parser.add_argument("-c", "--cfg", metavar="DIR", help="Path to the configuration file", required=True)
args = parser.parse_args()

# Parse the config file
config = utils.load_config(args.cfg)
device = config["device"]

# Create the direcctories for the tensorboard logs
path_logs = os.path.join(config["path_results"],args.cfg,"logs")
pathlib.Path(path_logs).mkdir(parents=True, exist_ok=True)

# Create the direcctories for the weight checkpoints
path_checkpoints = os.path.join(config["path_results"],args.cfg,"ckpt")
pathlib.Path(path_checkpoints).mkdir(parents=True, exist_ok=True)

# Instantiate the tensorboard writer
writer  = tensorboard.writer.SummaryWriter(path_logs)

# Instantiate the network. Two heads, one for key-points the other for depth
heads = {'hm_c':11, 'depth': 11}
hourglass  = large_hourglass.get_large_hourglass_net(heads, config["num_stacks"]).to(device)

# If we're training in a loop (for pseudo-labels) we automatically
# load the weights from the previous iteration
if config["isloop"]:
    id_checkpoint = int(args.cfg.split("_niter_")[-1].split(".json")[0])-1
    id_checkpoint = str(id_checkpoint).zfill(4) + ".json"

    path_pretrain = args.cfg.split("_niter_")[0] + "_niter_" +id_checkpoint
    path_pretrain = os.path.join(config["path_results"],path_pretrain,"ckpt", "init.pth")

    config["path_pretrain"] = path_pretrain

# Load pretrained weights
if config["path_pretrain"]:
    model_dict = torch.load(config["path_pretrain"])

    if not config["isloop"]:
        # Be careful here, strict is set to false because we have added two heads.
        # Problem is that it won't trhow an error if none of the weights are initalized.
        # For loading models trained in several gpus uncomment the following two lines.

        #model_dict = model_dict["state_dict"]
        #model_dict = OrderedDict((k.split("module.")[1], v) for k, v in model_dict.items())
        hourglass.load_state_dict(model_dict,strict=False)

    else:
        hourglass.load_state_dict(model_dict,strict=True)

print("\n--------------  Training started  -------------------\n")
print("  -- Using config from:\t", args.cfg)
print("  -- Using weights from:\t", config["path_pretrain"])
print("  -- Saving weights to:\t", path_checkpoints)
print("\n-----------------------------------------------------\n")

# ----------------------------- #
#           Transforms          #
# ----------------------------- #

# These are applied in the data loader
tforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((config["rows"], config["cols"]))
    ])

# These are applied in the torch tensor. Normalize by the mean, blur some images and add noise
aug_intensity = K.AugmentationSequential(
     K.Normalize(mean=config["mean"]/255.0,std=config["std"]/255.0),
     K.RandomGaussianBlur(kernel_size=[15,15],sigma=[0.8,0.8],p=0.1),
     K.RandomGaussianNoise(p=1,std=0.005)
)

aug_intensity_val = K.AugmentationSequential(
     K.Normalize(mean=config["mean_val"]/255.0,std=config["std_val"]/255.0),
     K.RandomGaussianBlur(kernel_size=[15,15],sigma=[0.8,0.8],p=0.1),
     K.RandomGaussianNoise(p=1,std=0.005)
)

# ----------------------------- #
#           Loaders             #
# ----------------------------- #

if config["isloop"]:
    if config["split_submission"] == "sunlamp":
        train_dataset = speedplus.PyTorchSatellitePoseEstimationDataset(split="sunlamp_train",
                        speed_root=config["root_dir"], transform_input=tforms, config=config)
    if config["split_submission"] == "lightbox":
        train_dataset = speedplus.PyTorchSatellitePoseEstimationDataset(split="lightbox_train",
                        speed_root=config["root_dir"], transform_input=tforms, config=config)

else:
    train_dataset = speedplus.PyTorchSatellitePoseEstimationDataset(split="train",
    speed_root=config["root_dir"], transform_input=tforms, config=config)

val_dataset = speedplus.PyTorchSatellitePoseEstimationDataset(split="validation",
              speed_root=config["root_dir"], transform_input=tforms, config=config)

train_loader  = torch.utils.data.DataLoader(train_dataset,
                batch_size=config["batch_size"],
                shuffle=True,
                num_workers=config["num_workers"],
                drop_last=True,
                pin_memory=True)

val_loader    = torch.utils.data.DataLoader(val_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=config["num_workers"],
                drop_last=False,
                pin_memory=False)


# ----------------------------- #
#        Optimizer/Loss         #
# ----------------------------- #

optim_params = [ {"params": hourglass.parameters(),
                  "lr": config["lr"]}]
optimizer = torch.optim.Adam(optim_params)
mse_loss = torch.nn.MSELoss()

# ----------------------------- #
#        Load Data              #
# ----------------------------- #

k_mat_input = utils.get_kmat_scaled(config, device) # Intrinsic matrix
dist_coefs  = utils.get_coefs(config, device) # Distortion coefficients
kpts_world  = utils.get_world_kpts(config,device) # Spacecraft key-points

# Instatiate Backpropagatable PnP
pnp_fast   = BPnP.BPnP_fast.apply

# Normalization factor
# TODO: Change the normalization factor for more image sizes (not only squared ones)
norm_factor = (float(config["rows"]) - 1)

# Dictionary used to update writer
dict_writer = {}
dict_writer["kpts_world"]  = kpts_world[0]
dict_writer["k_mat_input"] = k_mat_input[0]

best_val_score = 1e16

# ----------------------------- #
#        Train/Val Loop         #
# ----------------------------- #

for epoch in range(config["start_epoch"], config["total_epochs"]):
    print("Epoch: ", epoch, "\n")
    hourglass.train(True)

    # ----------------------------- #
    #        Train Epoch            #
    # ----------------------------- #
    for i, data in enumerate(tqdm(train_loader, ncols=50)):
        # Load the data
        img_source = data["image"].to(device)
        heatmap_gt = data["heatmap"].to(device)
        kpts_gt    = data["kpts_2Dim"].to(device)
        kpts_gt3d  = data["kpts_3Dcam"].to(device)
        kpts_vis   = data["visible_kpts"].to(device)

        q_gt = data["q0"].to(device)
        t_gt = data["r0"].to(device)
        R_gt = kornia.geometry.quaternion_to_rotation_matrix(q_gt, QuaternionCoeffOrder.WXYZ)

        # Augmentate the intensity of the input image
        # (if we augmentate the shape, pose-based losses won't work)
        img_source = aug_intensity(img_source)

        # Remove distortion
        img_source = kornia.geometry.undistort_image(img_source, k_mat_input, dist_coefs)

        # Update dictionary writer
        dict_writer["img_source"] = img_source
        dict_writer["heatmap_gt"] = heatmap_gt
        dict_writer["kpts_gt"]    = kpts_gt

        # Obtain the prediction
        output = hourglass(img_source) #BxNKPTSxROWSxCOLS

        # Initialize variables
        total_loss = 0
        loss_hm  = 0
        loss_pnp = 0
        loss_3d  = 0
        rot_err  = 0
        tra_err  = 0

        for level_id, level in enumerate(output):
            # Interpolate network output to input resolution
            heatmap_pred = F.interpolate(level["hm_c"], size=(config["rows"],config["cols"]),
                                                        mode='bilinear',
                                                        align_corners=False)

            depth_pred = F.interpolate(level["depth"], size=(config["rows"],config["cols"]),
                                                       mode='bilinear',
                                                       align_corners=False)

            # Convert the heatmap to points
            kpts_pred  = utils.heatmap_to_points(heatmap_pred)

            # ---------------------------------------------------------------------------- #
            #   Point-n-Perspective Loss.                                                  #
            #   We employ the BPnP algorithm from (https://arxiv.org/pdf/1909.06043.pdf)   #
            # ---------------------------------------------------------------------------- #
            kpts_gt_norm = kpts_gt/norm_factor

            if config["activate_lpnp"]:
                rt_source = pnp_fast(kpts_pred, kpts_world[0], k_mat_input[0]) # Bx6 [rot,pose]
                kpts_backprojected = BPnP.batch_project(rt_source, kpts_world[0], k_mat_input[0])

                # Clip the key-points to be in the maximum image range (avoid loss explosions)
                kpts_pred_norm = torch.clip(kpts_pred,min=0.0,max=norm_factor)/norm_factor

                kpts_backprojected_norm = torch.clip(kpts_backprojected,min=0.0,max=norm_factor)
                kpts_backprojected_norm = kpts_backprojected_norm/norm_factor

                # Compute the PnP-based loss
                loss_pnp = mse_loss(kpts_backprojected_norm, kpts_gt_norm)  + \
                           mse_loss(kpts_backprojected_norm, kpts_pred_norm)

            # ---------------------------------------------------------------------------- #
            #   3D aligment loss                                                           #
            # ---------------------------------------------------------------------------- #
            if config["activate_l3d"]:
                depth_points = kpts_gt_norm.unsqueeze(1)

                # Sample the depth at the predicted kpts location
                kpts_depths = []

                for kptid in range(11):
                    depth_grid = depth_pred[:,kptid,:,:].unsqueeze(1)
                    grid =  depth_points[:,:,kptid,:].unsqueeze(2)
                    pts = utils.point_sample(depth_grid,grid).squeeze()
                    kpts_depths.append(pts)

                kpts_depths = torch.stack(kpts_depths,dim=1).unsqueeze(2)

                # Project the points into the 3D space
                intrinsics = k_mat_input.unsqueeze(1)
                kpts_3d_depth = kornia.geometry.unproject_points(kpts_gt, kpts_depths, intrinsics)

                # Estimate the rigid rotation (https://arxiv.org/abs/2103.16317)
                Ro,to  = roma.rigid_points_registration(kpts_world, kpts_3d_depth)

                # For visualization
                ro_vec = kornia.geometry.conversions.rotation_matrix_to_angle_axis(Ro)

                # Compute the 3D loss term
                rot_err = mse_loss(Ro, R_gt)
                tra_err = mse_loss(to, t_gt)
                loss_3d = rot_err + tra_err

            # ---------------------------------------------------------------------------- #
            #   Heatmap loss                                                               #
            # ---------------------------------------------------------------------------- #

            for batch_index in range(heatmap_pred.shape[0]):
                # Get visible key-points
                flag_vis = kpts_vis[batch_index]

                # Get the batch predictions and ground-truth
                pred_batch = heatmap_pred[batch_index,flag_vis,:,:]
                gt_batch = heatmap_gt[batch_index,flag_vis,:,:]

                loss_hm += mse_loss(pred_batch, 100*gt_batch)/10

            # Total Loss
            total_loss +=  loss_hm + 1e-2*loss_pnp + 1e-2*loss_3d

            # Update tensorboard logs
            if not i%config["save_tensorboard"]:
                dict_writer["heatmap",level_id] = heatmap_pred
                dict_writer["depth",level_id] = depth_pred
                dict_writer["loss_hm",level_id] = loss_hm.item()
                dict_writer["total_loss",level_id] = total_loss.item()

                if config["activate_lpnp"]:

                    dict_writer["kpts_pnp",level_id] = kpts_backprojected
                    dict_writer["loss_pnp",level_id] = loss_pnp.item()

                if config["activate_l3d"]:

                    dict_writer["poses_3d",level_id] = torch.cat((ro_vec,to),dim=1)
                    dict_writer["loss_3d",level_id] = loss_3d.item()
                    dict_writer["rot_err",level_id] = rot_err.item()
                    dict_writer["tra_err",level_id] = tra_err.item()

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()


        if not i%config["save_tensorboard"]:

            utils.update_writer(writer,dict_writer, i + len(train_loader)*epoch, config)

    #----------------------------------------------------------------------------------------------
    #                                                   Eval Loop
    #----------------------------------------------------------------------------------------------
    if not config["isloop"]:
        with torch.no_grad():
            hourglass.eval()
            val_score, val_score_t, val_score_r = utils.eval_loop(val_loader,
                                                    aug_intensity_val,
                                                    hourglass,
                                                    kpts_world,
                                                    k_mat_input,
                                                    device,
                                                    config)

            writer.add_scalar("Validation Pose Score",  val_score, epoch)
            writer.add_scalar("Validation Translation Score",  val_score_t, epoch)
            writer.add_scalar("Validation Rotation Score",  val_score_r, epoch)
            print("Validation Score: \n", val_score)

            if val_score < best_val_score or not epoch%config["save_epoch"]:
                best_val_score = val_score

                string_model = "epoch_" + str(epoch) + "_" + str(best_val_score) + "model_seg.pth"
                torch.save(hourglass.state_dict(),  os.path.join(path_checkpoints, string_model))

                if config["save_optimizer"]:
                    string_optimizer = "epoch_" + str(epoch) + "_" + str(best_val_score) + "optimizer.pth"
                    torch.save(optimizer.state_dict(),  os.path.join(path_checkpoints, string_optimizer))

            if epoch+1 == config["total_epochs"]:
                torch.save(hourglass.state_dict(),  os.path.join(path_checkpoints, "last_epoch_" + str(epoch) +"model_seg.pth"))
    else:
        torch.save(hourglass.state_dict(),  os.path.join(path_checkpoints, "init.pth"))
