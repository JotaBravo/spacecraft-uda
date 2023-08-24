import numpy as np
import json
import cv2
import torch
import matplotlib.pyplot as plt
import os
import kornia
from kornia.geometry.conversions import QuaternionCoeffOrder
import scipy
from loaders import speedplus_segmentation_precomputed as speedplus
import torch.nn.functional as F
from . import speedscore
from BPnP import BPnP
from tqdm import tqdm
import shutil
def load_config(config_path):
    with open(config_path) as json_file:
        return json.load(json_file)    

def tensor_to_cvmat(tensor, BGR = True):
    mat = tensor.cpu().data.numpy().squeeze()

    if (tensor.shape[0]) == 3:
        mat = np.transpose(mat, (1, 2, 0))

    min_mat = np.min(mat)
    max_mat = np.max(mat)    
    mat = (mat-min_mat)/(max_mat-min_mat)
    out = (255*mat).astype(np.uint8)
    if BGR == True:
        out = cv2.cvtColor(out,cv2.COLOR_RGB2BGR)
    return out

def normalize_image(x):
    """Rescale image pixels to span range [0, 1]
    """
    ma = float(x.max().cpu().data)
    mi = float(x.min().cpu().data)
    d = ma - mi if ma != mi else 1e5
    return (x - mi) / d

def tensor_to_mat(tensor):
    tensor = tensor.cpu().data.numpy().squeeze()
    return tensor

def build_rt_mat(R,t): 
    a = torch.cat(R.shape[0]*[torch.Tensor([0,0,0,1]).unsqueeze(0)]).unsqueeze(1).to(R.get_device())
    b = torch.cat([R,t], dim = -1)
    c = torch.cat([b,a], dim = 1)
    return c

def quat2dcm(q):

    """ Computing direction cosine matrix from quaternion, adapted from PyNav. """

    # normalizing quaternion
    q = q/np.linalg.norm(q)

    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]

    dcm = np.zeros((3, 3))

    dcm[0, 0] = 2 * q0 ** 2 - 1 + 2 * q1 ** 2
    dcm[1, 1] = 2 * q0 ** 2 - 1 + 2 * q2 ** 2
    dcm[2, 2] = 2 * q0 ** 2 - 1 + 2 * q3 ** 2

    dcm[0, 1] = 2 * q1 * q2 + 2 * q0 * q3
    dcm[0, 2] = 2 * q1 * q3 - 2 * q0 * q2

    dcm[1, 0] = 2 * q1 * q2 - 2 * q0 * q3
    dcm[1, 2] = 2 * q2 * q3 + 2 * q0 * q1

    dcm[2, 0] = 2 * q1 * q3 + 2 * q0 * q2
    dcm[2, 1] = 2 * q2 * q3 - 2 * q0 * q1

    return dcm

def project(q, r, K,dist):

        """ Projecting points to image frame to draw axes """

        # reference points in satellite frame for drawing axes
        p_axes = np.array([[0, 0, 0, 1],
                           [1, 0, 0, 1],
                           [0, 1, 0, 1],
                           [0, 0, 1, 1]])
        points_body = np.transpose(p_axes)

        # transformation to frame
        pose_mat = np.hstack((np.transpose(quat2dcm(q)), np.expand_dims(r, 1)))
        

        p_cam = np.dot(pose_mat, points_body)
        
        # getting homogeneous coordinates
        points_camera_frame = p_cam
        points_camera_frame[:,3] /=points_camera_frame[2,3]
  
        # projection to image plane
        #points_image_plane = K.dot(points_camera_frame)
        points_image_plane = points_camera_frame

        
        x0, y0 = (points_image_plane[0], points_image_plane[1])
        
        # apply distortion
        #dist = Camera.dcoef
        #dist = [1., 1., 1., 1, 1.]
        r2 = x0*x0 + y0*y0
        cdist = 1 + dist[0]*r2 + dist[1]*r2*r2 + dist[4]*r2*r2*r2
        x1  = x0*cdist + dist[2]*2*x0*y0 + dist[3]*(r2 + 2*x0*x0)
        y1  = y0*cdist + dist[2]*(r2 + 2*y0*y0) + dist[3]*2*x0*y0
        
        x = K[0,0]*x1 + K[0,2]
        y = K[1,1]*y1 + K[1,2]
        
        return x, y

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def draw_keypoints(image, keypoints, diameter=8, color = (255, 0, 0)):
    image = image.copy()
    for (x, y) in keypoints:
        cv2.circle(image, (int(x), int(y)), diameter, color, -1)
    return image

def heatmap_to_points(mask):
    return kornia.geometry.subpix.spatial_soft_argmax2d(mask,temperature=torch.tensor(50.0,requires_grad=True),normalized_coordinates=False)

def get_world_kpts(config,gpu):
    world_kpts = scipy.io.loadmat(config["root_dir"] + "kpts.mat")
    world_kpts = world_kpts["corners"].astype(np.float32).T
    world_kpts = torch.from_numpy(world_kpts).unsqueeze(0).repeat([config["batch_size"],1,1]).cuda(gpu).type(torch.float)
    return world_kpts

def load_corners(config,gpu):
    corners = scipy.io.loadmat('corners.mat')
    corners_out = {}
    corners_out["c1"] = torch.from_numpy(np.array(corners['corners1'])).unsqueeze(0).repeat([config["batch_size"],1,1]).cuda(gpu).type(torch.float)
    corners_out["c2"] = torch.from_numpy(np.array(corners['corners2'])).unsqueeze(0).repeat([config["batch_size"],1,1]).cuda(gpu).type(torch.float)
    corners_out["c3"] = torch.from_numpy(np.array(corners['corners3'])).unsqueeze(0).repeat([config["batch_size"],1,1]).cuda(gpu).type(torch.float)
    corners_out["c4"] = torch.from_numpy(np.array(corners['corners4'])).unsqueeze(0).repeat([config["batch_size"],1,1]).cuda(gpu).type(torch.float)
    corners_out["c5"] = torch.from_numpy(np.array(corners['corners5'])).unsqueeze(0).repeat([config["batch_size"],1,1]).cuda(gpu).type(torch.float)
    return corners_out

def get_coefs(config,gpu):
    cam = speedplus.Camera(config["root_dir"])
    #coef = np.array(cam.dcoef)
    #coef = np.expand_dims(coef,axis=0)
    coef   = torch.from_numpy(np.array(cam.dcoef)).unsqueeze(0).repeat([config["batch_size"],1]).cuda(gpu).type(torch.float)
    return coef

def get_kmat_scaled(config, gpu):

    cam = speedplus.Camera(config["root_dir"])
    k = cam.K
    k[0, :] *= ((config["cols"])/1920)
    k[1, :] *= ((config["rows"])/1200)               
    k   = torch.from_numpy(k).unsqueeze(0).repeat([config["batch_size"],1,1]).cuda(gpu).type(torch.float)
    return k

def denormalize(grid):
    """Denormalize input grid from range [0, 1] to [-1, 1]
    Args:
        grid (Tensor): The grid to be denormalize, range [0, 1].

    Returns:
        Tensor: Denormalized grid, range [-1, 1].
    """

    return grid * 2.0 - 1.0

def point_sample(input, points, align_corners=True, **kwargs):
    add_dim = False
    if points.dim() == 3:
        add_dim = True
        points = points.unsqueeze(2)
    output = F.grid_sample(
        input, denormalize(points), align_corners=align_corners, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output     




def update_writer(writer, input_dict, iteration, config): # ppp + len(train_loader)*epoch
    
    second_map = 0
    # lazy workaround for the case when we have only one map
    if  ("heatmap",1) in input_dict:
        second_map = 1
        


    
    with torch.no_grad():

        writer.add_scalar("Total Train Loss ",  input_dict["total_loss",0] + input_dict["total_loss",second_map], iteration)
        writer.add_scalar("Heatmap Loss ",      input_dict["loss_hm",0]      + input_dict["loss_hm" ,second_map], iteration)
        if config["activate_lpnp"]:                
            writer.add_scalar("PnP Loss ",          input_dict["loss_pnp",0]     + input_dict["loss_pnp" ,second_map], iteration)
        if config["activate_l3d"]:                
            writer.add_scalar("3D  Loss ",          input_dict["loss_3d",0]      + input_dict["loss_3d" ,second_map],  iteration)
            writer.add_scalar("Rotation  Loss ",    input_dict["rot_err",0]      + input_dict["rot_err" ,second_map],  iteration)
            writer.add_scalar("Translation  Loss ", input_dict["tra_err",0]      + input_dict["tra_err" ,second_map],  iteration)

        image = tensor_to_cvmat(input_dict["img_source"].sum(1,True)[0])
        image = draw_keypoints(image,tensor_to_mat(input_dict["kpts_gt"][0]),color=(0,255,0))
        image = cv2.resize(image,(200,200))

        hm_gt  = cv2.resize(tensor_to_cvmat(input_dict["heatmap_gt"].sum(1,True)[0]),(200,200))

        hm_c_0 = tensor_to_cvmat(input_dict["heatmap",0].sum(1,True)[0])

        if config["activate_lpnp"]:                
            hm_c_0 = draw_keypoints(hm_c_0,tensor_to_mat(input_dict["kpts_pnp",0][0]),color=(0,255,255))

        hm_c_0 = cv2.resize(hm_c_0,(200,200))

        hm_c_1 = tensor_to_cvmat(input_dict["heatmap",second_map].sum(1,True)[0])
        if config["activate_lpnp"]:                
            hm_c_1 = draw_keypoints(hm_c_1,tensor_to_mat(input_dict["kpts_pnp",second_map][0]),color=(0,255,255))
        hm_c_1 = cv2.resize(hm_c_1,(200,200))
        if config["activate_l3d"]:                

            kpts_3d_1 = BPnP.batch_project(input_dict["poses_3d",0], input_dict["kpts_world"] , input_dict["k_mat_input"])  
            kpts_3d_2 = BPnP.batch_project(input_dict["poses_3d",second_map], input_dict["kpts_world"] , input_dict["k_mat_input"])  


        depth_0 = tensor_to_cvmat(input_dict["depth",0].sum(1,True)[0])
        if config["activate_l3d"]:                
            depth_0 = draw_keypoints(depth_0,tensor_to_mat(kpts_3d_1[0]),color=(0,255,255))
        depth_0 = cv2.resize(depth_0,(200,200))
        depth_1 = tensor_to_cvmat(input_dict["depth",second_map].sum(1,True)[0])
        if config["activate_l3d"]:                
            depth_1 = draw_keypoints(depth_1,tensor_to_mat(kpts_3d_2[0]),color=(0,255,255))
        depth_1 = cv2.resize(depth_1,(200,200))                        
        a = np.vstack((image, hm_gt))
        b = np.vstack((hm_c_0, hm_c_1))
        c = np.vstack((depth_0,depth_1))
        d = np.hstack((a,b,c))
        writer.add_image("GT -- Heatmap + PnP -- Depth + 3D", d ,iteration,dataformats='HWC')



def eval_loop(val_loader ,aug_intensity_val, hourglass, kpts_world, k_mat_input, device, config):

    cam = speedplus.Camera(config["root_dir"])

    speed_total = 0
    speed_t_total = 0
    speed_r_total = 0
    kpts_world = tensor_to_mat(kpts_world[0])
    k_mat_input = tensor_to_mat(k_mat_input[0])
    for i_val, data in enumerate(tqdm(val_loader,ncols=50)):
    # Load data
        img_source = data["image"].to(device)
        img_source = aug_intensity_val(img_source)
        q_gt = tensor_to_mat(data["q0"][0])
        t_gt = tensor_to_mat(data["r0"][0])
        # Call the model
        out = hourglass(img_source) 
        pred_mask = out[1]["hm_c"]
        pred_mask = torch.nn.functional.interpolate(pred_mask, size=(img_source.shape[2],img_source.shape[3]), mode='bilinear', align_corners=False)
        pred_kpts  = heatmap_to_points(pred_mask) # Convert the heatmap to points
        pred_kpts  = tensor_to_mat(pred_kpts[0])
        aux, rvecs, t_est, inliers= cv2.solvePnPRansac(kpts_world, pred_kpts, k_mat_input, np.array(cam.dcoef), confidence=0.99,reprojectionError=1.0,flags=cv2.SOLVEPNP_EPNP)
        rvecs = torch.from_numpy(np.squeeze(rvecs)).unsqueeze(0).to(device)
        q_est = kornia.geometry.angle_axis_to_quaternion(rvecs, kornia.geometry.conversions.QuaternionCoeffOrder.WXYZ).unsqueeze(0)
        q_est = tensor_to_mat(q_est[0])
        speed, speed_t, speed_r = speedscore.speed_score(t_est, q_est, t_gt, q_gt, applyThresh=False)
        speed_total+=speed
        speed_t_total+=speed_t
        speed_r_total+=speed_r

    return speed_total/len(val_loader), speed_t_total/len(val_loader), speed_r_total/len(val_loader)


def generate_json_loop(test_loader ,aug_intensity_test, hourglass, kpts_world, k_mat_input, device, config, path_json):

    cam = speedplus.Camera(config["root_dir"])

    kpts_world  = tensor_to_mat(kpts_world[0])
    k_mat_input = tensor_to_mat(k_mat_input[0])
    with open(path_json, 'w') as fp:

        for i_val, data in enumerate(tqdm(test_loader,ncols=50)):
        # Load data
            img = data["image"].to(device)
            img = aug_intensity_test(img)

            pred_mask_list = hourglass(img)
            pred_mask_0 = pred_mask_list[0]["hm_c"]
            pred_mask_0  = torch.nn.functional.interpolate(pred_mask_0, size=(img.shape[2],img.shape[3]), mode='bilinear', align_corners=True)
            pred_kpts_0 = heatmap_to_points(pred_mask_0)

            pred_mask_1 = pred_mask_list[1]["hm_c"]
            pred_mask_1  = torch.nn.functional.interpolate(pred_mask_1, size=(img.shape[2],img.shape[3]), mode='bilinear', align_corners=True)
            pred_kpts_1 = heatmap_to_points(pred_mask_1)
            pred_kpts_0 = tensor_to_mat(pred_kpts_0[0])
            pred_kpts_1 = tensor_to_mat(pred_kpts_1[0])

            responses_0 = np.max(np.max(tensor_to_mat(pred_mask_list[0]["hm_c"]),axis=1),axis=1)
            responses_1 = np.max(np.max(tensor_to_mat(pred_mask_list[1]["hm_c"]),axis=1),axis=1)

            index_max_0 = np.argsort(-responses_0)
            index_max_1 = np.argsort(-responses_1)

            world_kpts2 = kpts_world.copy()
            world_kpts2_1  = world_kpts2[index_max_1,:]
            world_kpts2_0  = world_kpts2[index_max_0,:]

            pred_kpts_0 = pred_kpts_0[index_max_0,:]
            pred_kpts_1 = pred_kpts_1[index_max_1,:]

            npts = 7
            world_kpts2_0   = world_kpts2_0[:npts,:]
            world_kpts2_1   = world_kpts2_1[:npts,:]

            pred_kpts_0     = pred_kpts_0[:npts,:]        
            pred_kpts_1     = pred_kpts_1[:npts,:]

            total_rp0 = responses_0[index_max_0]
            total_rp0 = np.sum(total_rp0[:npts])

            total_rp1 = responses_1[index_max_1]
            total_rp1 = np.sum(total_rp1[:npts])

            aux_0, rvecs_0, tvecs_0, inliers_0= cv2.solvePnPRansac(world_kpts2_0, pred_kpts_0, k_mat_input, np.array(cam.dcoef),confidence=0.99,reprojectionError=2.0,flags=cv2.SOLVEPNP_EPNP)
            aux_1, rvecs_1, tvecs_1, inliers_1= cv2.solvePnPRansac(world_kpts2_1, pred_kpts_1, k_mat_input, np.array(cam.dcoef),confidence=0.99,reprojectionError=2.0,flags=cv2.SOLVEPNP_EPNP)

            
            if aux_0 and aux_1:

                if total_rp0 > total_rp1:
                    out_tvec = tvecs_0
                    out_rvec = rvecs_0
                else:
                    out_tvec = tvecs_1
                    out_rvec = rvecs_1
                q_est = torch.from_numpy(np.squeeze(out_rvec)).unsqueeze(0).to(device)
                q_est = kornia.geometry.angle_axis_to_quaternion(q_est,kornia.geometry.conversions.QuaternionCoeffOrder.WXYZ).unsqueeze(0)
                q_est = tensor_to_mat(q_est[0])
                t_est = out_tvec
                sample = {}
                sample["q_vbs2tango_true"] = q_est.tolist()
                sample["r_Vo2To_vbs_true"] = np.squeeze(out_tvec.T).tolist()
                sample["filename"] = data["y"][0]              

                json.dump(sample, fp)



import time
from scipy.io import savemat
def test_loop(val_loader ,aug_intensity_val, hourglass, kpts_world, k_mat_input, device, config):

    cam = speedplus.Camera(config["root_dir"])

    speed_total = 0
    speed_t_total = 0
    speed_r_total = 0
    kpts_world = tensor_to_mat(kpts_world[0])
    k_mat_input = tensor_to_mat(k_mat_input[0])
    speed_vals = []
    speed_t_vals = []
    speed_r_vals = []

    for i_val, data in enumerate(tqdm(val_loader,ncols=50)):
        img  = data["image"].to(device)

        q_gt = tensor_to_mat(data["q0"][0])
        t_gt = tensor_to_mat(data["r0"][0])

        img = aug_intensity_val(img)
        pred_mask_list = hourglass(img)
        pred_mask_0 = pred_mask_list[0]["hm_c"]
        pred_mask_0  = torch.nn.functional.interpolate(pred_mask_0, size=(img.shape[2],img.shape[3]), mode='bilinear', align_corners=True)
        pred_kpts_0 = heatmap_to_points(pred_mask_0)
        pred_mask_1 = pred_mask_list[1]["hm_c"]
        pred_mask_1  = torch.nn.functional.interpolate(pred_mask_1, size=(img.shape[2],img.shape[3]), mode='bilinear', align_corners=True)
        pred_kpts_1 = heatmap_to_points(pred_mask_1)


        pred_kpts_0 = tensor_to_mat(pred_kpts_0[0])
        pred_kpts_1 = tensor_to_mat(pred_kpts_1[0])

        
        #cv2.imwrite( "test_mse/ktpts" + str(i_val) + ".jpg" ,np.hstack((draw_keypoints(tensor_to_cvmat(img[0]),pred_kpts_0),draw_keypoints(tensor_to_cvmat(img[0]),pred_kpts_1))))
        responses_0 = np.max(np.max(tensor_to_mat(pred_mask_list[0]["hm_c"]),axis=1),axis=1)
        responses_1 = np.max(np.max(tensor_to_mat(pred_mask_list[1]["hm_c"]),axis=1),axis=1)
        index_max_0 = np.argsort(-responses_0)
        index_max_1 = np.argsort(-responses_1)
        world_kpts2 = kpts_world.copy()
        world_kpts2_1  = world_kpts2[index_max_1,:]
        world_kpts2_0  = world_kpts2[index_max_0,:]
        pred_kpts_0 = pred_kpts_0[index_max_0,:]
        pred_kpts_1 = pred_kpts_1[index_max_1,:]
        npts = 7
        world_kpts2_0   = world_kpts2_0[:npts,:]
        world_kpts2_1   = world_kpts2_1[:npts,:]
        pred_kpts_0     = pred_kpts_0[:npts,:]        
        pred_kpts_1     = pred_kpts_1[:npts,:]
        total_rp0 = responses_0[index_max_0]
        total_rp0 = np.sum(total_rp0[:npts])
        total_rp1 = responses_1[index_max_0]

        total_rp1 = np.sum(total_rp1[:npts])
        aux_0, rvecs_0, tvecs_0, inliers= cv2.solvePnPRansac(world_kpts2_0, pred_kpts_0, k_mat_input, np.array(cam.dcoef),confidence=0.99,reprojectionError=1.0,flags=cv2.USAC_MAGSAC)
        aux_1, rvecs_1, tvecs_1, inliers= cv2.solvePnPRansac(world_kpts2_1, pred_kpts_1, k_mat_input, np.array(cam.dcoef),confidence=0.99,reprojectionError=1.0,flags=cv2.USAC_MAGSAC)
  
        if total_rp0 > total_rp1:
            out_tvec = tvecs_0
            out_rvec = rvecs_0
        else:
            out_tvec = tvecs_1
            out_rvec = rvecs_1
        #out_tvec = tvecs_1
        #out_rvec = rvecs_1

        q_est = torch.from_numpy(np.squeeze(out_rvec)).unsqueeze(0).to(device)
        q_est = kornia.geometry.angle_axis_to_quaternion(q_est,kornia.geometry.conversions.QuaternionCoeffOrder.WXYZ).unsqueeze(0)
        q_est = tensor_to_mat(q_est[0])
        t_est = out_tvec
        #aux, rvecs, t_est, inliers= cv2.solvePnPRansac(kpts_world, pred_kpts, k_mat_input, np.array(cam.dcoef), confidence=0.99,reprojectionError=1.0,flags=cv2.SOLVEPNP_EPNP)
        #rvecs = torch.from_numpy(np.squeeze(rvecs)).unsqueeze(0).to(device)
        #q_est = kornia.geometry.angle_axis_to_quaternion(rvecs, kornia.geometry.conversions.QuaternionCoeffOrder.WXYZ).unsqueeze(0)
        #q_est = tensor_to_mat(q_est[0])



        speed, speed_t, speed_r = speedscore.speed_score(t_est, q_est, t_gt, q_gt, applyThresh=False)
        speed_total+=speed
        speed_t_total+=speed_t
        speed_r_total+=speed_r
        speed_vals.append(speed)
        speed_t_vals.append(speed_t)
        speed_r_vals.append(speed_r)        
     #   vprint(speed)
    #time.sleep(1)

    savemat('lb_all.mat', mdict={'speed_all': speed_vals,'speed_t_all': speed_t_vals,'speed_r_all': speed_r_vals})
    return speed_total/len(val_loader), speed_t_total/len(val_loader), speed_r_total/len(val_loader)    





def format_and_copy_json(previous_config):
    config = load_config(previous_config)
    print("Changing the format to the labels")

    input_file  = os.path.join(config["path_results"], previous_config,"train.json")
    output_file = os.path.join(config["path_results"], previous_config,"train_clean.json")
    print("\t --input file: ", input_file)
    print("\t --output file: ", output_file)


    # Read in the file
    with open(input_file, 'r') as file :
        filedata = file.read()

    # Replace the target string
    filedata = filedata.replace('}', '},')
    filedata = "[" + filedata[:-1] + "]"

    # Write the file out again
    with open(output_file, 'w') as file:
        file.write(filedata)

    src = os.path.join(output_file)
    
    dst = os.path.join(config["root_dir"],"sunlamp_train","train.json")
    print("coping labels to: ", dst)
    shutil.copyfile(src, dst)        