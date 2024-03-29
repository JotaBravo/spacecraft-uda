import numpy as np
import json
import cv2
import scipy.io
import os
from tqdm import tqdm
from src import utils
import argparse
from multiprocessing import Pool
from multiprocessing import freeze_support
import time

# Old legacy method with npz files

'''Define function to run mutiple processors and pool the results together'''
def run_multiprocessing(func, i, n_processors):
    with Pool(processes=n_processors) as pool:
        return pool.map(func, i)

class Camera:
    def __init__(self,speed_root):

        """" Utility class for accessing camera parameters. """
        with open(os.path.join(speed_root, 'camera.json'), 'r') as f:
            camera_params = json.load(f)
        self.fx = camera_params['fx'] # focal length[m]
        self.fy = camera_params['fy'] # focal length[m]
        self.nu = camera_params['Nu'] # number of horizontal[pixels]
        self.nv = camera_params['Nv'] # number of vertical[pixels]
        self.ppx = camera_params['ppx'] # horizontal pixel pitch[m / pixel]
        self.ppy = camera_params['ppy'] # vertical pixel pitch[m / pixel]
        self.fpx = self.fx / self.ppx  # horizontal focal length[pixels]
        self.fpy = self.fy / self.ppy  # vertical focal length[pixels]
        self.k = camera_params['cameraMatrix']
        self.K = np.array(self.k) # cameraMatrix
        self.dcoef = camera_params['distCoeffs']

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

def draw_labelmap(img, pt, sigma, type='Gaussian'):
    # Draw a 2D gaussian
    # Adopted from https://github.com/anewell/pose-hg-train/blob/master/src/pypose/draw.py
    
    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if (ul[0] >= 0.95*img.shape[1] or ul[1] >= 0.95*img.shape[0] or br[0] < 10 or br[1] < 10):
        # If not, just return the image as is
        return img, False
    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if type == 'Gaussian':
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    elif type == 'Cauchy':
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)
    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])
    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img, True


parser = argparse.ArgumentParser(description='Pose Estimation')
parser.add_argument('--cfg', '--config', metavar='DIR', help='Path to the configuration', required=True)


args = parser.parse_args()
config = utils.load_config(args.cfg)


speed_root = config["root_dir"]
if config["split_submission"] == "sunlamp":
    dataset_id = "sunlamp_train"
if config["split_submission"] == "lightbox":
    dataset_id = "lightbox_train"
if config["split_submission"] == "synthetic":
    dataset_id = "synthetic"

image_root = os.path.join(speed_root, dataset_id, 'images')

if dataset_id == "sunlamp_train" or dataset_id == "lightbox_train": # reset all maps
    kptsmap_root = os.path.join(speed_root, dataset_id, 'kptsmap')
    os.system("rm " + kptsmap_root + "/*.*")

with open(os.path.join(speed_root, dataset_id, "train" + '.json'), 'r') as f:
        label_list = json.load(f)
if config["split_submission"] == "synthetic":
    with open(os.path.join(speed_root, dataset_id, "validation" + '.json'), 'r') as f:
        label_list += json.load(f)      
          
sample_ids = [label['filename'] for label in label_list]
labels = {label['filename']: {'q': label['q_vbs2tango_true'], 'r': label['r_Vo2To_vbs_true']} for label in label_list}
tmp = scipy.io.loadmat(speed_root + "kpts.mat")
kpts = np.array(tmp["corners"])
cam = Camera(speed_root)


'''Define task function'''
def save_map(sample_id):
    
    sample_number = float(sample_id.split("img")[1].split(".jpg")[0])

    img_name  = os.path.join(image_root, sample_id)
    pil_image = cv2.imread(img_name)

    q_return, r = labels[sample_id]['q'], labels[sample_id]['r']


    q = quat2dcm(q_return)

    r2 = np.expand_dims(np.array(r),axis=1)
    r2 = q@(r2)



    tmp     = q.T@(kpts+r2)
    kpts_im = cam.K@(tmp/tmp[2,:])
    kpts_im = np.transpose(kpts_im[:2,:])
    #kpts_im[kpts_im[:,0] >= 1920,0] = 1920-1
    #kpts_im[kpts_im[:,0] <= 0   ,0] = 0
    #kpts_im[kpts_im[:,1] >= 1200,1] = 1200-1
    #kpts_im[kpts_im[:,1] <= 0   ,1] = 0
    pil_drawkpts = np.zeros((1200, 1920, 11))
    vis = np.zeros((11),dtype=bool)
    for i, pt in enumerate(kpts_im):
        aux, vis[i] =  draw_labelmap(pil_drawkpts[:,:,i], pt, 15.0, type='Gaussian')
        if vis[i]:
            aux = ((aux-np.min(aux))/(np.max(aux)-np.min(aux)))
            pil_drawkpts[:,:,i]  = aux


    filt_kpts_im = kpts_im[vis,:]
    filt_kpts    = kpts[:,vis]


    savename = os.path.join(kptsmap_root,sample_id.split(".jpg")[0])

    np.savez_compressed(savename, pil_drawkpts, filt_kpts_im, filt_kpts, vis, allow_pickle=True)




def main():
    #start = time.perf_counter()
#
    '''
    set up parameters required by the task
    '''
    num_max = 1000000
    n_processors =8
    x_ls = sample_ids

    '''
    pass the task function, followed by the parameters to processors
    '''
    out = run_multiprocessing(save_map, x_ls, n_processors)
#
#
    #print("Mutiprocessing time: {}mins\n".format((time.clock()-start)/60))
    #print("Mutiprocessing time: {}secs\n".format((time.clock()-start)))
if __name__ == "__main__":
    #freeze_support()   # required to use multiprocessinsg
    main()   