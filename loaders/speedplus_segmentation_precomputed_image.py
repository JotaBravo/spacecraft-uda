import numpy as np
import json
import os
from PIL import Image
from matplotlib import pyplot as plt

from torch.utils.data import Dataset
import scipy.io
import cv2

import time
def process_json_dataset(root_dir):
    with open(os.path.join(root_dir, 'synthetic', 'train.json'), 'r') as f:
        train_images_labels = json.load(f)

    with open(os.path.join(root_dir, 'synthetic', 'validation.json'), 'r') as f:
        test_image_list = json.load(f)

    with open(os.path.join(root_dir, 'sunlamp', 'test.json'), 'r') as f:
        sunlamp_image_list = json.load(f)

    with open(os.path.join(root_dir, 'lightbox', 'test.json'), 'r') as f:
        lightbox_image_list = json.load(f)

    partitions = {'validation': [], 'train': [], 'sunlamp': [], 'lightbox': []}
    labels = {}

    for image_ann in train_images_labels:
        partitions['train'].append(image_ann['filename'])
        labels[image_ann['filename']] = {'q': image_ann['q_vbs2tango_true'], 'r': image_ann['r_Vo2To_vbs_true']}

    for image in test_image_list:
        partitions['validation'].append(image['filename'])

    for image in sunlamp_image_list:
        partitions['sunlamp'].append(image['filename'])

    for image in lightbox_image_list:
        partitions['lightbox'].append(image['filename'])

    return partitions, labels

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

def project(q, r):

        """ Projecting points to image frame to draw axes """

        # reference points in satellite frame for drawing axes
        p_axes = np.array([[0, 0, 0, 1],
                           [1, 0, 0, 1],
                           [0, 1, 0, 1],
                           [0, 0, 1, 1]])
        points_body = np.transpose(p_axes)

        # transformation to camera frame
        pose_mat = np.hstack((np.transpose(quat2dcm(q)), np.expand_dims(r, 1)))
        p_cam = np.dot(pose_mat, points_body)

        # getting homogeneous coordinates
        points_camera_frame = p_cam / p_cam[2]

        # projection to image plane
        points_image_plane = Camera.K.dot(points_camera_frame)
        x0, y0 = (points_image_plane[0], points_image_plane[1])
        
        # apply distortion
        dist = Camera.dcoef
        
        r2 = x0*x0 + y0*y0
        cdist = 1 + dist[0]*r2 + dist[1]*r2*r2 + dist[4]*r2*r2*r2
        x1  = x0*cdist + dist[2]*2*x0*y0 + dist[3]*(r2 + 2*x0*x0)
        y1  = y0*cdist + dist[2]*(r2 + 2*y0*y0) + dist[3]*2*x0*y0
        
        x = Camera.K[0,0]*x1 + Camera.K[0,2]
        y = Camera.K[1,1]*y1 + Camera.K[1,2]
        
        return x, y

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

class SatellitePoseEstimationDataset:

    """ Class for dataset inspection: easily accessing single images, and corresponding ground truth pose data. """

    def __init__(self, root_dir='datasets/'):
        self.partitions, self.labels = process_json_dataset(root_dir)
        self.root_dir = root_dir

    def get_image(self, i=0, split='train'):

        """ Loading image as PIL image. """

        img_name = self.partitions[split][i]
        if split=='train':
            img_name = os.path.join(self.root_dir, 'synthetic','images', img_name)
        elif split=='validation':
            img_name = os.path.join(self.root_dir, 'synthetic','images', img_name)
        elif split=='sunlamp_train':
            img_name = os.path.join(self.root_dir, 'sunlamp_train','images', img_name)
        elif split=='lightbox_train':
            img_name = os.path.join(self.root_dir, 'lightbox_train','images', img_name)
        else:
            print()
            # raise error?
        
        image = Image.open(img_name).convert('RGB')
        return image

    def get_pose(self, i=0):

        """ Getting pose label for image. """

        img_id = self.partitions['train'][i]
        q, r = self.labels[img_id]['q'], self.labels[img_id]['r']
        return q, r

    def visualize(self, i, partition='train', ax=None):

        """ Visualizing image, with ground truth pose with axes projected to training image. """

        if ax is None:
            ax = plt.gca()
        img = self.get_image(i)
        ax.imshow(img)

        # no pose label for test
        if partition == 'train':
            q, r = self.get_pose(i)
            xa, ya = project(q, r)
            ax.arrow(xa[0], ya[0], xa[1] - xa[0], ya[1] - ya[0], head_width=30, color='r')
            ax.arrow(xa[0], ya[0], xa[2] - xa[0], ya[2] - ya[0], head_width=30, color='g')
            ax.arrow(xa[0], ya[0], xa[3] - xa[0], ya[3] - ya[0], head_width=30, color='b')

        return

class PyTorchSatellitePoseEstimationDataset(Dataset):

        """ SPEED dataset that can be used with DataLoader for PyTorch training. """

        def __init__(self, split='train', speed_root='datasets/', transform_input=None,  transform_gt=None, config=None):

            if split not in {'train', 'validation', 'sunlamp', 'lightbox', 'sunlamp_train', 'lightbox_train'}:
                raise ValueError('Invalid split, has to be either \'train\', \'validation\', \'sunlamp\' or \'lightbox\'')

            if split in {'train', 'validation'}:
                self.image_root = os.path.join(speed_root, 'synthetic', 'images')
                self.mask_root  = os.path.join(speed_root, 'synthetic', 'kptsmap')

                # We separate the if statement for train and val as we may need different training splits
                if split in {'train'}: 
                    with open(os.path.join(speed_root, "synthetic", split + '.json'), 'r') as f:
                        label_list = json.load(f)

                if split in {'validation'}:
                    with open(os.path.join(speed_root, "synthetic", split + '.json'), 'r') as f:
                        label_list = json.load(f)  

            elif split in {'sunlamp_train', 'lightbox_train'}:
                self.image_root = os.path.join(speed_root, split, 'images')
                self.mask_root  = os.path.join(speed_root, split, 'kptsmap')

                with open(os.path.join(speed_root, split, 'train.json'), 'r') as f:
                    label_list = json.load(f)

            else:
                self.image_root = os.path.join(speed_root, split, 'images')

                with open(os.path.join(speed_root, split, 'test.json'), 'r') as f:
                    label_list = json.load(f)

            # Parse inputs
            self.sample_ids = [label['filename'] for label in label_list]
            self.train = (split == 'train') or (split == 'sunlamp_train') or (split == 'lightbox_train')
            self.validation = (split == 'validation') #or (split == 'sunlamp_train') or (split == 'lightbox_train')

            if self.train or self.validation:
                self.labels = {label['filename']: {'q': label['q_vbs2tango_true'], 'r': label['r_Vo2To_vbs_true']} for label in label_list}

            # Load assets
            kpts_mat = scipy.io.loadmat(speed_root + "kpts.mat") 
            self.kpts = np.array(kpts_mat["corners"])   # Spacecraft key-points
            self.cam  = Camera(speed_root)              # Camera parameters
            self.K = self.cam.K

            self.K[0, :] *= ((config["cols"])/1920)
            self.K[1, :] *= ((config["rows"])/1200)    

            # Transforms for the tensors inputed to the network
            self.transform_input  = transform_input
            self.col_factor_input = ((config["cols"])/1920)
            self.row_factor_input = ((config["rows"])/1200)   
            self.config = config

            

        def __len__(self):
            return len(self.sample_ids)

        def __getitem__(self, idx):
            
            # Load image
            sample_id = self.sample_ids[idx]
            #if sample_id == 'img019693.jpg':
            #    sample_id = 'img019694.jpg'
            img_name  = os.path.join(self.image_root, sample_id)
            pil_image = cv2.imread(img_name)
            torch_image  = self.transform_input(pil_image)

            y = sample_id

            # For validation, just load the gt pose
            if self.validation:
                q0, r0 = self.labels[sample_id]['q'], self.labels[sample_id]['r']

            # For training, we need more stuff
            if self.train:
                q0, r0 = self.labels[sample_id]['q'], self.labels[sample_id]['r']

                ## Before computing the true key-point positions we need some transformations
                q  = quat2dcm(q0)
                r = np.expand_dims(np.array(r0),axis=1)
                r = q@(r)

                ## Spacecraft kpts placed in front of the camera
                kpts_cam = q.T@(self.kpts+r)

                ## Project key-points to the camera
                kpts_im = self.K@(kpts_cam/kpts_cam[2,:])
                kpts_im = np.transpose(kpts_im[:2,:])
                

    
                # Load the ground-truth heatmaps
                heatmap = np.zeros((self.config["rows"], self.config["cols"],11),dtype='uint8')
                target_name_02   = os.path.join(self.mask_root, sample_id.replace(".jpg","_02.png"))
                target_name_35   = os.path.join(self.mask_root, sample_id.replace(".jpg","_35.png"))
                target_name_69   = os.path.join(self.mask_root, sample_id.replace(".jpg","_69.png"))
                target_name_1011 = os.path.join(self.mask_root, sample_id.replace(".jpg","_1011.png"))

                img_02   = cv2.imread(target_name_02)
                img_35   = cv2.imread(target_name_35)
                img_69   = cv2.imread(target_name_69)
                img_1011 = cv2.imread(target_name_1011)

                heatmap[:,:,0:3]  = img_02
                heatmap[:,:,3:6]  = img_35
                heatmap[:,:,6:9]  = img_69
                heatmap[:,:,9:11] = img_1011[:,:,1:]

                visible_kpts = np.max(np.max(heatmap,axis=0),axis=0)
                torch_heatmap = self.transform_input(heatmap)
            

        
            sample = dict()
            if self.train:
                sample["image"]      = torch_image
                sample["heatmap"]    = torch_heatmap
                sample["kpts_3Dcam"] = kpts_cam.astype(np.float32)
                sample["kpts_2Dim"]  = kpts_im.astype(np.float32)
                sample["visible_kpts"] = visible_kpts.astype(np.bool)

                sample["q0"]         = np.array(q0).astype(np.float32)  
                sample["r0"]         = np.array(r0).astype(np.float32)   
                sample["y"]          = y

            if self.validation:
                sample["image"]      = torch_image
                sample["q0"]         = np.array(q0).astype(np.float32)  
                sample["r0"]         = np.array(r0).astype(np.float32)   
                sample["y"]          = y                

            else:
                sample["image"]    = torch_image
                sample["y"] = y


            return sample

    
