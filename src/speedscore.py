from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np



def rpy2quat(rpy=None):
    r = rpy[1] 
    p = rpy[2]
    y = rpy[0]
    cRh = np.cos(r/2)
    sRh = np.sin(r/2)
    cPh = np.cos(p/2)
    sPh = np.sin(p/2)
    cYh = np.cos(y/2)
    sYh = np.sin(y/2)
    qs_cmpl = np.array([ -(np.multiply(np.multiply(sRh,cPh),cYh) - np.multiply(np.multiply(cRh,sPh),sYh)),
                         -(np.multiply(np.multiply(cRh,sPh),cYh) + np.multiply(np.multiply(sRh,cPh),sYh)),
                         -(np.multiply(np.multiply(cRh,cPh),sYh) - np.multiply(np.multiply(sRh,sPh),cYh)),
                         np.multiply(np.multiply(cRh,cPh),cYh) + np.multiply(np.multiply(sRh,sPh),sYh)])
    qs = np.real(qs_cmpl)
    return qs


def eul2quat(ax, ay, az, atol=1e-8):
    '''
    Translate between Euler angle (ZYX) order and quaternion representation of a rotation.
    Args:
        ax: X rotation angle in radians.
        ay: Y rotation angle in radians.
        az: Z rotation angle in radians.
        atol: tolerance used for stable quaternion computation (qs==0 within this tolerance).
    Return:
        Numpy array with three entries representing the vectorial component of the quaternion.

    '''
    # Create rotation matrix using ZYX Euler angles and then compute quaternion using entries.
    cx = np.cos(ax)
    cy = np.cos(ay)
    cz = np.cos(az)
    sx = np.sin(ax)
    sy = np.sin(ay)
    sz = np.sin(az)
    r=np.zeros((3,3))
    r[0,0] = cz*cy 
    r[0,1] = cz*sy*sx - sz*cx
    r[0,2] = cz*sy*cx+sz*sx     

    r[1,0] = sz*cy 
    r[1,1] = sz*sy*sx + cz*cx 
    r[1,2] = sz*sy*cx - cz*sx

    r[2,0] = -sy   
    r[2,1] = cy*sx             
    r[2,2] = cy*cx

    # Compute quaternion: 
    qs = 0.5*np.sqrt(r[0,0] + r[1,1] + r[2,2] + 1)
    qv = np.zeros(3)
    # If the scalar component of the quaternion is close to zero, we
    # compute the vector part using a numerically stable approach
    if np.isclose(qs,0.0,atol): 
        i= np.argmax([r[0,0], r[1,1], r[2,2]])
        j = (i+1)%3
        k = (j+1)%3
        w = np.sqrt(r[i,i] - r[j,j] - r[k,k] + 1)
        qv[i] = 0.5*w
        qv[j] = (r[i,j] + r[j,i])/(2*w)
        qv[k] = (r[i,k] + r[k,i])/(2*w)
    else:
        denom = 4*qs
        qv[0] = (r[2,1] - r[1,2])/denom
        qv[1] = (r[0,2] - r[2,0])/denom
        qv[2] = (r[1,0] - r[0,1])/denom
    return qv

def error_translation(t_pr, t_gt):
    t_pr = np.reshape(t_pr, (3,))
    t_gt = np.reshape(t_gt, (3,))

    return np.sqrt(np.sum(np.square(t_gt - t_pr)))

def error_orientation(q_pr, q_gt):
    # q must be [qvec, qcos]
    q_pr = np.reshape(q_pr, (4,))
    q_gt = np.reshape(q_gt, (4,))

    qdot = np.abs(np.dot(q_pr, q_gt))
    qdot = np.minimum(qdot, 1.0)
    return np.rad2deg(2*np.arccos(qdot)) # [deg]

def speed_score(t_pr, q_pr, t_gt, q_gt, applyThresh=True, rotThresh=0.5, posThresh=0.005):
    # rotThresh: rotation threshold [deg]
    # posThresh: normalized translation threshold [m/m]
    err_t = error_translation(t_pr, t_gt)
    err_q = error_orientation(q_pr, q_gt) # [deg]

    t_gt = np.reshape(t_gt, (3,))
    speed_t = err_t / np.sqrt(np.sum(np.square(t_gt)))
    speed_r = np.deg2rad(err_q)

    # Check if within threshold
    if applyThresh and err_q < rotThresh and speed_t < posThresh:
        speed = 0.0
    else:
        speed = speed_t + speed_r

    # Accuracy of within threshold
    acc   = float(err_q < rotThresh and speed_t < posThresh)

    return speed, speed_t, speed_r