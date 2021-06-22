# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2021 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import torch
import cv2
import numpy as np
import math

def euler_angles_from_rotmat(R):
    """
    computer euler angles for rotation around x, y, z axis
    from rotation amtrix
    R: 4x4 rotation matrix
    https://www.gregslabaugh.net/publications/euler.pdf
    """
    r21 = np.round(R[:,2,0].item(), 4)
    if abs(r21) != 1:
        y_angle1 = -1 * torch.asin(R[:,2,0])
        y_angle2 = math.pi + torch.asin(R[:,2,0])
        cy1, cy2 = torch.cos(y_angle1), torch.cos(y_angle2)

        x_angle1 = torch.atan2(R[:,2,1] / cy1, R[:,2,2] / cy1)
        x_angle2 = torch.atan2(R[:,2,1] / cy2, R[:,2,2] / cy2)
        z_angle1 = torch.atan2(R[:,1,0] / cy1, R[:,0,0] / cy1)
        z_angle2 = torch.atan2(R[:,1,0] / cy2, R[:,0,0] / cy2)

        s1 = (x_angle1, y_angle1, z_angle1)
        s2 = (x_angle2, y_angle2, z_angle2)
        s = (s1, s2)

    else:
        z_angle = torch.tensor([0], device=R.device).float()
        if r21 == -1:
            y_angle = torch.tensor([math.pi / 2], device=R.device).float()
            x_angle = z_angle + torch.atan2(R[:,0,1], R[:,0,2])
        else:
            y_angle = -torch.tensor([math.pi / 2], device=R.device).float()
            x_angle = -z_angle + torch.atan2(-R[:,0,1], R[:,0,2])
        s = ((x_angle, y_angle, z_angle), )
    return s



def estimate_translation_np(S, joints_2d, joints_conf, focal_length_x=5000, focal_length_y=5000, W=224, H=224):
    """Find camera translation that brings 3D joints S closest to 2D the corresponding joints_2d.
    Taken from: https://github.com/nkolot/SPIN/blob/master/utils/geometry.py
    Input:
        S: (25, 3) 3D joint locations
        joints: (25, 3) 2D joint locations and confidence
    Returns:
        (3,) camera translation vector
    """
    num_joints = S.shape[0]
    # focal length
    f = np.array([focal_length_x, focal_length_y])
    # optical center
    center = np.array([W/2., H/2.])
    # 2d joints mean
    p2d = joints_2d.mean(0)
    # 3d joints mean
    P3d = S.mean(0)[:2]

    trans_xy = (p2d-center) / f - P3d

    # transformations
    Z = np.reshape(np.tile(S[:,2],(2,1)).T,-1)
    XY = np.reshape(S[:,0:2],-1)
    O = np.tile(center,num_joints)
    F = np.tile(f,num_joints)
    weight2 = np.reshape(np.tile(np.sqrt(joints_conf),(2,1)).T,-1)

    # least squares
    Q = np.array([F*np.tile(np.array([1,0]),num_joints),
        F*np.tile(np.array([0,1]),num_joints), O-np.reshape(joints_2d,-1)]).T
    c = (np.reshape(joints_2d,-1)-O)*Z - F*XY

    # weighted least squares
    W = np.diagflat(weight2)
    Q = np.dot(W,Q)
    c = np.dot(W,c)

    # square matrix
    A = np.dot(Q.T,Q)
    b = np.dot(Q.T,c)

    # solution
    trans = np.linalg.solve(A, b)
    #trans[:2] = trans_xy

    return trans
