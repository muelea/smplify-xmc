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
import numpy as np
import torch.nn as nn
import torchgeometry as tgm
import math

class GlobalOrientLoss(nn.Module):

    def euler_angles_from_rotmat(self, R):
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


    def forward(self, global_orient, init_global_orient):
        rotmat_bm = tgm.angle_axis_to_rotation_matrix(global_orient)
        rotmat_init = tgm.angle_axis_to_rotation_matrix(init_global_orient)
        euler_angle_bm = self.euler_angles_from_rotmat(rotmat_bm)[0]
        euler_angle_init = self.euler_angles_from_rotmat(rotmat_init)[0]
        if euler_angle_bm[0]*euler_angle_init[0] > 0:
            global_orient_x_loss1 = (euler_angle_bm[0]* 180 / math.pi - euler_angle_init[0]* 180 / math.pi)
            global_orient_x_loss2 = (-euler_angle_bm[0]* 180 / math.pi + euler_angle_init[0]* 180 / math.pi)
            global_orient_x_loss = torch.min(global_orient_x_loss1, global_orient_x_loss2)
        else:
            if euler_angle_bm[0] < 0:
                global_orient_x_loss1 = (360 + euler_angle_bm[0]* 180 / math.pi - euler_angle_init[0]* 180 / math.pi)
                global_orient_x_loss2 = (-1 * euler_angle_bm[0]* 180 / math.pi + euler_angle_init[0]* 180 / math.pi)
                global_orient_x_loss = torch.min(global_orient_x_loss1, global_orient_x_loss2)
            else:
                global_orient_x_loss1 = (-1 * euler_angle_bm[0]* 180 / math.pi + 360 + euler_angle_init[0]* 180 / math.pi)
                global_orient_x_loss2 = (euler_angle_bm[0]* 180 / math.pi - euler_angle_init[0]* 180 / math.pi)
                global_orient_x_loss = torch.min(global_orient_x_loss1, global_orient_x_loss2)
        if euler_angle_bm[2]*euler_angle_init[2] > 0:
            global_orient_z_loss1 = (euler_angle_bm[2]* 180 / math.pi - euler_angle_init[2]* 180 / math.pi)
            global_orient_z_loss2 = (-euler_angle_bm[2]* 180 / math.pi + euler_angle_init[2]* 180 / math.pi)
            global_orient_z_loss = torch.min(global_orient_z_loss1, global_orient_z_loss2)
        else:
            if euler_angle_bm[2] < 0:
                global_orient_z_loss1 = (360 + euler_angle_bm[2]* 180 / math.pi - euler_angle_init[2]* 180 / math.pi)
                global_orient_z_loss2 = (-1*euler_angle_bm[2]* 180 / math.pi + euler_angle_init[2]* 180 / math.pi)
                global_orient_z_loss = torch.min(global_orient_z_loss1, global_orient_z_loss2)
            else:
                global_orient_z_loss1 = (-1*euler_angle_bm[2]* 180 / math.pi + 360 + euler_angle_init[2]* 180 / math.pi)
                global_orient_z_loss2 = (euler_angle_bm[2]* 180 / math.pi - euler_angle_init[2]* 180 / math.pi)
                global_orient_z_loss = torch.min(global_orient_z_loss1, global_orient_z_loss2)

        global_orient_xz_loss = 0.0001 * (global_orient_x_loss**4 + global_orient_z_loss**4)

        return global_orient_xz_loss


# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
def rotationMatrixToEulerAngles(R) :

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])
