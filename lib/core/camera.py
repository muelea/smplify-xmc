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

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from collections import namedtuple

import torch
import torch.nn as nn

from smplx.lbs import transform_mat


PerspParams = namedtuple('ModelOutput',
                         ['rotation', 'translation', 'center',
                          'focal_length'])


def create_camera(camera_type='persp', **kwargs):
    if camera_type.lower() == 'persp':
        return PerspectiveCamera(**kwargs)
    elif camera_type.lower() == 'persp_rot_ea':
        return PerspectiveCameraRotEa(**kwargs)
    else:
        raise ValueError('Uknown camera type: {}'.format(camera_type))


class PerspectiveCamera(nn.Module):

    def __init__(self, rotation=None, translation=None,
                 focal_length_x=5000, focal_length_y=5000,
                 batch_size=1,
                 center=None, dtype=torch.float32, **kwargs):
        super(PerspectiveCamera, self).__init__()
        self.batch_size = batch_size
        self.dtype = dtype
        # Make a buffer so that PyTorch does not complain when creating
        # the camera matrix
        self.register_buffer('zero',
                             torch.zeros([batch_size], dtype=dtype))

        # create focal length parameter
        focal_length_x = focal_length_x * torch.ones([batch_size, 1], dtype=dtype)
        focal_length_x = nn.Parameter(focal_length_x, requires_grad=True)
        self.register_parameter('focal_length_x', focal_length_x)
        focal_length_y = focal_length_y * torch.ones([batch_size, 1], dtype=dtype)
        focal_length_y = nn.Parameter(focal_length_y, requires_grad=True)
        self.register_parameter('focal_length_y', focal_length_y)

        if center is None:
            center = torch.zeros([batch_size, 2], dtype=dtype)
        self.register_buffer('center', center)

        if rotation is None:
            rotation = torch.eye(3, dtype=dtype).unsqueeze(dim=0).repeat(batch_size, 1, 1)

        rotation = nn.Parameter(rotation, requires_grad=False)
        self.register_parameter('rotation', rotation)

        pitch = nn.Parameter(torch.zeros([batch_size,1], dtype=dtype),
            requires_grad=True)
        self.register_parameter('pitch', pitch)
        roll = nn.Parameter(torch.zeros([batch_size,1], dtype=dtype),
            requires_grad=True)
        self.register_parameter('roll', roll)
        yaw = nn.Parameter(torch.zeros([batch_size,1], dtype=dtype),
            requires_grad=True)
        self.register_parameter('yaw', yaw)

        if translation is None:
            translation = torch.zeros([batch_size, 3], dtype=dtype)
        translation = nn.Parameter(translation,
            requires_grad=True)
        self.register_parameter('translation', translation)

    @torch.no_grad()
    def reset_params(self, **params_dict) -> None:
        for param_name, param in self.named_parameters():
            if param_name in params_dict:
                param[:] = torch.tensor(params_dict[param_name])
            else:
                param.fill_(0)

    def ea2rm(self):
        x = self.pitch
        y = self.yaw
        z = self.roll
        cos_x, sin_x = torch.cos(x), torch.sin(x)
        cos_y, sin_y = torch.cos(y), torch.sin(y)
        cos_z, sin_z = torch.cos(z), torch.sin(z)

        R = torch.stack(
              [torch.cat([cos_y*cos_z, sin_x*sin_y*cos_z - cos_x*sin_z, cos_x*sin_y*cos_z + sin_x*sin_z], dim=1),
               torch.cat([cos_y*sin_z, sin_x*sin_y*sin_z + cos_x*cos_z, cos_x*sin_y*sin_z - sin_x*cos_z], dim=1),
               torch.cat([-sin_y, sin_x*cos_y, cos_x*cos_y], dim=1)], dim=1)

        return R

    def forward(self, points):
        device = points.device

        with torch.no_grad():
            camera_mat = torch.zeros([self.batch_size, 2, 2],
                                     dtype=self.dtype, device=points.device)
        camera_mat[:, 0, 0] = self.focal_length_x.flatten()
        camera_mat[:, 1, 1] = self.focal_length_y.flatten()

        rotation = self.ea2rm()
        self.rotation[:] = rotation.detach()
        camera_transform = transform_mat(rotation,
                                         self.translation.unsqueeze(dim=-1))
        homog_coord = torch.ones(list(points.shape)[:-1] + [1],
                                 dtype=points.dtype,
                                 device=device)
        # Convert the points to homogeneous coordinates
        points_h = torch.cat([points, homog_coord], dim=-1)

        projected_points = torch.einsum('bki,bji->bjk',
                                        [camera_transform, points_h])

        img_points = torch.div(projected_points[:, :, :2],
                               projected_points[:, :, 2].unsqueeze(dim=-1))
        img_points = torch.einsum('bki,bji->bjk', [camera_mat, img_points]) \
            + self.center.unsqueeze(dim=1)

        return img_points