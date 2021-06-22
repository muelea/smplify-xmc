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
# Author: Vasileios Choutas, vassilis.choutas@tuebingen.mpg.de
#
# Contact: ps-license@tuebingen.mpg.de

import os.path as osp
import yaml

import torch
import torch.nn as nn
import numpy as np

class MeasurementsLoss(nn.Module):

    # The density of the human body is 985 kg / m^3
    DENSITY = 985

    def __init__(self, measurements_path, faces=None, **kwargs):
        ''' Loss that penalizes deviations in weight and height
        '''
        super(MeasurementsLoss, self).__init__()
        #  self.reduction = get_reduction_method(reduction)
        #  self.reduction_str = reduction

        assert faces is not None, 'Faces tensor is none'
        if type(faces) is not torch.Tensor:
            faces = torch.tensor(faces.astype(np.int64), dtype=torch.long)
        self.register_buffer('faces', faces)

        with open(measurements_path, 'r') as f:
            meas_data = yaml.load(f)
        head_top = meas_data['HeadTop']
        left_heel = meas_data['HeelLeft']

        self.left_heel_bc = left_heel['bc']
        self.left_heel_face_idx = left_heel['face_idx']
        self.head_top_bc = head_top['bc']
        self.head_top_face_idx = head_top['face_idx']

    def extra_repr(self):
        msg = []
        msg.append(f'Human Body Density: {self.DENSITY}')
        return '\n'.join(msg)

    def compute_height(self, shaped_triangles):
        ''' Compute the height using the heel and the top of the head
        '''
        head_top_tri = shaped_triangles[:, self.head_top_face_idx]
        head_top = (head_top_tri[:, 0, :] * self.head_top_bc[0] +
                    head_top_tri[:, 1, :] * self.head_top_bc[1] +
                    head_top_tri[:, 2, :] * self.head_top_bc[2])
        left_heel_tri = shaped_triangles[:, self.left_heel_face_idx]
        left_heel = (left_heel_tri[:, 0, :] * self.left_heel_bc[0] +
                     left_heel_tri[:, 1, :] * self.left_heel_bc[1] +
                     left_heel_tri[:, 2, :] * self.left_heel_bc[2])

        return torch.abs(head_top[:, 1] - left_heel[:, 1])

    def compute_mass(self, tris):
        ''' Computes the mass from volume and average body density
        '''
        x = tris[:, :, :, 0]
        y = tris[:, :, :, 1]
        z = tris[:, :, :, 2]
        volume = (
            -x[:, :, 2] * y[:, :, 1] * z[:, :, 0] +
            x[:, :, 1] * y[:, :, 2] * z[:, :, 0] +
            x[:, :, 2] * y[:, :, 0] * z[:, :, 1] -
            x[:, :, 0] * y[:, :, 2] * z[:, :, 1] -
            x[:, :, 1] * y[:, :, 0] * z[:, :, 2] +
            x[:, :, 0] * y[:, :, 1] * z[:, :, 2]
        ).sum(dim=1).abs() / 6.0
        return volume * self.DENSITY

    def forward(self, v_shaped, **kwargs):
        batch_size = v_shaped.shape[0]

        v_shaped_triangles = torch.index_select(
            v_shaped, 1, self.faces.view(-1)).reshape(batch_size, -1, 3, 3)

        mesh_height = self.compute_height(v_shaped_triangles)
        mesh_mass = self.compute_mass(v_shaped_triangles)

        measurements = {'mass': mesh_mass, 'height': mesh_height}
        return measurements
