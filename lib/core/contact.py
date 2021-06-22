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
import torch.nn as nn

class MimickedSelfContactLoss(nn.Module):
    def __init__(self,
                 geodesics_mask,
    ):
        super().__init__()
        """
        Loss that lets vertices in contact on presented mesh attract vertices that are close.
        """
        # geodesic distance mask
        self.register_buffer('geomask', geodesics_mask)

    def forward(self, presented_contact, v2v=None, vertices=None,
                contact_mode="dist_tanh", contact_thresh=1):

        contactloss = 0.0

        if v2v is None:
            #compute pairwise distances
            verts = vertices.contiguous()
            nv = verts.shape[1]
            v2v = verts.squeeze().unsqueeze(1).expand(nv, nv, 3) - \
                  verts.squeeze().unsqueeze(0).expand(nv, nv, 3)
            v2v = torch.norm(v2v, 2, 2)

        # loss for self-contact from mimic'ed pose
        if len(presented_contact) > 0:
            # without geodesic distance mask, compute distances
            # between each pair of verts in contact
            with torch.no_grad():
                cvertstobody = v2v[presented_contact, :]
                maskgeo = self.geomask[presented_contact, :]
                weights = torch.ones_like(cvertstobody).to(verts.device)
                weights[~maskgeo] = float('inf')
                min_idx = torch.min((cvertstobody+1) * weights, 1)[1]
            v2v_min = v2v[presented_contact, min_idx]

            # tanh will not pull vertices that are ~more than contact_thres far apart
            if contact_mode == "dist_tanh":
                contactloss = contact_thresh * torch.tanh(v2v_min / contact_thresh)
                contactloss = contactloss.mean()
            else:
                contactloss = v2v_min.mean()

        return contactloss