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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.utils import utils
from lib.core.measurements import MeasurementsLoss
from lib.core.globalorient import GlobalOrientLoss
from lib.core.contact import MimickedSelfContactLoss
from selfcontact.losses import SelfContactLoss as GenSelfContactLoss
from smplx import lbs
import torchgeometry as tgm
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

class SMPLifyXMCLoss(nn.Module):
    def __init__(
            self,
            body_model=None,
            presented_global_orient=None,
            presented_body_pose=None,
            presented_left_hand_pose=None,
            presented_right_hand_pose=None,
            presented_contact=None,
            shape_prior=None,
            expr_prior=None,
            angle_prior=None,
            jaw_prior=None,
            use_joints_conf=True,
            use_face=True,
            use_hands=True,
            left_hand_prior=None,
            right_hand_prior=None,
            measurements_path='',
            dtype=torch.float32,
            data_weight=1.0,
            body_pose_weight=0.0,
            shape_weight=0.0,
            bending_prior_weight=0.0,
            hand_prior_weight=0.0,
            expr_prior_weight=0.0,
            jaw_prior_weight=0.0,
            scopti_weight=0.0,
            use_contact=True,
            mimicked_contact_weight=0.0,
            camera_rot_weight=0.0,
            body_global_orient_weight=0.0,
            focal_length_weight=0.0,
            sc_module=None,
            rho=100,
            gen_sc_inside_weight=0.5,
            gen_sc_outside_weight=0.0,
            gen_sc_contact_weight=0.5,
            **kwargs
    ):
        super(SMPLifyXMCLoss, self).__init__()

        self.use_joints_conf = use_joints_conf
        self.angle_prior = angle_prior

        self.robustifier = utils.GMoF(rho=rho)
        self.rho = rho
        self.bs = presented_body_pose.shape[0]

        self.J_regressor = body_model.J_regressor

        self.shape_prior = shape_prior

        self.use_hands = use_hands
        if self.use_hands:
            self.left_hand_prior = left_hand_prior
            self.right_hand_prior = right_hand_prior

        self.use_face = use_face
        if self.use_face:
            self.expr_prior = expr_prior
            self.jaw_prior = jaw_prior

        self.use_contact = use_contact

        self.measurements_crit = MeasurementsLoss(measurements_path, faces=body_model.faces)

        self.globalorient_crit = GlobalOrientLoss()

        self.msc_crit = MimickedSelfContactLoss(geodesics_mask=sc_module.geomask)

        self.sc_crit = GenSelfContactLoss(
            contact_module=sc_module,
            inside_loss_weight=gen_sc_inside_weight, 
            outside_loss_weight=gen_sc_outside_weight, 
            contact_loss_weight=gen_sc_contact_weight, 
            align_faces=True, 
            use_hd=True, 
            test_segments=True, 
            device='cuda',
            model_type='smplx'
        )

        # register body model in mean pose for measurement loss
        nbp = body_model.body_pose.shape[1]
        self.register_buffer('body_mean_pose',
            torch.tensor(torch.zeros(1,nbp), dtype=dtype))
        self.body_model = body_model
        self.body_model.reset_params(body_pose=self.body_mean_pose)

        # loss weights
        self.register_buffer('data_weight',
                             torch.tensor(data_weight, dtype=dtype))
        self.register_buffer('body_pose_weight',
                             torch.tensor(body_pose_weight, dtype=dtype))
        self.register_buffer('shape_weight',
                             torch.tensor(shape_weight, dtype=dtype))
        self.register_buffer('bending_prior_weight',
                             torch.tensor(bending_prior_weight, dtype=dtype))
        self.register_buffer('camera_rot_weight',
                             torch.tensor(camera_rot_weight, dtype=dtype))
        self.register_buffer('body_global_orient_weight',
                             torch.tensor(body_global_orient_weight, dtype=dtype))
        self.register_buffer('focal_length_weight',
                             torch.tensor(focal_length_weight, dtype=dtype))
        if self.use_hands:
            self.register_buffer('hand_prior_weight',
                            torch.tensor(hand_prior_weight, dtype=dtype))
        if self.use_face:
            self.register_buffer('expr_prior_weight',
                                 torch.tensor(expr_prior_weight, dtype=dtype))
            self.register_buffer('jaw_prior_weight',
                                 torch.tensor(jaw_prior_weight, dtype=dtype))
        if self.use_contact:
            self.register_buffer('mimicked_contact_weight',
                            torch.tensor(mimicked_contact_weight, dtype=dtype))
            self.register_buffer('scopti_weight',
                            torch.tensor(scopti_weight, dtype=dtype))

        self.register_buffer('init_global_orient', presented_global_orient)
        self.register_buffer('presented_left_hand_pose', presented_left_hand_pose)
        self.register_buffer('presented_right_hand_pose', presented_right_hand_pose)
        self.register_buffer('presented_body_pose', presented_body_pose)
        self.register_buffer('presented_contact', presented_contact)

    def reset_loss_weights(self, loss_weight_dict):
        for key in loss_weight_dict:
            if hasattr(self, key):
                weight_tensor = getattr(self, key)
                if 'torch.Tensor' in str(type(loss_weight_dict[key])):
                    weight_tensor = loss_weight_dict[key].clone().detach()
                else:
                    weight_tensor = torch.tensor(loss_weight_dict[key],
                                                 dtype=weight_tensor.dtype,
                                                 device=weight_tensor.device)
                setattr(self, key, weight_tensor)

    def forward(self, body_model_output, camera, gt_joints, joints_conf,
                joint_weights, gt_height=None, gt_weight=None):

        # ==== joint loss / 2d keypoint loss ====
        # get 2D joints
        joints = body_model_output.joints
        projected_joints = camera(joints)

        # Calculate the weights for each joints
        weights = (joint_weights * joints_conf
                   if self.use_joints_conf else
                   joint_weights).unsqueeze(dim=-1)

        # Calculate the distance of the projected joints from
        # the ground truth 2D detections
        joint_diff = self.robustifier(gt_joints - projected_joints)
        joint_loss = (torch.sum(weights ** 2 * joint_diff) *
                      self.data_weight ** 2)

        # ==== loss that prevents rotation around x and z axis, but allows y axis rotation ====
        global_orient_xz_loss = 0.0
        if self.presented_body_pose is not None and self.body_pose_weight == 0:
            global_orient_xz_loss = self.globalorient_crit(body_model_output.global_orient,
                                                           self.init_global_orient)
            global_orient_xz_loss = self.body_global_orient_weight * global_orient_xz_loss

        # ==== mimic'ed pose prior loss ====
        body_pose_prior_loss = 0.0
        if self.presented_body_pose is not None:
            body_pose_prior_loss = F.mse_loss(body_model_output.body_pose,
                            self.presented_body_pose, reduction='sum') * \
                            self.body_pose_weight

        # ==== more prios for hands and face  ====
        # Apply the prior on the pose space of the hand
        left_hand_prior_loss, right_hand_prior_loss = 0.0, 0.0
        if self.use_hands and self.left_hand_prior is not None:
            left_hand_prior_loss = torch.sum(
                self.left_hand_prior(
                    body_model_output.left_hand_pose)) * \
                self.hand_prior_weight ** 2

        if self.use_hands and self.right_hand_prior is not None:
            right_hand_prior_loss = torch.sum(
                self.right_hand_prior(
                    body_model_output.right_hand_pose)) * \
                self.hand_prior_weight ** 2

        expression_loss, jaw_prior_loss = 0.0, 0.0
        if self.use_face and self.expr_prior_weight > 0:
            expression_loss = torch.sum(self.expr_prior(
                body_model_output.expression)) * \
                self.expr_prior_weight ** 2

        if self.use_face:
            if hasattr(self, 'jaw_prior'):
                jaw_prior_loss = torch.sum(
                    self.jaw_prior(
                        body_model_output.jaw_pose.mul(
                            self.jaw_prior_weight)))

        # ==== shape prior / measurements loss ====
        shape_loss = 0.0
        if self.shape_weight > 0:
            # smplify-x shape prior
            shape_prior_loss = torch.sum(self.shape_prior(
                body_model_output.betas)) ** 2

            if gt_height is not None or gt_weight is not None:
                # get fitted height and mass from betas
                # shape_components = torch.cat([body_model_output.betas,
                #                        body_model_output.expression], dim=-1)
                betas = body_model_output.betas if body_model_output.betas is not None else self.body_model.betas
                v_shaped = self.body_model.v_template + \
                       lbs.blend_shapes(betas,
                                        self.body_model.shapedirs)
                fitted_measurements = self.measurements_crit(v_shaped)
                f_height = fitted_measurements['height']
                f_mass = fitted_measurements['mass']
                # shape loss
                measurements_loss = torch.exp(100*(abs(f_height-gt_height))) + \
                            torch.exp(abs(f_mass-gt_weight))
                shape_loss = (shape_prior_loss + measurements_loss) * self.shape_weight
            else:
                shape_loss = shape_prior_loss * self.shape_weight

        # ==== self contact loss for mimic'ed poses ====
        msc_loss = 0.0
        if self.use_contact and self.mimicked_contact_weight.item() > 0:
            msc_loss = self.msc_crit(presented_contact =
                                        self.presented_contact,
                                        vertices=body_model_output.vertices)
            msc_loss = self.mimicked_contact_weight.item() * msc_loss

        # ==== general self contact loss ====
        faces_angle_loss, gsc_contact_loss = 0.0, 0.0
        if self.use_contact and self.scopti_weight.item() > 0:
            gsc_contact_loss, faces_angle_loss = \
                self.sc_crit(vertices=body_model_output.vertices)
            gsc_contact_loss = self.scopti_weight.item() * gsc_contact_loss
            faces_angle_loss = 0.1 * faces_angle_loss
        
        # ==== final loss value ====
        total_loss = (joint_loss + global_orient_xz_loss + shape_loss + \
                      body_pose_prior_loss + \
                      msc_loss + gsc_contact_loss + faces_angle_loss + \
                      left_hand_prior_loss + right_hand_prior_loss + \
                      jaw_prior_loss + expression_loss)

        # ==== print loss values ====
        losses = [[total_loss],
                    [joint_loss],
                    [body_pose_prior_loss],
                    [msc_loss, gsc_contact_loss, faces_angle_loss],
                    [shape_loss],
                    [left_hand_prior_loss, right_hand_prior_loss],
                    [jaw_prior_loss, expression_loss],
                    [global_orient_xz_loss]]
        losses = [[np.round(x.item(),2) if torch.is_tensor(x) else np.round(x,2) for x in y] for y in losses]
        print(losses[0], '||', losses[1], '|', losses[2], '|', losses[3], '|', losses[4], '|',
              losses[5], '|', losses[6], '|', losses[7])

        return total_loss
