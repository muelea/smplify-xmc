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

import pickle

import sys
import os
import os.path as osp

import numpy as np
import torch
import torchgeometry as tgm
from tqdm import tqdm
import trimesh

from collections import defaultdict

from lib.optimizers import optim_factory
from lib.utils import loadweights
from lib.core import fitting
from lib.core.losses import SMPLifyXMCLoss
from lib.rendering.render_singlemesh import overlay_mesh
from lib.utils.estimate_global_orient import best_fit_presented_global_orientation
from lib.utils.save_results import save_results

def fit_single_frame(img,
                     keypoints,
                     height,
                     weight,
                     body_model,
                     sc_module,
                     camera,
                     use_calibration,
                     joint_weights,
                     jaw_prior,
                     left_hand_prior,
                     right_hand_prior,
                     shape_prior,
                     expr_prior,
                     angle_prior,
                     presented_contact,
                     presented_body_pose,
                     presented_left_hand_pose,
                     presented_right_hand_pose,
                     out_img_fn='out.png',
                     result_fn='out.pkl',
                     mesh_fn='out.obj',
                     use_face=True,
                     use_hands=True,
                     data_weights=None,
                     body_pose_prior_weights=None,
                     hand_pose_prior_weights=None,
                     jaw_pose_prior_weights=None,
                     shape_weights=None,
                     expr_weights=None,
                     hand_joints_weights=None,
                     face_joints_weights=None,
                     scopti_weights=None,
                     camera_rot_weights=None,
                     body_global_orient_weights=None,
                     focal_length_weights=None,
                     mimicked_contact_weights=None,
                     focal_length_x=5000.,
                     focal_length_y=5000.,
                     rho=100,
                     use_joints_conf=False,
                     use_contact=True,
                     kpts_sel_strategy='confidence',
                     go_sel_strategy='frontal',
                     gen_sc_inside_weight=0.5,
                     gen_sc_outside_weight=0.0,
                     gen_sc_contact_weight=0.5,
                     measurements_path='',
                     **kwargs):

    batch_size = 1
    device = torch.device('cuda') if torch.cuda.is_available() \
             else torch.device('cpu')
    dtype = torch.float32

    # reset camera parameters
    new_camera_params = dict(
        focal_length_x = focal_length_x * torch.ones([batch_size, 1],
            dtype=dtype).clone().detach().requires_grad_(True),
        focal_length_y = focal_length_y * torch.ones([batch_size, 1],
            dtype=dtype).clone().detach().requires_grad_(True),
        rotation = torch.eye(3, dtype=dtype).unsqueeze(dim=0) \
            .repeat(batch_size, 1, 1).clone().detach().requires_grad_(False)
    )

    # if no initial pose is given, start from mean pose
    new_params = defaultdict(
        body_pose = presented_body_pose.clone().detach().requires_grad_(True),
        left_hand_pose = presented_left_hand_pose.clone().detach().requires_grad_(True),
        right_hand_pose = presented_right_hand_pose.clone().detach().requires_grad_(True)
    )

    # process keypoints
    keypoint_data = torch.tensor(keypoints, dtype=dtype, device=device).clone().detach()
    gt_joints = keypoint_data[:, :, :2]
    joints_conf = keypoint_data[:, :, 2].reshape(1, -1)

    # create body fitting loss
    loss = SMPLifyXMCLoss(
                body_model=body_model,
                sc_module=sc_module,
                presented_global_orient=torch.zeros(batch_size, 3),
                presented_contact=presented_contact,
                presented_body_pose=presented_body_pose,
                presented_left_hand_pose=presented_left_hand_pose,
                presented_right_hand_pose=presented_right_hand_pose,
                shape_prior=shape_prior,
                angle_prior=angle_prior,
                expr_prior=expr_prior,
                left_hand_prior=left_hand_prior,
                right_hand_prior=right_hand_prior,
                jaw_prior=jaw_prior,
                joint_weights=joint_weights,
                use_joints_conf=use_joints_conf,
                use_face=use_face,
                use_hands=use_hands,
                use_contact=use_contact,
                rho=rho,
                dtype=dtype,
                gen_sc_inside_weight=gen_sc_inside_weight,
                gen_sc_outside_weight=gen_sc_outside_weight,
                gen_sc_contact_weight=gen_sc_contact_weight,
                measurements_path=measurements_path,
                **kwargs)
    loss = loss.to(device=device)

    # load weights
    opt_weights = loadweights.load_weights(**locals())

    with fitting.FittingMonitor(**kwargs) as monitor:

        img = torch.tensor(img, dtype=dtype)
        H, W, _ = img.shape
        img_data_weight = 1000 / H

        # Reset the parameters to estimate the initial translation of the
        # body model. Set to initial pose if available, else use mean pose
        body_model.reset_params(**new_params)
        camera.reset_params(**new_camera_params)
        camera.center[:] = torch.tensor([W, H], dtype=dtype) * 0.5

        #######################################################################
        # get best global orientation and translation
        with torch.no_grad():
            init_translation, init_global_orient = \
                  best_fit_presented_global_orientation(keypoints, camera, \
                  mesh_fn, body_model, focal_length_x, focal_length_y, H, W,
                  kpts_selection_stratey=kpts_sel_strategy, 
                  global_orientation_stratey=go_sel_strategy)
            camera.translation[:] = torch.tensor(init_translation) \
                .view_as(camera.translation).clone().detach().requires_grad_(True)
            body_model.global_orient[:] = torch.tensor(init_global_orient) \
               .view_as(body_model.global_orient).clone().detach().requires_grad_(True)
            loss.init_global_orient = torch.tensor(init_global_orient).to('cuda') \
                .float().unsqueeze(0)

        #######################################################################
        # start fitting
        if use_calibration:
            all_optiparamnames = {
              0: ['global_orient', 'translation', 'betas', 'pitch', 'roll', 'yaw'],
              1: ['body_pose', 'left_hand_pose', 'translation', 'right_hand_pose', 'betas', 'pitch', 'roll', 'yaw'],
              2: ['body_pose', 'left_hand_pose', 'translation', 'right_hand_pose', 'betas', 'pitch', 'roll', 'yaw']
          }
        else:
            all_optiparamnames = {
              0: ['global_orient', 'translation', 'betas', 'pitch', 'roll', 'yaw', 'focal_length_x', 'focal_length_y'],
              1: ['body_pose', 'left_hand_pose', 'translation', 'right_hand_pose', 'betas', 'pitch', 'roll', 'yaw', 'focal_length_x', 'focal_length_y'],
              2: ['body_pose', 'left_hand_pose', 'translation', 'right_hand_pose', 'betas', 'pitch', 'roll', 'yaw', 'focal_length_x', 'focal_length_y']
          }

        for opt_idx, curr_weights in enumerate(tqdm(opt_weights, desc='Stage')):
            optiparamnames = all_optiparamnames[opt_idx]
            final_params = [x[1] for x in body_model.named_parameters() \
                               if x[0] in optiparamnames and x[1].requires_grad] + \
                           [x[1] for x in camera.named_parameters() \
                               if x[0] in optiparamnames and x[1].requires_grad]

            # set weights - ToDo: check data_weight
            curr_weights['data_weight'] *= img_data_weight
            curr_weights['bending_prior_weight'] = (3.17 * curr_weights['body_pose_weight'])
            if use_hands:
                joint_weights[:, 25:67] = curr_weights['hand_weight']
            if use_face:
                joint_weights[:, 67:] = curr_weights['face_weight']
            loss.reset_loss_weights(curr_weights)

            body_optimizer, body_create_graph = optim_factory.create_optimizer(
                        final_params, **kwargs)
            body_optimizer.zero_grad()

            # create closure for body fitting
            closure = monitor.create_fitting_closure(
                        body_optimizer,
                        body_model,
                        height=height, weight=weight,
                        camera=camera,
                        gt_joints=gt_joints,
                        joints_conf=joints_conf,
                        joint_weights=joint_weights,
                        loss=loss,
                        create_graph=body_create_graph,
                        return_verts=True,
                        return_full_pose=True)

            final_loss_val = monitor.run_fitting(
                                   body_optimizer,
                                   closure,
                                   final_params,
                                   body_model)

            # fitting stage X is done, prepare data for output of stage X
            model_output = body_model(return_verts=True)
            vertices = model_output.vertices.detach().cpu().numpy().squeeze()

        #######################################################################
        # fitting done save final result
        if final_loss_val is not np.isnan(final_loss_val):
            save_results(
                camera, body_model, vertices, keypoints, img,
                mesh_fn, result_fn, out_img_fn
            )