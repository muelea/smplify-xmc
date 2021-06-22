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
from lib.utils.geometry import estimate_translation_np
from data.global_orient import go_map

def select_keypoints(keypoints, strategy, thres=0.3):
    """ 
    Returns the keypoints + their index either based on detection confidence
    or for a fixed set. Use fixed only when the full body is clearly visible.
    """
    if strategy == 'confidence':
        kpts_idx = np.where(keypoints[0,:,2] > thres)[0]
        kpts_val = keypoints[0,kpts_idx,:2]
        kpts_conf = np.ones(len(kpts_idx))
    if strategy == 'fixed':
        kpts_idx = [0, 2,  3,  4,  5,  6,  7,  8, 10, 11, 13, \
            14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        kpts_val = keypoints[0,kpts_idx,:2]
        kpts_conf = keypoints[0,kpts_idx,2]
    return kpts_idx, kpts_val, kpts_conf


def select_global_orient_keys(strategy):
    """
    Returns the keys of the go_map for the global orientations that are tested 
    to initialize the fitting. 
    strategy: all - 360 degree  
    frontal: uses the frontal views of the mesh only (i.e. the MTP presented global orientations)
    """
    if strategy == 'all':
        gos_test = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
    elif strategy == 'frontal':
        gos_test = ['0', '1', '2', '3', '4', '5', '6']
    return gos_test


def best_fit_presented_global_orientation(keypoints, camera, mesh_fn, body_model,
        focal_length_x, focal_length_y, H, W, 
        kpts_selection_stratey='fixed', 
        global_orientation_stratey='frontal'):
    """
    Return the best fit form a set of global orientations and the estimated translation.
    """

    device = camera.center.device
    kpts = keypoints.detach().cpu().numpy()

    # find the best presented global orientation (go) to initialize model orientation
    # test both global orientations, in case image was flipped
    gos_test = select_global_orient_keys(global_orientation_stratey)
    translations = []
    init_joint_losses = []
    for go in gos_test:
        global_orient = torch.tensor([go_map[go]])
        body_model.global_orient[:] = torch.tensor(global_orient) \
                                        .view_as(body_model.global_orient) \
                                        .clone().detach().requires_grad_(True)
        body = body_model()
        rot_joints = body.joints.detach()
        kpts_idx, kpts_val, kpts_conf = select_keypoints(kpts, kpts_selection_stratey)
        ts = estimate_translation_np(rot_joints[0,kpts_idx,:].detach().cpu().numpy(),
                kpts_val, kpts_conf, focal_length_x=focal_length_x,
                focal_length_y=focal_length_y, W=W, H=H)

        camera.translation[:] = torch.tensor(ts).view_as(camera.translation) \
                                  .clone().detach().requires_grad_(True)
        translations += [ts]
        # get initial joint loss
        projected_joints = camera(rot_joints)
        joint_diff = torch.tensor(kpts[:,:,:2]).to(device).float() - projected_joints[:,:,:]
        joint_loss = torch.norm(torch.tensor(kpts[:,:,2]).to(device).float().unsqueeze(dim=-1) ** 2 * \
            joint_diff, dim=-1).sum()
        init_joint_losses += [joint_loss.item()]

    # reset camera and body model params to with initial translation and global orient
    init_idx = np.argmin(np.array(init_joint_losses))
    init_translation = translations[init_idx]
    init_global_orient = go_map[gos_test[init_idx]]

    return init_translation, init_global_orient


# you can also run ePNP to solve for global orientation.
# Our experience is that this does not work as well as using the
# presented global orientations. To try it, use the next few lines of code
#import cv2
#from lib.utils.geometry import batch_estimate_global_orient
#body = body_model()
#Rs, ts = batch_estimate_global_orient_ePnP(body, keypoints, W, H,
#         focal_length, focal_length, None)
#rvec, _ = cv2.Rodrigues(Rs[0])
#global_orient = torch.tensor(rvec).view(-1,3).float()

def estimate_global_orient_ePnP(joints_3d, joints_2d, cx, cy, focal_length_x,
 focal_length_y, tvec=None):

    import cv2
    import numpy as np

    assert joints_3d.shape[0] == joints_2d.shape[0], \
           "num joints and keypoints must be equal"

    camera_mat = np.zeros([3, 3])
    camera_mat[0,0] = focal_length_x
    camera_mat[1,1] = focal_length_y
    camera_mat[0,2] = cx
    camera_mat[1,2] = cy
    camera_mat[2,2] = 1

    if tvec is None:
        tvec = np.zeros((3,1))

    _, R_cal, t = cv2.solvePnP(joints_3d, joints_2d, camera_mat,
                                np.zeros(4),np.zeros((3,1)), tvec,
                                useExtrinsicGuess=False,
                                flags=cv2.SOLVEPNP_EPNP)

    R, _ = cv2.Rodrigues(R_cal)

    return R, t


def batch_estimate_global_orient_ePnP(body, keypoints, img_W, img_H, focal_length_x, focal_length_y, tvec=None):
    batch_size = keypoints.shape[0]
    Rs = np.zeros((batch_size, 3, 3))
    ts = np.zeros((batch_size, 3, 1))

    # use major body joints, ignore neck and left / right hip
    use_joints = [0, 2,  3,  4,  5,  6,  7,  8, 10, 11, 13, 14, 15, 16, 17, 18, 19,
                  20, 21, 22, 23, 24]
    keypoints = keypoints[:,use_joints,:]
    joints = body.joints[:,use_joints,:]

    for idx in range(keypoints.shape[0]):
        if tvec is not None:
            tvec = tvec[idx,:].reshape(3,1)

        conf = np.where(keypoints[idx,:,2].cpu().numpy() > 0)[0].tolist()
        joints_3d = joints[idx].detach().cpu().numpy().astype(np.float32)
        joints_2d = keypoints[idx,:,:2].detach().cpu().numpy().astype(np.float32)

        if len(conf) > 3:
            R, t = estimate_global_orient_ePnP(joints_3d[conf,:], joints_2d[conf,:],
                   img_W/2, img_H/2, focal_length_x, focal_length_y, tvec)
            Rs[idx] = R
            ts[idx] = t
        else:
            print(f'Error for {idx} not enough keypoints found.')

    return Rs, ts