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

import sys
import os
import glob

# for cluster rendering
import os.path as osp

import time
import yaml
import torch
import pickle
import numpy as np

import smplx
from selfcontact import SelfContact


from lib.utils.utils import JointMapper
from configs.cmd_parser import parse_config
from lib.dataextra.data_loader import create_dataset
from lib.core.fit_single_frame import fit_single_frame

from lib.core.camera import create_camera
from lib.utils.loadpriors import load_all_priors
from configs.dbsconfig import *

torch.backends.cudnn.enabled = False
os.environ['PYOPENGL_PLATFORM'] = 'egl'

def main(**args):

    start = time.time()

    # create output folders
    output_folder = osp.expandvars(args.pop('output_dir'))
    if not osp.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
        for f in ['results', 'meshes', 'images']:
            os.makedirs(osp.join(output_folder, f), exist_ok=True)

    # save arguments of current experiment
    conf_fn = osp.join(output_folder, 'conf.yaml')
    with open(conf_fn, 'w') as conf_file:
        yaml.dump(args, conf_file)

    # get device and set dtype
    dtype = torch.float32
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # create Dataset from db file if available else create from folders
    dataset_obj = create_dataset(
                    dataset=args.pop('dataset'),
                    output_folder=output_folder,
                    input_base_dir=args.pop('input_base_dir', ''),
                    input_dir_images=args.pop('input_dir_images'),
                    input_dir_keypoints=args.pop('input_dir_keypoints', ''),
                    input_dir_poses=args.pop('input_dir_poses', ''),
                    **args
    )

    joint_mapper = JointMapper(dataset_obj.get_model2data())
    num_betas = args.get('num_shape_comps')

    # creat smplx model
    model_type = args.pop('model_type', 'smplx')
    body_model_params = dict(model_path=args.get('model_folder'),
                        model_type=model_type,
                        joint_mapper=joint_mapper,
                        create_global_orient=True,
                        create_body_pose=True,
                        create_betas=True,
                        num_betas=num_betas,
                        create_left_hand_pose=True,
                        create_right_hand_pose=True,
                        create_expression=True,
                        create_jaw_pose=True,
                        create_leye_pose=True,
                        create_reye_pose=True,
                        create_transl=True,
                        dtype=dtype,
                        **args)

    # Create the camera object
    use_calibration = args.pop('use_calibration')
    focal_length = args.pop('focal_length')
    focal_length_x, focal_length_y = focal_length, focal_length
    camera = create_camera(focal_length_x=focal_length_x,
                           focal_length_y=focal_length_y,
                           dtype=dtype,
                           **args)
    camera = camera.to(device=device)

    # if hands and face should be used
    use_hands = args.get('use_hands', True)
    use_face = args.get('use_face', True)

    # create priors
    angle_prior, shape_prior, expr_prior, \
    jaw_prior, left_hand_prior, right_hand_prior = \
        load_all_priors(args, dtype, use_face, use_hands, device)

    # A weight for every joint of the model
    joint_weights = dataset_obj.get_joint_weights()\
                        .to(device=device, dtype=dtype)
    # Add a fake batch dimension for broadcasting
    joint_weights.unsqueeze_(dim=0)

    # module for self-contact
    geo_thres = args.pop('geo_thres', 0.3)
    eucl_thres = args.pop('geo_thres', 0.02)
    essentials_dir = args.pop('essentials_dir', 'data/essentials')
    sc_module = SelfContact( 
            essentials_folder=essentials_dir,
            geothres=geo_thres, 
            euclthres=eucl_thres, 
            model_type=model_type,
            test_segments=True,
            compute_hd=True
        )
    
    # file for measurement loss 
    measurements_path = osp.join(essentials_dir, 
        f'measurements/{model_type}/{model_type}_measurements.yaml')

    # params how to initialize the fitting
    kpts_strategy = args.pop('kpts_sel_strategy', 'confidence')
    go_strategy = args.pop('go_sel_strategy', 'all')
    use_presented_hand_pose = args.get('use_presented_hand_pose')

    # select index for cluster usage
    print('Dataset has {} objects'.format(len(dataset_obj)))
    idxs = np.arange(len(dataset_obj))
    cbs = len(dataset_obj) if args.get('cluster_bs') is None else args.get('cluster_bs')
    sidx = args.get('ds_start_idx')
    
    for idx in idxs[sidx*cbs: sidx*cbs+cbs]:
        # read data
        data = dataset_obj[idx]
        print('Processing: {}'.format(data['img_path']))

        # create subdir to write output
        for f in ['results', 'meshes', 'images']:
            curr_output_folder = osp.join(output_folder, f, data['relpath'])
            os.makedirs(curr_output_folder, exist_ok=True)

        # get params for dataset
        # image details
        img = data['img']
        fn = data['fn']
        keypoints = data['keypoints']

        # subject details
        height = data['height']
        weight = data['weight']

        # presented pose and contact details, that are used to initialize fitting
        presented_contact = data['pp_contact']
        presented_body_pose = data['pp_body_pose']

        mean_hand = torch.zeros((1, body_model_params['num_pca_comps']))
        presented_left_hand_pose =  data['pp_left_hand_pose'] \
            if use_presented_hand_pose else mean_hand
        presented_right_hand_pose = data['pp_right_hand_pose'] \
            if use_presented_hand_pose else mean_hand

        if use_calibration:
            focal_length_x = data['focal_length_x']
            focal_length_y = data['focal_length_y']

        # process each person (usually only one person visible)
        for person_id in range(keypoints.shape[0]):
            # output filenames for person
            curr_mesh_fn = osp.join(osp.join(output_folder, 'meshes', data['relpath'], 
                fn+'_{:03d}.obj'.format(person_id)))
            curr_img_fn = osp.join(osp.join(output_folder, 'images', data['relpath'], 
                fn+'_{:03d}.png'.format(person_id)))
            curr_result_fn = osp.join(osp.join(output_folder, 'results', data['relpath'], 
                fn+'_{:03d}.pkl'.format(person_id)))

            # read gender and select model
            gender_pid = 'neutral' if data['gender'][person_id] is None \
                             else data['gender'][person_id]
            if gender_pid == 'neutral':
                body_model = smplx.create(gender='neutral', **body_model_params).to(device)
            elif gender_pid == 'female':
                body_model = smplx.create(gender='female', **body_model_params).to(device)
            elif gender_pid == 'male':
                body_model = smplx.create(gender='male', **body_model_params).to(device)

            # start fitting
            fit_single_frame(img,
                             keypoints[[person_id]],
                             height=height[person_id],
                             weight=weight[person_id],
                             presented_contact=presented_contact,
                             presented_body_pose=presented_body_pose,
                             presented_left_hand_pose=presented_left_hand_pose,
                             presented_right_hand_pose=presented_right_hand_pose,
                             body_model=body_model,
                             sc_module=sc_module,
                             camera=camera,
                             use_calibration=use_calibration,
                             focal_length_x=focal_length_x,
                             focal_length_y=focal_length_y,
                             joint_weights=joint_weights,
                             left_hand_prior=left_hand_prior,
                             right_hand_prior=right_hand_prior,
                             jaw_prior=jaw_prior,
                             shape_prior=shape_prior,
                             expr_prior=expr_prior,
                             angle_prior=angle_prior,
                             out_img_fn=curr_img_fn,
                             result_fn=curr_result_fn,
                             mesh_fn=curr_mesh_fn,
                             kpts_sel_strategy=kpts_strategy,
                             go_sel_strategy=go_strategy,
                             measurements_path=measurements_path,
                             **args)

    elapsed = time.time() - start
    time_msg = time.strftime('%H hours, %M minutes, %S seconds',
                             time.gmtime(elapsed))
    print('Processing the data took: {}'.format(time_msg))


if __name__ == "__main__":
    args = parse_config()
    main(**args)
