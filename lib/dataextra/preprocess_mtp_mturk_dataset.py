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

import argparse
import pickle
import sys
import glob
import os.path as osp
import numpy as np
import cv2
import json
from tqdm import tqdm

def read_keypoints(keypoint_fn):
    with open(keypoint_fn) as keypoint_file:
        data = json.load(keypoint_file)

    keypoints = []
    for idx, person_data in enumerate(data['people']):
        body_keypoints = np.array(
            person_data['pose_keypoints_2d'],
            dtype=np.float32).reshape([-1, 3])

        left_hand_keyp = np.array(
            person_data['hand_left_keypoints_2d'],
            dtype=np.float32).reshape([-1, 3])

        right_hand_keyp = np.array(
            person_data['hand_right_keypoints_2d'],
            dtype=np.float32).reshape([-1, 3])

        face_keypoints = np.array(
            person_data['face_keypoints_2d'],
            dtype=np.float32).reshape([-1, 3])[17: 17 + 51, :]

        contour_keyps = np.array(
            person_data['face_keypoints_2d'],
            dtype=np.float32).reshape([-1, 3])[:17, :]

        body_keypoints = np.concatenate([body_keypoints, left_hand_keyp,
            right_hand_keyp, face_keypoints, contour_keyps], axis=0)

        keypoints.append(body_keypoints)
    return keypoints


def select_keypoints(keypoints, img):
    img_center = np.array(img.shape[:2])/2 #height, width
    # select keypoints closest to image center weighted by inverse confidence
    if len(keypoints) > 1:
        kpselect = np.inf * np.ones(len(keypoints))
        # major joints openpose
        op_to_12 = [9, 10, 11, 12, 13, 14, 2, 3, 4, 5, 6, 7]
        for idx, personkpts in enumerate(keypoints):
            kpdist = personkpts[op_to_12, :2] - img_center
            kpdist = np.linalg.norm(kpdist, axis=1)
            kpconf = np.dot(kpdist, (- personkpts[op_to_12, 2] + 1))
            kpselect[idx] = kpconf
        kpselidx = np.argmin(kpselect)
    elif len(keypoints) == 1:
        kpselidx = 0
    else:
        keypoints = None
    keypoints = np.stack(keypoints[kpselidx:kpselidx+1])
    return keypoints

def mtp_extra(ds_folder):

    imgpaths_ = []
    imgnames_ = []
    subject_ids_ = []
    relpaths_ = []
    heights_ = []
    weights_ = []
    genders_ = []
    keypoints_ = []
    pp_contacts_ = []
    pp_body_poses_ = []
    pp_left_hand_poses_ = []
    pp_right_hand_poses_ = []

    subject_meta_path = osp.join(ds_folder, 'subject_meta.json')
    with open(subject_meta_path, 'r') as f:
        subject_meta = json.load(f)

    # load images, loop through and read smplx and keypoints
    img_dir = osp.join(ds_folder, 'images')
    images = glob.glob(osp.join(img_dir, '**', '*.png'), recursive=True)

    for img_path in tqdm(images):
        img = cv2.imread(img_path).astype(np.float32)[:, :, ::-1] / 255.0
        img_fn, _ = osp.splitext(osp.split(img_path)[1])
        img_id, pp_subset, pp_id, subj_id = img_fn.split('_')
        relpath = osp.dirname(img_path).replace(img_dir, '').strip('/')

        # read mimic'ed pose smplx parameters
        pp_dir = osp.join(ds_folder, 'presented_pose', 'params', pp_subset)
        pp_path = osp.join(pp_dir, pp_id+'.npz')
        pp_params = np.load(pp_path)
        #pp_global_orient = subject_meta[img_fn]['pp_body_global_orient']

        # read and find 2d keypoints
        keypoint_dir = osp.join(ds_folder, 'keypoints', 'openpose', pp_subset)
        keypoint_fn = osp.join(keypoint_dir, img_fn+'.json')
        keypoints = read_keypoints(keypoint_fn)
        keypoints = select_keypoints(keypoints, img)

        # read height, weight, gender meta
        height = subject_meta[img_fn]['SubjectHeightMeter']
        weight = subject_meta[img_fn]['SubjectWeightKg']
        gender = subject_meta[img_fn]['SubjectGender']

        # add to dict
        imgpaths_.append(img_path)
        imgnames_.append(img_fn)
        subject_ids_.append(subj_id)
        relpaths_.append(relpath)
        heights_.append(height)
        weights_.append(weight)
        genders_.append(gender)
        keypoints_.append(keypoints[0].tolist())
        pp_contacts_.append(pp_params['contact'].tolist())
        pp_body_poses_.append(pp_params['body_pose'].tolist())
        pp_left_hand_poses_.append(pp_params['left_hand_pose'].tolist())
        pp_right_hand_poses_.append(pp_params['right_hand_pose'].tolist())

    # save as db file
    outfile = osp.join('data','dbs','mtp_mturk.npz')
    np.savez(outfile, 
        imgpath=imgpaths_,
        subject_id = subject_ids_,
        imgname = imgnames_,
        relpath=relpaths_,
        height=heights_,
        weight=weights_,
        gender=genders_,
        keypoint=keypoints_,
        contact=pp_contacts_,
        body_pose=pp_body_poses_,
        left_hand_pose=pp_left_hand_poses_,
        right_hand_pose=pp_right_hand_poses_
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds_dir', required=True,
                        help='the mtp dataset folder, e.g. /home/dataset/mtp')
    args = parser.parse_args()
    mtp_extra(ds_folder=args.ds_dir)