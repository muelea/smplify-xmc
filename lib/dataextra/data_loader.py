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

import glob
import os.path as osp

import json
import pickle

from collections import namedtuple

import cv2
import numpy as np

import torch
from torch.utils.data import Dataset
import yaml

from data import openpose
from lib.utils.utils import smpl_to_openpose
from selfcontact import SelfContactSmall
from configs.dbsconfig import *


Keypoints = namedtuple('Keypoints',
                ['keypoints', 'gender_gt', 'gender_pd'])

Keypoints.__new__.__defaults__ = (None,) * len(Keypoints._fields)


def create_dataset(
        dataset, 
        output_folder,
        input_dir_images=None,
        input_dir_keypoints=None,
        input_base_dir=None,
        input_dir_poses=None, 
        db_file=None, 
        **kwargs):
        
    if dataset.lower() == 'mtp':
        return MTP(
                    db_file, 
                    output_folder, 
                    **kwargs
        )
    elif dataset.lower() == 'mtp_demo':
        return MTP_DEMO(
                    input_base_dir,
                    input_dir_images,
                    input_dir_keypoints,
                    input_dir_poses, 
                    **kwargs
        )
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))

def get_scale_factor(size, new_size):
    scale_factor = new_size / size

    # if aspect ration should be kept
    if (new_size == -1).any():
        #aspect_ratio = size[0] / size[1]
        update = np.where(new_size == -1)[0][0]
        keep =  np.where(new_size != -1)[0][0]
        scale_factor[update] = scale_factor[keep]
    return scale_factor

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

def select_center_person(keypoints, img):
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
        keypoints = np.stack(keypoints[kpselidx:kpselidx+1])
    elif len(keypoints) == 1:
        kpselidx = 0
        keypoints = np.stack(keypoints[kpselidx:kpselidx+1])
    else:
        keypoints = None
    return keypoints

def select_keypoints(keypoints, use_hands=True, use_face=True,
                     use_face_contour=False):

    """For mimicthepose datasets we read all keypoints. Select only the
    keypoints that should be used in optimization"""

    bkn = openpose.body_keypoints_num
    hkn = openpose.left_hand_keyp_num + openpose.right_hand_keyp_num
    fkn = openpose.face_keyp_num
    fckn = openpose.face_contour_keyp_num

    idx_body = np.arange(bkn)
    idx_hands = np.arange(bkn, bkn + hkn) if use_hands else np.arange(0)
    idx_face = np.arange(bkn+hkn, bkn+hkn+fkn) if use_face else np.arange(0)
    idx_face_contour = np.arange(bkn+hkn+fkn, bkn+hkn+fkn+fckn) \
                       if use_face_contour else np.arange(0)

    indexes = np.hstack((idx_body, idx_hands, idx_face, idx_face_contour))
    keypoints = keypoints[:, indexes, :]

    return keypoints

class MTP(Dataset):

    NUM_BODY_JOINTS = 25
    NUM_HAND_JOINTS = 20

    def __init__(self,
                 dbspath,
                 output_folder=None,
                 use_hands=False,
                 use_face=False,
                 dtype=torch.float32,
                 model_type='smplx',
                 use_face_contour=False,
                 joints_to_ign = None,
                 openpose_format='coco25',
                 norm_height=500,
                 **kwargs):
        super(MTP, self).__init__()

        self.use_hands = use_hands
        self.use_face = use_face
        self.model_type = model_type
        self.dtype = dtype
        self.joints_to_ign = joints_to_ign
        self.use_face_contour = use_face_contour
        self.openpose_format = openpose_format
        self.num_joints = (self.NUM_BODY_JOINTS +
                           2 * self.NUM_HAND_JOINTS * use_hands)
        self.cnt = 0
        # allow pickle because length of contact arrays not equal
        self.db = np.load(dbspath, allow_pickle=True)
        self.img_paths = self.db['imgpath']

        self.norm_height = norm_height

    def get_model2data(self):
        return smpl_to_openpose(self.model_type, use_hands=self.use_hands,
                                use_face=self.use_face,
                                use_face_contour=self.use_face_contour,
                                openpose_format=self.openpose_format)

    def get_joint_weights(self):
        # The weights for the joint terms in the optimization
        optim_weights = np.ones(self.num_joints + 2 * self.use_hands +
                                self.use_face * 51 +
                                17 * self.use_face_contour,
                                dtype=np.float32)

        # These joints are ignored becaue SMPL has no neck.
        if self.joints_to_ign is not None and -1 not in self.joints_to_ign:
            optim_weights[self.joints_to_ign] = 0.

        # put higher weights on knee and elbow joints for mimic'ed poses
        optim_weights[[3,6,10,13]] = 2

        return torch.tensor(optim_weights, dtype=self.dtype)

    def __iter__(self):
        return self

    def __next__(self):
        if self.cnt >= len(self.img_paths):
            raise StopIteration

        self.cnt += 1

        return self.read_item(self.cnt - 1)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        return self.read_item(idx)

    def read_item(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path).astype(np.float32)[:, :, ::-1] / 255.0

        keypoints = torch.from_numpy(self.db['keypoint'][idx]).unsqueeze(0)
        keypoints = select_keypoints(keypoints, self.use_hands,
                                     self.use_face, self.use_face_contour)

        if self.norm_height is not None:
            # resize image to height 500, keep aspect ratio
            new_size = np.array([self.norm_height, -1])
            scale_factor = get_scale_factor(img.shape[:2], new_size)
            new_size = (img.shape[:2]*scale_factor).astype(np.int)
            img = cv2.resize(img, dsize=(new_size[1], new_size[0]))
            # resize keypoints accordingly
            keypoints[:,:,0] = keypoints[:,:,0] * scale_factor[1]
            keypoints[:,:,1] = keypoints[:,:,1] * scale_factor[0]

        contact = torch.tensor(self.db['contact'][idx])
        contact = contact.flatten().unique()
        contact_type = 'v2v'

        body_pose = torch.from_numpy(np.array(self.db['body_pose'][idx])).unsqueeze(0).float()
        left_hand_pose = torch.from_numpy(np.array(self.db['left_hand_pose'][idx])).unsqueeze(0).float()
        right_hand_pose = torch.from_numpy(np.array(self.db['right_hand_pose'][idx])).unsqueeze(0).float()

        # gender, height and weight
        gender = self.db['gender'][[idx]]
        height = self.db['height'][[idx]]
        weight = self.db['weight'][[idx]]

        output_dict = {'img_path':img_path,
                       'fn': self.db['imgname'][idx],
                       'relpath': self.db['relpath'][idx],
                       'keypoints': keypoints,
                       'img': img,
                       'gender': gender,
                       'height': height,
                       'weight': weight,
                       'pp_body_pose': body_pose,
                       'pp_left_hand_pose':left_hand_pose,
                       'pp_right_hand_pose':right_hand_pose,
                       'pp_contact': contact,
                       'contact_type': contact_type}
        return output_dict


class MTP_DEMO(Dataset):

    NUM_BODY_JOINTS = 25
    NUM_HAND_JOINTS = 20

    def __init__(self,
                input_base_dir,
                image_folder,
                keypoint_folder,
                presented_poses_folder,
                use_hands=False,
                use_face=False,
                dtype=torch.float32,
                model_type='smplx',
                use_face_contour=False,
                joints_to_ign = None,
                openpose_format='coco25',
                norm_height=500,
                essentials_dir='data/essentials',
                **kwargs):
        super(MTP_DEMO, self).__init__()

        self.use_hands = use_hands
        self.use_face = use_face
        self.model_type = model_type
        self.dtype = dtype
        self.joints_to_ign = joints_to_ign
        self.use_face_contour = use_face_contour
        self.openpose_format = openpose_format
        self.num_joints = (self.NUM_BODY_JOINTS +
                           2 * self.NUM_HAND_JOINTS * use_hands)
        self.cnt = 0
        self.norm_height = norm_height

        self.input_base_dir = input_base_dir
        self.input_dir_images = osp.join(input_base_dir, image_folder)
        self.input_dir_keypoints = osp.join(input_base_dir, keypoint_folder)
        self.input_dir_poses = presented_poses_folder

        self.img_paths = sorted(glob.glob(osp.join(self.input_dir_images,
                            '**'), recursive=True))
        self.img_paths = [x for x in filter(lambda x: x.split('.')[-1] \
            in ['png', 'jpg', 'jpeg'], self.img_paths)][-10:]

        self.sc_module = SelfContactSmall( 
            essentials_folder=essentials_dir, 
            geothres=0.3, 
            euclthres=0.01, 
            model_type=model_type
        )

        # read subject metadata
        with open(osp.join(self.input_base_dir, 'meta.yaml'), 'r') as f:
            self.metadata = yaml.safe_load(f)

    def get_model2data(self):
        return smpl_to_openpose(self.model_type, use_hands=self.use_hands,
                                use_face=self.use_face,
                                use_face_contour=self.use_face_contour,
                                openpose_format=self.openpose_format)

    def get_joint_weights(self):
        # The weights for the joint terms in the optimization
        optim_weights = np.ones(self.num_joints + 2 * self.use_hands +
                                self.use_face * 51 +
                                17 * self.use_face_contour,
                                dtype=np.float32)

        # These joints are ignored becaue SMPL has no neck.
        if self.joints_to_ign is not None and -1 not in self.joints_to_ign:
            optim_weights[self.joints_to_ign] = 0.

        # put higher weights on knee and elbow joints for mimic'ed poses
        optim_weights[[3,6,10,13]] = 2

        return torch.tensor(optim_weights, dtype=self.dtype)

    def __iter__(self):
        return self

    def __next__(self):
        if self.cnt >= len(self.img_paths):
            raise StopIteration

        self.cnt += 1

        return self.read_item(self.cnt - 1)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        return self.read_item(idx)

    def read_item(self, idx):
        img_path = self.img_paths[idx]
        x = img_path.replace(self.input_dir_images, '').strip('/')
        rel_path, img_name_ext = osp.split(x)
        img_name, _ = img_name_ext.split('.')

        # read image
        img = cv2.imread(img_path).astype(np.float32)[:, :, ::-1] / 255.0

        # read keypoints
        kpf = osp.join(rel_path, img_name + '.json')
        keypoint_fn = osp.join(self.input_dir_keypoints, kpf)
        keypoints = read_keypoints(keypoint_fn)
        keypoints = select_center_person(keypoints, img)
        keypoints = select_keypoints(keypoints, self.use_hands,
                                     self.use_face, self.use_face_contour)
        keypoints = torch.from_numpy(keypoints).float()

        # resize image
        if self.norm_height is not None:
            # resize image to height 500, keep aspect ratio
            new_size = np.array([self.norm_height, -1])
            scale_factor = get_scale_factor(img.shape[:2], new_size)
            new_size = (img.shape[:2]*scale_factor).astype(np.int)
            img = cv2.resize(img, dsize=(new_size[1], new_size[0]))
            # resize keypoints accordingly
            keypoints[:,:,0] = keypoints[:,:,0] * scale_factor[1]
            keypoints[:,:,1] = keypoints[:,:,1] * scale_factor[0]

        # read presented pose
        pp_path = osp.join(self.input_dir_poses, rel_path + '.pkl')
        pp_data = pickle.load(open(pp_path, 'rb'))
        body_pose = torch.from_numpy(np.array(pp_data['body_pose'])).float()
        left_hand_pose = torch.from_numpy(np.array(pp_data['left_hand_pose'])).float()
        right_hand_pose = torch.from_numpy(np.array(pp_data['right_hand_pose'])).float()
        betas = torch.from_numpy(np.array(pp_data['betas'])).float().numpy() \
            if 'betas' in pp_data.keys() else None 
        presented_gender = torch.from_numpy(np.array(pp_data['gender'])).float().numpy() \
            if 'gender' in pp_data.keys() else None 

        # gender, height and weight
        gender = [self.metadata['gender']] if 'gender' in self.metadata.keys() \
            else ['neutral']
        height = [float(self.metadata['height'])] if 'height' in self.metadata.keys() \
            else [None]
        weight = [float(self.metadata['weight'])] if 'weight' in self.metadata.keys() \
            else [None]

        # check if contact available, if not compute contact from mesh
        contact_type = 'v2v'
        if 'contact' in pp_data.keys():
            contact = torch.from_numpy(np.array(pp_data['contact']))
        else:
            vertices = torch.tensor(pp_data['v']).view(1,-1,3)
            _, contact = self.sc_module.verts_in_contact(vertices, return_idx=True)
            # just one item in batch, so nothing to do here with batch_idx

        # get calibration
        focal_length_x, focal_length_y = [None], [None]
        f_pix, pitch, roll = None, None, None

        output_dict = {'img_path':img_path,
                       'fn': img_name,
                       'relpath': rel_path,
                       'keypoints': keypoints,
                       'img': img,
                       'gender': gender,
                       'pp_gender': presented_gender,
                       'pp_betas': betas,
                       'pp_contact': contact,
                       'contact_type': contact_type,
                       'height': height,
                       'weight': weight,
                       'pp_body_pose': body_pose,
                       'pp_left_hand_pose':left_hand_pose,
                       'pp_right_hand_pose':right_hand_pose,
                       'focal_length_x': focal_length_x,
                       'focal_length_y': focal_length_y,
                       'f_pix': f_pix,
                       'pitch': pitch,
                       'roll': roll}
        return output_dict