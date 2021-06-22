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

import sys
import os

import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.utils import utils

class FittingMonitor(object):
    def __init__(self,
                 maxiters=100, 
                 ftol=2e-09, 
                 gtol=1e-05,
                 model_type='smpl',
                 **kwargs):
        super(FittingMonitor, self).__init__()

        self.maxiters = maxiters
        self.ftol = ftol
        self.gtol = gtol

        self.model_type = model_type

    def __enter__(self):
        self.steps = 0
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        pass

    def run_fitting(self, optimizer, closure, params, body_model):
        ''' Helper function for running an optimization process
            Parameters
            ----------
                optimizer: torch.optim.Optimizer
                    The PyTorch optimizer object
                closure: function
                    The function used to calculate the gradients
                params: list
                    List containing the parameters that will be optimized
                body_model: nn.Module
                    The body model PyTorch module
            Returns
            -------
                loss: float
                The final loss value
        '''
        prev_loss = None
        for n in range(self.maxiters):
            old_params = [x for x in body_model.named_parameters()]
            loss = optimizer.step(closure)

            if torch.isnan(loss).sum() > 0 or torch.isinf(loss).sum() > 0 or loss is None:
                print('Inf or NaN loss value, rolling back to old params!')
                old_params = dict([(x[0], x[1].data) for x in old_params])
                body_model.reset_params(**old_params)
                break

            if n > 0 and prev_loss is not None and self.ftol > 0:
                loss_rel_change = utils.rel_change(prev_loss, loss.item())

                if loss_rel_change <= self.ftol:
                    break

            if all([torch.abs(var.grad.view(-1).max()).item() < self.gtol
                    for var in params if var.grad is not None]):
                break

            prev_loss = loss.item()

        return prev_loss

    def create_fitting_closure(self,
                               optimizer, body_model,
                               height=None, weight=None, camera=None,
                               gt_joints=None, loss=None,
                               joints_conf=None,
                               joint_weights=None,
                               return_verts=True, return_full_pose=False,
                               create_graph=False):

        def fitting_func(backward=True):
            stop = False
            # we might encounter nan values in the optimization. In this case,
            # stop iterations here. This necessart, as lbfgs for example
            # perfroms multiple optimization steps in a single optimizer step.
            for param in body_model.parameters():
                if np.any(np.isnan(param.data.cpu().numpy())) or \
                   np.any(np.isinf(param.data.cpu().numpy())):
                    print('nan in model')
                    backward = False
                    total_loss = torch.tensor(float('inf'))
            if not stop:
                if backward:
                    optimizer.zero_grad()

                body_model_output = body_model(return_verts=return_verts,
                                           body_pose=None,
                                           return_full_pose=return_full_pose)
                total_loss = loss(
                                body_model_output=body_model_output, 
                                camera=camera,
                                gt_joints=gt_joints,
                                joints_conf=joints_conf,
                                joint_weights=joint_weights,
                                gt_height=height, 
                                gt_weight=weight,
                )

                if torch.isnan(total_loss).sum() > 0 or torch.isinf(total_loss).sum() > 0:
                    print('lbfgs - Inf or NaN loss value, skip backward pass!')
                    # skip backward step in this case
                else:
                    if body_model_output.vertices is not None:
                        loss.previousverts = body_model_output.vertices.detach().clone()
                    total_loss.backward(create_graph=create_graph)

                    self.steps += 1
            #print('fitting closure loss ', total_loss)
            return total_loss

        return fitting_func
