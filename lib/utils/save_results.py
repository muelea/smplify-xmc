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

import trimesh
import pickle 
import numpy as np
import PIL.Image as pil_img
from PIL import ImageDraw
from lib.rendering.render_singlemesh import overlay_mesh


def save_results_mesh(vertices, faces, filename):
    mesh = trimesh.Trimesh(vertices, faces)
    mesh.export(filename)
    print(f'save results to {filename}')


def save_results_params(result, filename):
    with open(filename, 'wb') as result_file:
        pickle.dump(result, result_file, protocol=2)
    print(f'save mesh to {filename}')

def save_results_image( 
    camera_center,
    camera_transl,
    camera_rotation,
    focal_length_x,
    focal_length_y,
    input_img, 
    vertices, 
    faces, 
    keypoints,
    filename
):

    # draw keypoints
    input_img_kp = input_img.detach().cpu().numpy()
    input_img_kp = pil_img.fromarray((input_img_kp * 255).astype(np.uint8))
    if keypoints is not None:
        draw = ImageDraw.Draw(input_img_kp)
        draw_kpts = [(p[0]-2, p[1]-2, p[0]+2, p[1]+2) for p in keypoints[0, :, :].numpy()]
        #fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 40)
        for _, elipse in enumerate(draw_kpts):
            draw.ellipse(elipse, fill=(255,0,0,255), outline =(255,0,0,255))
        del draw

    # render fitted mesh from different views
    H, W, _ = input_img.shape
    overlay_fit_img = overlay_mesh(vertices, faces,
                            camera_transl, focal_length_x, focal_length_y,
                            camera_center, H, W, input_img, camera_rotation, rotaround=None)
    view1_fit = overlay_mesh(vertices, faces,
                            camera_transl.astype(np.float32), focal_length_x, focal_length_y,
                            camera_center, H, W, None, None, rotaround=-45)
    view2_fit = overlay_mesh(vertices, faces,
                            camera_transl.astype(np.float32), focal_length_x, focal_length_y,
                            camera_center, H, W, None, None, rotaround=None)
    view3_fit = overlay_mesh(vertices, faces,
                            camera_transl.astype(np.float32), focal_length_x, focal_length_y,
                            camera_center, H, W, None, None, rotaround=45)

    # stack images
    IMG = np.hstack((np.asarray(input_img_kp),
                    np.asarray(overlay_fit_img), np.asarray(view1_fit),
                    np.asarray(view2_fit), np.asarray(view3_fit)))
    IMG = pil_img.fromarray(IMG)
    IMG.thumbnail((2000,2000))

    # save image
    IMG.save(filename)
    print(f'save stacked images to {filename}')


def save_results(
    camera, 
    body_model, 
    vertices,
    keypoints,
    input_img,
    mesh_fn,
    result_fn,
    out_img_fn,
    save_mesh=True,
    save_image=True,
    save_params=True,

):

    # create result dict
    result = {'camera_' + str(key): val.detach().cpu().numpy()
        for key, val in camera.named_parameters()}
    result['camera_center'] = camera.center
    result.update({key: val.detach().cpu().numpy()
        for key, val in body_model.named_parameters()})
    result['focal_length_x'] = camera.focal_length_x
    result['focal_length_y'] = camera.focal_length_y
    result['vertices'] = vertices

    # save result dict
    if save_params:
        save_results_params(result, result_fn)

    # save mesh 
    if save_mesh:
        save_results_mesh(vertices, body_model.faces, mesh_fn)

    # visualization - render presented mesh and fit
    if save_image:
        save_results_image( 
            camera_center = result['camera_center'].squeeze(),
            camera_transl = result['camera_translation'].squeeze(),
            camera_rotation = result['camera_rotation'],
            focal_length_x = result['focal_length_x'],
            focal_length_y = result['focal_length_y'],
            vertices=result['vertices'],
            input_img=input_img, 
            faces=body_model.faces, 
            keypoints=keypoints,
            filename=out_img_fn
        )
