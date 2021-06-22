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


import os.path as osp
import trimesh
import pyrender
import numpy as np
import PIL.Image as pil_img

def overlay_mesh(verts, faces, camera_transl, focal_length_x, focal_length_y, camera_center,
        H, W, img, camera_rotation=None, rotaround=None, contactlist=None, color=False):

    material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.0,
                alphaMode='OPAQUE',
                baseColorFactor=(1.0, 1.0, 0.9, 1.0))
    out_mesh = trimesh.Trimesh(verts, faces, process=False)
    out_mesh_col = np.array(out_mesh.visual.vertex_colors)
    if color:
        out_mesh_col[:,:3] = [112, 127, 250]
        out_mesh.visual.vertex_colors = out_mesh_col

    if contactlist is not None:
        for v1, v2 in contactlist:
            c1 = int(v1.float().mean()%255)
            c2 = int(v2.float().mean()%255)
            c3 = int((v1.float().sum()+v2.float().sum())%255)
            color = [c1, c2, c3, 255]
            out_mesh_col[v1] = color
            out_mesh_col[v2] = color
        out_mesh.visual.vertex_colors = out_mesh_col

    if camera_rotation is None:
        camera_rotation = np.eye(3)
    else:
        camera_rotation = camera_rotation[0]

    # rotate mesh and stack output images
    if rotaround is None:
        out_mesh.vertices = np.matmul(verts, camera_rotation.T) + camera_transl
    else:
        base_mesh = trimesh.Trimesh(verts, faces, process=False)
        rot_center = (base_mesh.vertices[5615] + base_mesh.vertices[5614] ) / 2
        rot = trimesh.transformations.rotation_matrix(
                np.radians(rotaround), [0, 1, 0], base_mesh.vertices[4297])
        base_mesh.apply_transform(rot)
        out_mesh.vertices = np.matmul(base_mesh.vertices, camera_rotation.T) + camera_transl

    # add mesh to scene
    mesh = pyrender.Mesh.from_trimesh(
                out_mesh,
                material=material)
    if img is not None:
        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                            ambient_light=(0.3, 0.3, 0.3))
    else:
        scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0],
                            ambient_light=(0.3, 0.3, 0.3))
    scene.add(mesh, 'mesh')

    # create and add camera
    camera_pose = np.eye(4)
    camera_pose[1, :] = - camera_pose[1, :]
    camera_pose[2, :] = - camera_pose[2, :]
    pyrencamera = pyrender.camera.IntrinsicsCamera(
                fx=focal_length_x, fy=focal_length_y,
                cx=camera_center[0], cy=camera_center[1])
    scene.add(pyrencamera, pose=camera_pose)

    # create and add light
    light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=1)
    light_pose = np.eye(4)
    for lp in [[1,1,-1], [-1,1,-1],[1,-1,-1],[-1,-1,-1]]:
        light_pose[:3, 3] = out_mesh.vertices.mean(0) + np.array(lp)
        #out_mesh.vertices.mean(0) + np.array(lp)
        scene.add(light, pose=light_pose)

    r = pyrender.OffscreenRenderer(viewport_width=W,
                                viewport_height=H,
                                point_size=1.0)
    color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
    color = color.astype(np.float32) / 255.0

    if img is not None:
        valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
        input_img = img.detach().cpu().numpy()
        output_img = (color[:, :, :-1] * valid_mask +
                        (1 - valid_mask) * input_img)
    else:
        output_img = color

    output_img = pil_img.fromarray((output_img * 255).astype(np.uint8))
    output_img= np.asarray(output_img)[:,:,:3]

    return output_img