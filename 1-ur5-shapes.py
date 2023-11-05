#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : 1-demo-scene1.py
# Author : Maomao Mao
# Email  : maojiayuan@gmail.com
# Date   : 05/11/2023
#
# This file is part of HACL-PyTorch.
# Distributed under terms of the MIT license.

import numpy as np
import pybullet as p
from os.path import join, dirname, abspath, isfile
import json

from demo_utils import demo_runner, DATASET_PATH, MODELS_DIR, unit_quat, get_initial_poses, \
    get_tiles, triangle_generator


########################################################################


def get_pose(center, kw, kl, tray_pose):
    x = tray_pose[0] - kw * center[0]
    y = tray_pose[1] + kl * center[1]
    pose_g = (x, y, tray_pose[2])
    return pose_g


def box_2d_generator(tray_dim, tray_pose):

    eval_dir = join(DATASET_PATH, 'RandomSplitWorld(20000)_train_m=ULA_t=500_m=ULA_t=500_nowraper_id=vmt3jxz3')
    eval_name = 'denoised_t=eval_n=4_i=14_k=1.json'
    eval_name = 'denoised_t=eval_n=8_i=59_k=2.json'

    poses_0 = get_initial_poses(eval_name)

    objects, kw, kl = get_tiles(join(eval_dir, eval_name), tray_dim)

    for i, o in enumerate(objects):
        dim = [kw * o['extents'][0], kl * o['extents'][1], tray_dim[2]]
        color = [n / 255 for n in o['rgba']]
        pose_0 = poses_0[i]
        pose_g = get_pose(o['center'], kw, kl, tray_pose)
        yield dim, color, pose_0, pose_g, o['label']


def box_2d_demo(video_path):
    tray_dim = (0.2, 0.2, 0.05)
    tray_pose = (0.3, 0.0, 0.025)
    gen_fn = box_2d_generator(tray_dim, tray_pose)

    def load_fn(c):
        placements = []
        for dim, color, pose_i, pose_g, name in gen_fn:
            c.load_urdf_template(
                'assets://box/box-template.urdf',
                {'DIM': dim, 'LATERAL_FRICTION': 1.0, 'MASS': 0.2, 'COLOR': color},
                pose_i, body_name=name, group='rigid', static=False
            )
            placements.append({
                'name': name, 'place_pose': (pose_g, unit_quat),
            })
        return placements

    demo_runner(load_fn, video_path, tray_dim=tray_dim, tray_pose=tray_pose)


##########################################################################################


def triangle_2d_demo(video_path):

    tray_dim = (0.2, 0.2, 0.05)
    tray_pose = (0.4, 0.0, 0.025)
    gen_fn = triangle_generator(tray_dim, tray_pose)

    def load_fn(c):
        placements = []
        for mesh_file, scale, color, pose_i, pose_g, name in gen_fn:
            c.load_urdf_template(
                abspath(join(MODELS_DIR, 'obj-template.urdf')),
                {'MESH_FILE': mesh_file, 'LATERAL_FRICTION': 1.0, 'MASS': 0.2, 'COLOR': color},
                pose_i, body_name=name, group='rigid', static=False, scale=scale
            )
            placements.append({
                'name': name, 'place_pose': (pose_g, unit_quat),
            })
        return placements

    gap = 0.01
    tray_dim = (tray_dim[0] + gap, tray_dim[1] + gap, tray_dim[2])
    demo_runner(load_fn, video_path, tray_dim=tray_dim, tray_pose=tray_pose, render=True)


##########################################################################################


def load_urdf_demo(video_path):

    tray_dim = (0.2, 0.2, 0.05)
    tray_pose = (0.3, 0.0, 0.025)

    def load_fn(c):
        body = c.load_urdf(join(MODELS_DIR, 'test.urdf'), pos=tray_pose, body_name='test', scale=1)
        return ['test'], [tray_pose]

    demo_runner(load_fn, video_path, tray_dim=tray_dim, tray_pose=tray_pose)


if __name__ == '__main__':
    # box_2d_demo('1-demo-scene1.mp4')
    triangle_2d_demo('1-demo-scene-triangle1.mp4')
    # load_urdf_demo('1-demo-scene-urdf1.mp4')

