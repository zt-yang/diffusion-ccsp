#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : 1-demo-scene1.py
# Author : Maomao Mao
# Email  : maojiayuan@gmail.com
# Date   : 05/11/2023
#
# This file is part of HACL-PyTorch.
# Distributed under terms of the MIT license.
import copy
import numpy as np

import pybullet_planning as pp

from demo_utils import demo_runner, get_rainbow_colors, err, load_packing_object
from packing_models.assets import get_model_ids, bottom_to_center
from packing_models.bullet_utils import get_loaded_scale, nice, draw_goal_pose


##########################################################################################

"""
need to add the following contact info to all urdf otherwise grasp generation won't work

        <contact>
          <lateral_friction value="1"/>
          <rolling_friction value="0.0001"/>
          <inertia_scaling value="3.0"/>
        </contact>
        <inertial>
          <origin rpy="0 0 0" xyz="0 0 0"/>
           <mass value="0.2"/>
           <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
        </inertial>
"""


def load_container(c, cat='Safe', pos=(0.8, 0.0), debug=False, **kwargs):
    point = copy.deepcopy(pos)
    if cat == 'Box':
        pos = (pos, pp.quat_from_euler((0, 0, np.pi/2)))
    container, model_id = load_packing_object(c, cat, pos=pos, static=True, **kwargs)

    ## the placement height of the safe varies with scale, find out through debug = True
    z, x = 0.2, 0.6
    asset_params = {
        "101363": (0.127 / 0.6855046014074985, 0.6),
        "100550": (0.145 / 0.38006969308541744, 0.4),
        "100426": (0.02 / 0.27005079937819576, point[0]),
    }
    if model_id in asset_params:
        z, x = asset_params[model_id]
        z *= get_loaded_scale(c.client_id, container)
    else:
        debug = True

    ## measure the height by hand
    if debug:
        handles = pp.draw_aabb(pp.AABB((point[0]-0.05, -0.05, z), (point[0]+0.05, 0.05, z+0.1)), color=(1, 0, 0, 1))
        # remove_handles(c.client_id, handles)
        c.wait_for_user()
    return container, z, x


def pack_block_in_safe_demo(video_path='2-demo-panda-storage-safe.mp4'):
    def load_fn(c):
        container_body, z, x = load_container(c, 'Safe')

        ## test box
        placements = []
        grasp = ((0, 0, 0.01), pp.quat_from_euler((0, np.pi, 0)))
        for j in range(-1, 2):
            name = f'box_{j}'
            pose_i = (-0.4, j*0.2, 0.025)
            pose_g = (0.7, j*0.1, z+0.025)  ## (0.6, 0, 0.075)

            body = c.load_urdf_template(
                'assets://box/box-template.urdf',
                {'DIM': (0.05, 0.05, 0.05), 'LATERAL_FRICTION': 1.0, 'MASS': 0.2, 'COLOR': (1, 0, 0, 1)},
                pose_i, body_name=name, group='rigid', static=False
            )
            placements.append({
                'name': name, 'body': body, 'place_pose': pose_g, 'grasp_pose': grasp,
            })
        return placements

    demo_runner(load_fn, video_path, robot_name='panda', render=True)


def placement_rotation_demo(video_path='2-demo-panda-storage-bottle.mp4'):
    def load_fn(c, robot):
        placements = []
        kwargs = dict(robot=robot, static=False)

        cat = 'Bowl'
        pose_i = (0.4, 0.0)
        name = cat.lower()
        body, grasp_poses = load_packing_object(c, cat, name=name, pos=pose_i, **kwargs)
        grasp = grasp_poses[0]
        h = bottom_to_center(c.client_id, body) + grasp[0][2]

        num = 12
        for i in range(1, num+1):
            theta = np.deg2rad(i * 360 / num)
            cs = np.cos(theta)
            sn = np.sin(theta)

            pose_g = 0.4 * np.array(((cs, -sn), (sn, cs))).dot((1, 0))
            pose_g = (pose_g.tolist() + [h], pp.quat_from_euler((0, 0, theta)))

            placements.append({
                'name': name, 'body': body, 'place_pose': pose_g, 'grasp_pose': grasp,
            })

        return placements

    demo_runner(load_fn, video_path, robot_name='panda', render=True)


def pack_demo(container='Safe', video_path='2-demo-panda-pack-CONTAINER.mp4'):

    rainbow_colors = get_rainbow_colors()

    def load_fn(c, robot):
        x = 0.7 if container == 'Safe' else 0.6
        z = 0.005
        # container_body, z, x = load_container(c, container, pos=(x, 0.0), debug=False)

        if container == 'Box':
            target = c.w.get_debug_camera().target
            c.w.set_debug_camera(1.4, 120, -45, target=target)

        placements = []

        ## ['Dispenser', 'Bowl', 'StaplerFlat', 'Eyeglasses',
        # 'Pliers', 'Scissors', 'Camera', 'Bottle', 'BottleOpened', 'Mug']
        cats = ['Scissors']
        m, n = 0, 0

        faces = None ## [(1, 0, 0)]  ## [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0)]  ## None ##
        model_ids = get_model_ids(cats[0])[m:]
        if len(cats) > 1:
            model_ids = [None] * len(cats)
        # else:
        #     cats = cats * len(model_ids)
        kwargs = dict(robot=robot, static=faces is not None, faces=faces)

        err('model_ids', model_ids[0])

        for j, (cat, model_id) in enumerate(zip(cats, model_ids)):

            ## initial pose
            offset = (j-len(cats)/2+0.5)*0.2
            pose_i = (-0.6, offset)
            if cat in ['Mug']:
                pose_i = (pose_i, pp.quat_from_euler((0, 0, np.pi)))

            ## get grasps of object
            body, grasp_poses = load_packing_object(c, cat, model_id=model_id, pos=pose_i[:2], **kwargs)
            if grasp_poses is None: continue

            name = c.w.get_body_name(body)
            if len(grasp_poses) == 0:
                continue
            grasp = grasp_poses[n]

            ## arrange goal poses
            gap = 0.005
            z_g = z + bottom_to_center(c.client_id, body) + gap
            if container == 'Safe':
                pose_g = (x, offset, z_g)
            else:
                pose_g = (x-offset, 0, z_g)
            pose_g = (pose_g, pp.quat_from_euler((0, 0, np.pi)))
            draw_goal_pose(c.client_id, body, pose_g, color=rainbow_colors[j])
            placements.append({
                'name': name, 'body': body, 'place_pose': pose_g, 'grasp_pose': grasp,
            })

        return placements

    video_path = video_path.replace('CONTAINER', container.lower())
    demo_runner(load_fn, video_path, robot_name='panda', render=True, fps=480)


if __name__ == '__main__':
    pack_block_in_safe_demo()
    placement_rotation_demo()
    pack_demo("Safe")
    pack_demo("Box")
