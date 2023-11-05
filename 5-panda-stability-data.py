#! /usr/bin/env python3
import copy
import sys
import time

import numpy as np
import math
import random
import pybullet as p
import matplotlib.pyplot as plt

import os
from os import listdir
from os.path import join, dirname, abspath, isfile, isdir, basename
import json
import pandas as pd
from collections import defaultdict
import functools
import shutil
err = functools.partial(print, flush=True, file=sys.stderr)

import pybullet_planning as pp
from pybullet_engine.client import BulletClient

from robot_data_monitor import get_dir_name_all_dirs
from test_worlds import test_3d_box_split_world, test_2d_box_split_world
from config import RENDER_PATH, DATASET_PATH
from mesh_utils import DARKER_GREY
from demo_utils import get_color, get_rainbow_colors, build_and_test_data, \
    build_and_test_test_data, DEMO_DIR, VideoSaver, has_ffmpeg, check_stable, \
    stability_given_solution_json, draw_world_bb, run_simulation, load_stability_shelf, \
    check_intermediate_stable, take_stability_image, get_support_structure, \
    check_all_feasibility, check_exist_bridges, create_tamp_test_suites, \
    run_rejection_sampling_baseline, sample_pose_in_tray, rejection_sample_given_solution_json
from packing_models.bullet_utils import set_pose, get_aabb, draw_aabb, equal, \
    get_pose, take_bullet_image, get_closest_points, aabb_from_extent_center


##########################################################################################


def load_objects(c, tiles, w, scaling=2, friction=0.5):
    rainbow_colors = get_rainbow_colors()
    bodies = []
    for i, tile in enumerate(tiles):
        name = tile['label']
        bw, bl, bh = [n / scaling for n in tile['extents']]
        bx, by, bz = [n / scaling for n in tile['centroid']]
        body = c.load_urdf_template(
            'assets://box/box-template.urdf',
            {'DIM': [bl, bh, bw],
             'LATERAL_FRICTION': friction, 'MASS': 0.2, 'COLOR': rainbow_colors[i]},
            [by, bz, bx + w / 2], body_name=name, group='rigid', static=False
        )
        bodies.append(body)
    return bodies


def draw_transparent_block(c, w, l, h, y_offset=0):
    tray_dim = [w, l, h]
    tray_pose = [0, -1, w / 2]
    body = c.load_urdf_template(
        'assets://box/box-template.urdf',
        {'DIM': tray_dim, 'HALF': tuple([d / 2 for d in tray_dim])},
        tray_pose, rgba=(1, 1, 1, 0), body_name='sides', static=True
    )
    tray_pose = [0, h / 2 + y_offset, w / 2]
    tray_pose = (tray_pose, pp.quat_from_euler([0, np.pi / 2, np.pi / 2]))
    set_pose(c.client_id, body, tray_pose)
    # draw_aabb(c.client_id, get_aabb(c.client_id, body))
    return body


def load_stability_world(c, json_file, mp4=False, debug=False):

    objects = json.load(open(json_file, 'r'))['objects']
    tiles = [v for v in objects.values() if v['label'].startswith('tile')]
    tray = list(objects.values())[0]
    scaling = 2
    catch_z = -40

    ## --------- add a lower support
    thickness = 0.1
    shelf_attr = dict({'DIM': [1, 0.25, thickness], 'LATERAL_FRICTION': 1.0, 'MASS': 0.2,
                       'COLOR': get_color(DARKER_GREY)})
    lower_shelf = load_stability_shelf(c, thickness, shelf_attr)
    (x1, y1, _), (x2, y2, z1) = get_aabb(c.client_id, lower_shelf)  ## z1 = 0

    ## --------- load objects
    w, l, h = [n / scaling for n in tray['extents']]
    bodies = load_objects(c, tiles, w, scaling)
    # draw_transparent_block(c, w, l, h, y_offset=h+0.001)
    # draw_transparent_block(c, w, l, h, y_offset=-h-0.001)

    ## --------- verify final configuration is stable
    run_simulation(mp4=mp4)
    if not check_stable(c, bodies):
        if debug:
            err('FAILED\tnot stable')
        return None
    z2 = draw_world_bb(c, bodies, l, h, thickness, shelf_attr)
    take_stability_image(c.client_id, json_file.replace('.json', '.png'))

    placements = []
    for i, body in enumerate(bodies):
        pose = get_pose(c.client_id, body)
        euler = pp.euler_from_quat(pose[1])
        x, y, z = pose[0]
        w, l, h = [n / scaling for n in tiles[i]['extents']]
        placements.append({
            'extents': [l, w, h],
            'centroid': [x, z, y],
            'theta': euler[1],
        })
    supports = get_support_structure(c, bodies, debug=debug)
    if debug:
        print('\n' + '-' * 20 + '\n')
        print(supports)
        print('\n' + '-' * 20 + '\n')
        # print('\n' + '-' * 20 + '\n')
        for k in range(20):
            p.stepSimulation()
        supports2 = get_support_structure(c, bodies, debug=debug)
        print('\n' + '-' * 20 + '\n')
        print(supports2)
        print('\n' + '-' * 20 + '\n')
        if len(supports) != len(supports2):
            err('FAILED\tdifferent supports???', len(supports), len(supports2))
            return None
    # c.wait_for_user()

    if not check_exist_bridges(supports):
        if debug:
            err('FAILED\tno bridges')
        return None

    order = check_all_feasibility(c, supports, bodies, debug=debug, mp4=mp4)
    if order is None:
        return None

    ## --------- save a datadir in data/raw with json and png
    data = {
        'container': {
            'shelf_pose': ((x1 + x2) / 2, (z1 + z2) / 2, (y1 + y2) / 2),
            'shelf_extent': (x2 - x1, z2 - z1, y2 - y1),
        },
        'placements': placements,
        'supports': supports,
        'order': order,
    }
    if debug: print('\n\nsuccess')
    return True, data


def pack_given_world_json(output_dir=None, mp4=False, render=False, **kwargs):
    if output_dir is None:
        output_dir = join(DATASET_PATH, 'RandomSplitWorld', 'raw')
    json_file = join(output_dir, 'world.json')

    c = BulletClient(is_gui=render, fps=120, render_fps=120)

    video_dir = join(DEMO_DIR, 'RandomSplitWorld3D')
    if not isdir(video_dir):
        os.makedirs(video_dir, exist_ok=True)

    video_path = join(video_dir, basename(output_dir) + '.mp4')
    enable = has_ffmpeg and mp4
    with VideoSaver(video_path, c.client_id, enable=enable):
        result = load_stability_world(c, json_file, mp4=mp4, **kwargs)

    if enable:
        suffix = '_success' if result else '_failed'
        shutil.move(video_path, video_path.replace('.mp4', f'{suffix}.mp4'))

    if result:
        with open(join(output_dir, 'solution.json'), 'w') as f:
            json.dump(result[1], f, indent=2)
        output = len(result[1]['placements'])
    else:
        output = False
    c.disconnect()
    return output


domain_args = dict(world_name='RandomSplitWorld', balance_data=False, input_mode='stability_flat')


def collect_data(num_data, min_num_objects=5, max_num_objects=7, **kwargs):
    data_dir = None
    num_each_n = num_data // (max_num_objects - min_num_objects + 1)
    min_n = min_num_objects
    max_n = max_num_objects
    n_data = num_data // 10
    num_all_collected = 0
    start = time.time()
    while num_all_collected < num_data:
        dirs = build_and_test_data(test_2d_box_split_world, pack_given_world_json, num_data=num_data, break_data=n_data,
                                   min_num_objects=min_n, max_num_objects=max_n, **domain_args, **kwargs)
        dirs = [d for d in dirs if d is not None]
        if len(dirs) == 0:
            continue
        if data_dir is None:
            data_dir = dirname(dirs[0])

        all_collected = listdir(data_dir)
        num_all_collected = len(all_collected)
        num_collected = defaultdict(int)
        for d in all_collected:
            n = int(d[d.find('n=')+2:])
            num_collected[n] += 1
        if num_collected[min_n] >= num_each_n:
            min_n += 1
        num_collected = dict(sorted(num_collected.items(), key=lambda x: x[0]))
        print(f'\t collected {num_all_collected} out of {num_data}', num_collected)
    print(f'num_collected ({len(all_collected)}) in {round(time.time() - start, 2)} sec', num_collected)


def collect_test_data(**kwargs):
    return build_and_test_test_data(test_2d_box_split_world, pack_given_world_json, **domain_args, **kwargs)


def summarize_data(dir_name='RandomSplitWorld(24000)_stability_flat_train', dir_names=None, show=False):
    import seaborn as sns
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})

    dir_name, dirs = get_dir_name_all_dirs(dir_name, dir_names)
    stats = defaultdict(list)
    for d in dirs:
        solution_file = join(d, 'solution.json')
        if not isfile(solution_file):
            continue
        data = json.load(open(solution_file, 'r'))
        stats['num_objects'].append(len(data['placements']))
        stats['theta'].extend([obj['theta'] for obj in data['placements']])

        ## the number of supports for each object
        all_supports = defaultdict(int)
        for (i, j) in data['supports']:
            all_supports[i] += 1

        for k, v in all_supports.items():
            stats['num_supports'].append(v)

    plt.figure(figsize=(16, 4))
    for i, (k, v) in enumerate(stats.items()):
        ax = plt.subplot(1, 3, i+1)
        unit = 1
        if k == 'theta':
            bins = np.arange(np.min(v)/6, np.max(v)/6, np.pi/60)
            bins = np.arange(np.min(v), np.max(v), np.pi/10)
            h = ax.hist(v, bins - np.pi/120)
            ax.set_xticks([round(b, 2) for b in bins])
        else:
            bins = np.arange(math.floor(np.min(v)) - 1, math.ceil(np.max(v)) + 2)
            h = ax.hist(v, bins - 0.5*unit)
            ax.set_xticks(bins)
        if k != 'num_objects':
            ax.set_yscale('log')

        ## mark y values on bars
        rects = ax.patches
        for rect, label in zip(rects, h[0]):
            if label == 0:
                continue
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, height + 0.01, int(label),
                    ha='center', va='bottom')
        ax.set_title(k, fontsize=16)
        print(f'{k}:\t {np.round(np.mean(v), 3)} +- {np.round(np.std(v), 3)}\t [{np.min(v)}, {np.max(v)}]')
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(join(RENDER_PATH, f'{dir_name}.png'))
        plt.close()


def clean_data(dir_name='RandomSplitWorld(10000)_stability_train', dir_names=None):
    dir_name, dirs = get_dir_name_all_dirs(dir_name, dir_names)
    for d in dirs:
        solution_file = join(d, 'solution.json')
        if not isfile(solution_file):
            shutil.rmtree(d)
            print('unsolved', d)
            continue
        result = stability_given_solution_json(solution_file, png=False)
        if result is None:
            shutil.rmtree(d)
            print('removed', d)


def clean_data_unsolved(dir_name='RandomSplitWorld(10000)_stability_train', dir_names=None):
    dir_name, dirs = get_dir_name_all_dirs(dir_name, dir_names)
    for d in dirs:
        solution_file = join(d, 'solution.json')
        if not isfile(solution_file):
            shutil.rmtree(d)
            print('removed', d)


def sample_tamp_skeleton(new_data):
    """ sample a support structure given a set of objects, most may be unstable """
    random.shuffle(new_data['placements'])


def rejection_sampling_fn(solution_json, prediction_json, **kwargs):
    if not rejection_sample_given_solution_json(solution_json, prediction_json, **kwargs):
        return False
    return stability_given_solution_json(prediction_json, png=True)


def run_rejection_baseline(json_name=None):
    run_rejection_sampling_baseline(
        {i: f'RandomSplitWorld(100)_stability_flat_test_{i}_object' for i in range(4, 9)},
        'RandomSplitWorld(100)_eval_m=smarter_stability_flat',
        rejection_sampling_fn, input_mode='stability_flat', json_name=json_name
    )


def run_batch_baseline():
    for json_name in range(2, 6):
        run_rejection_baseline(json_name=json_name)
        with open(join('logs2', 'run_solve_csp_2_log.txt'), 'a+') as f:
            f.write(f"python solve_csp_2.py -input_mode stability_flat -json_name t=0_{json_name}\n")


#########################################################################################


def generate_data():
    collect_data(num_data=2, min_num_objects=4, max_num_objects=7, parallel=False, render=True)  ## 1381 sec
    # collect_data(num_data=1, parallel=False, render=False)
    # collect_data(num_data=20, min_num_objects=6, max_num_objects=8, parallel=False, render=True, debug=False)  ## 25 sec

    # collect_test_data(num_data=100, min_num_objects=8, max_num_objects=8, parallel=False, render=False)

    test_dir_names = [f'RandomSplitWorld(100)_stability_flat_test_{i}_object' for i in range(4, 9)]
    # summarize_data(dir_name='RandomSplitWorld(24000)_stability_flat_train', show=False)
    # summarize_data(dir_names=test_dir_names, show=False)

    # clean_data(dir_name='RandomSplitWorld(4000)_stability_train')
    # clean_data(dir_names=[f'RandomSplitWorld(10)_stability_test_{i}_object' for i in range(4, 9)])
    # clean_data_unsolved(dir_name='RandomSplitWorld(24000)_stability_train')


def visualize_trajectories():
    solution_json = join(DATASET_PATH, 'RandomSplitWorld(1)_stability_train', 'raw',
                         '230604_204624_i=0_n=7', 'solution.json')  ## interestingly missing one support
    solution_json = join(DATASET_PATH, 'RandomSplitWorld(20)_stability_train', 'raw',
                         '230604_204636_i=14_n=7', 'solution.json')  ## interestingly missing one support
    stability_given_solution_json(solution_json, render=True, mp4=True)


#########################################################################################


if __name__ == '__main__':
    generate_data()
    # visualize_trajectories()

    ## ------------ integrated TAMP & CSP solver ----------------------
    # create_tamp_test_suites(test_dir_names, sample_tamp_skeleton)

    ## ------------ baselines ----------------------
    # run_batch_baseline()

