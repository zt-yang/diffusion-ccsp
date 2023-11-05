#! /usr/bin/env python3
import copy
import sys
import numpy as np
import math
import random
import pybullet as p
import matplotlib.pyplot as plt
from pprint import pprint

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

from test_worlds import test_robot_world
from config import RENDER_PATH, DATASET_PATH
from gym_utils import get_packing_replay_dirs
from data_utils import compute_qualitative_constraints
from demo_utils import demo_runner, load_packing_object, get_rainbow_colors, get_initial_pose_alphas, \
    exist_cfree_ik, get_panda_ready, pack_given_solution_json, robot_data_config, g_z_gap, \
    check_pairwise_collisions, build_and_test_data, build_and_test_test_data, create_tamp_test_suites, \
    run_rejection_sampling_baseline, sample_pose_in_tray, rejection_sample_given_solution_json
from packing_models.bullet_utils import get_grasp_poses, remove_handles, get_loaded_scale, nice, \
    draw_goal_pose, get_pose, set_pose, take_top_down_image, merge_images, get_aabb, draw_aabb, \
    parse_datetime, get_datetime, sample_reachable_pose_nearest_neighbor, aabb_from_extent_center
from packing_models.assets import CATEGORIES_SIDE_GRASP, CATEGORIES_DIFFUSION_CSP, get_model_ids, \
    get_grasp_id, get_grasps, get_grasp_side_by_id


##########################################################################################


def get_new_xy(xy, dxy, rot):
    xy = xy @ np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
    return xy + dxy


def shuffle_tiles(tiles, dxy, rot, strategy='squared'):
    if strategy == 'random':
        random.shuffle(tiles)
    elif strategy == 'squared':
        distances = [np.linalg.norm(get_new_xy(tile['centroid'][:2], dxy, rot)) for tile in tiles]
        tiles = [t for _, t in sorted(zip(distances, tiles), reverse=True)]
    return tiles


def pack_given_world_json(output_dir=None, num_trials=20, render=False, debug=False, draw_pose=False,
                          robot_qualitative=True):

    def sample_tray_pose():
        dx = random.uniform(0.5, 0.7)
        dy = random.uniform(-0.2, 0.2)
        dxy = np.array([dx, dy])

        offset = np.array([dx, dy, tray_dim[2] / 2 + 0.025])
        tray_pose = tuple((np.array(tray['centroid']) + offset).tolist())
        return dxy, tray_pose

    def is_side_grasp(cat):
        return cat in CATEGORIES_SIDE_GRASP

    rainbow_colors = get_rainbow_colors()
    if output_dir is None:
        output_dir = join(DATASET_PATH, 'TableToBoxWorld')
    json_file = join(output_dir, 'world.json')
    objects = json.load(open(json_file, 'r'))['objects']
    tiles = [v for v in objects.values() if v['label'].startswith('tile') and '[' in v['label']]
    alphas = get_initial_pose_alphas(len(tiles))

    ## expand the container by 0.05 along each dimension
    tray = list(objects.values())[0]
    tray_dim = tuple((np.array(tray['extents']) + np.array([0.05, 0.05, 0])).tolist())
    dxy, tray_pose = sample_tray_pose()

    def load_fn(c, robot, dxy=dxy, tray_dim=tray_dim, tray_pose=tray_pose, tiles=tiles):
        cid = c.client_id
        get_panda_ready(c, robot)
        kwargs = dict(robot=robot, pos=(4, 0))

        ## find a valid dxy that enables the robot to reach all tiles
        found = False
        for k in range(num_trials+len(tiles)**2):
            placements = []
            if robot_qualitative:
                rrot = rot = 0

            ## optionally change the orientation of the tray and shuffle the order of tiles
            else:
                rrot = random.choice(range(4))
                rot = rrot * np.pi / 2
                tiles = shuffle_tiles(tiles, dxy, rot)

            set_pose(cid, c.w.get_body_index('container'), (tray_pose, pp.quat_from_euler((0, 0, rot))))

            names = []
            for i, tile in enumerate(tiles):
                name = tile['label']
                name = name[name.index('[')+1:name.index(']')]
                cat, model_id = name.split('_')

                ## prevent same names
                if name in names:
                    name = f'{name}_{i}'
                names.append(name)

                body, grasp_poses = load_packing_object(c, cat, model_id=model_id, name=name, **kwargs)
                scale = get_loaded_scale(cid, body)

                ## at least one grasp is valid
                if grasp_poses is None:
                    continue

                ## optionally, rotate each object by pi
                theta = tile['theta'] if random.random() < 0.5 else tile['theta'] + np.pi

                ## optionally, rotate the whole tray by rot
                xy = list(get_new_xy(tile['centroid'][:2], dxy, rot))
                z = get_pose(cid, body)[0][2] + g_z_gap
                theta += rot
                pose_g = (xy + [z], pp.quat_from_euler((0, 0, theta)))

                ## later, let it sample among valid grasps
                valid_grasps = []
                hand_poses = []
                with c.disable_rendering(not render):
                    for j, g in enumerate(grasp_poses):
                        solved, reason = exist_cfree_ik(c, robot, body, pose_g, g, debug=debug)
                        if solved:
                            valid_grasps.append(j)
                        hand_poses.append(solved)
                        # else:
                        #     err(f'trial {k}, body {name}, invalid grasp {j} because of {reason}')
                if len(valid_grasps) == 0:
                    c.remove_body(body)
                    continue

                extent = pp.get_aabb_extent(get_aabb(cid, body)).tolist()
                placements.append({
                    'name': name, 'body': body, 'place_pose': pose_g, 'grasp_pose': grasp_poses,
                    'valid_grasps': valid_grasps, 'z': z, 'scale': scale, 'extent': extent,
                    'is_side_grasp': is_side_grasp(cat), 'hand_pose': hand_poses
                })

            if len(placements) == len(tiles) and not check_pairwise_collisions(c, placements):
                if draw_pose:
                    for i, pc in enumerate(placements):
                        draw_goal_pose(cid, pc['body'], pc['place_pose'], color=rainbow_colors[i])
                found = True
                if rrot in [1, 3]:
                    tray_dim = tray_dim[1], tray_dim[0], tray_dim[2]
                data = {
                    'container': {'tray_dim': tray_dim, 'tray_pose': tray_pose},
                    'stats': {'num_side_grasps': sum([p['is_side_grasp'] for p in placements])}
                }
                break

            ## resample dxy
            dxy, tray_pose = sample_tray_pose()
            tray_body = c.w.get_body_index('container')
            # with c.disable_rendering(False):
            #     c.wait_for_user()
            for p in placements:
                c.remove_body(p['body'])
            c.remove_body(tray_body)
            with c.disable_rendering(not render):
                c.load_urdf_template(
                    'assets://container/container-template.urdf',
                    {'DIM': tray_dim, 'HALF': tuple([d / 2 for d in tray_dim])},
                    tray_pose, rgba=(0.5, 1.0, 0.5, 1.0), body_name='container', static=True
                )

        if not found:
            err('failed to find a valid dxy')
            return None

        # after = take_top_down_image(cid, robot, tray_pose)
        take_top_down_image(cid, robot, tray_pose, png_name=join(output_dir, 'problem.png'), goal_pose_only=True)

        ## set all objects to a place that's reachable and has a valid grasp
        for k in range(num_trials):
            found = True
            reason = None
            r = random.uniform(0.4, 0.8)
            pose_i = np.array((-r * np.cos(alphas), -r * np.sin(alphas))).T
            pose_and_grasp = []
            for i, pc in enumerate(placements):
                body = pc['body']
                original_name = get_original_name(pc['name'])
                scale = get_loaded_scale(c.client_id, body)
                pose, grasp_id = sample_reachable_pose_nearest_neighbor(
                    original_name, scale, pose_i[i], z=pc['z'], valid_grasps=pc['valid_grasps'])
                pc['grasp_id'] = int(grasp_id)

                ## test before executing
                solved, reason = exist_cfree_ik(c, robot, body, pose, pc['grasp_pose'][grasp_id], debug=debug)
                if not solved:
                    found = False
                    reason += f' at {i}'
                    break
                pose_and_grasp.append((pose, grasp_id))

            if found:
                for i, pc in enumerate(placements):
                    original_name = get_original_name(pc['name'])

                    body = pc['body']
                    pose, grasp_id = pose_and_grasp[i]
                    pc['pick_pose'] = pose
                    pc['grasp_pose'] = pc['grasp_pose'][grasp_id]
                    pc['hand_pose'] = pc['hand_pose'][grasp_id]
                    pc['grasp_side'] = get_grasp_side_by_id(original_name, grasp_id)

                    ## R = ((0, 0, 0.1), (0, 0, 1, 0))
                    # R = pp.multiply(pp.invert(pc['hand_pose']), pp.multiply(pc['place_pose'], pc['grasp_pose']))
                    # place_pose = (np.array(pc['place_pose'][0]) - np.array((tray_pose[0], tray_pose[1], 0)).tolist(),
                    #               pc['place_pose'][1])
                    # hand_pose = pp.multiply(pc['place_pose'], pc['grasp_pose'], pp.invert(R))

                    pc.pop('z')
                    pc.pop('hand_pose')
                    set_pose(c.client_id, body, pose)
                    if draw_pose:
                        draw_goal_pose(cid, body, pose, color=rainbow_colors[i])

                # before = take_top_down_image(cid, robot, tray_pose)
                # merge_images(before, after, join(temp_dir, 'problem.png'))

                data['placements'] = placements
                return placements, data
            # err('trial', k, reason)
        return None

    return demo_runner(load_fn, tray_dim=tray_dim, tray_pose=tray_pose, render=render,
                       is_gui=render, render_fps=120, gif=True, mp4=False,
                       output_dir=output_dir, video_name='solution.mp4', **robot_data_config)


def get_original_name(name):
    if len(name) - len(name.replace('_', '')) > 1:
        return name[:name.rfind('_')]
    return name


def summarize_data(data_name='TableToBoxWorld(1000)', data_names=None,
                   correct_names=False, side_grasps_only=False):
    from robot_data_monitor import get_dir_name_all_dirs
    import seaborn as sns
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})

    data_name, dirs = get_dir_name_all_dirs(data_name, data_names)

    # move_dir = False
    # data_dir = join(DATASET_PATH, data_name, 'raw')
    # if not isdir(data_dir) or len(listdir(data_dir)) == 0:
    #     os.makedirs(data_dir, exist_ok=True)
    #     data_dir = join(DATASET_PATH, data_name)
    #     move_dir = True

    assets = defaultdict(int)
    grasps = defaultdict(int)
    stats = defaultdict(list)

    # t_last = None
    # files = [f for f in listdir(data_dir) if f != 'raw']

    # if len(files[0]) < 4:  ## folders are named by index
    #     timestamps = {f: int(f) for f in files}
    # else:  ## folders are named by timestamp
    #     timestamps = {f: parse_datetime(f.split('_i=')[0]) for f in files}
    # files = sorted(files, key=lambda x: timestamps[x])

    heights = []
    for d in dirs:
        solution_file = join(d, 'solution.json')
        if not isfile(solution_file):
            continue
        data = json.load(open(solution_file, 'r'))
        stats['num_objects'].append(len(data['placements']))
        for k, v in data['stats'].items():
            if k in ['hostname', 'timestamp']:
                continue
            # if k == 'timestamp':
            #     t = parse_datetime(v)
            #     if t_last is not None:
            #         v = t - t_last
            #         t_last = t
            #     else:
            #         t_last = t
            #         continue
            stats[k].append(v)

        changes = False
        names = []
        for i, pc in enumerate(data['placements']):
            name = pc['name']
            if name in names:
                name += f'_{i}'
                changes = True
            names.append(name)

            assets[name] += 1
            grasp_id = pc['grasp_id']
            grasps[f"{name}_{grasp_id}"] += 1
            # heights.append(pc['extent'][2])

        if correct_names and changes:
            json.dump(data, open(solution_file, 'w'), indent=2)
            print(solution_file)

        # if move_dir:
        #     shutil.move(join(data_dir, d), join(data_dir, 'raw', d))

    if correct_names:
        sys.exit()
    # print(f'h:\t [{min(heights)}, {max(heights)}]')

    sum_num = defaultdict(int)
    for k in stats['num_objects']:
        sum_num[k] += 1
    print('num objects:')
    print(sum_num)

    """ plot stats """
    plt.figure(figsize=(18, 4))
    for i, (k, v) in enumerate(stats.items()):
        ax = plt.subplot(1, 3, i+1)
        unit = 1
        bins = np.arange(math.floor(np.min(v)) - 1, math.ceil(np.max(v)) + 2)
        if len(bins) > 10:
            unit = len(bins) // 10
            bins = bins[::unit]
        _ = ax.hist(v, bins - 0.5*unit)
        ax.set_title(k, fontsize=16)
        ax.set_xticks(bins)
        print(f'{k}:\t {np.round(np.mean(v), 3)} +- {np.round(np.std(v), 3)}\t [{np.min(v)}, {np.max(v)}]')
    plt.tight_layout()
    plt.savefig(join(RENDER_PATH, f'{data_name}.png'))
    plt.close()

    """ plot asset and grasp distribution """
    all_assets = {}
    all_grasps = {}
    for cat in CATEGORIES_DIFFUSION_CSP:
        for model_id in get_model_ids(cat):
            name = f'{cat}_{model_id}'
            all_assets[name] = assets[name] if name in assets else 0
            for i in range(len(get_grasps(name))):
                grasp_name = f'{name}_{i}'
                all_grasps[grasp_name] = grasps[grasp_name] if grasp_name in grasps else 0

    plt.figure(figsize=(30, 8))
    for i, (data, attr, font) in enumerate(
            [(all_assets, 'distribution of assets', 14), (all_grasps, 'distribution of grasps', 8)]):
        ax = plt.subplot(2, 1, i+1)
        x = np.arange(len(data))
        y = list(data.values())
        ax.bar(x, y)
        ax.set_title(attr, fontsize=26)
        ax.set_xticks(x, list(data.keys()), rotation=90, fontsize=font)
        if max(y) < 5:
            plt.yticks(np.arange(max(y) + 1))

    plt.tight_layout()
    # plt.show()
    plt.savefig(join(RENDER_PATH, f'{data_name}_assets.png'))
    plt.close()


def summarize_num_side_grasps(data_names):
    import seaborn as sns
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})

    num_side_grasps = defaultdict(list)
    for i, d in data_names.items():
        dataset_dir = join(DATASET_PATH, d, 'raw')
        data_dirs = [join(dataset_dir, dd) for dd in listdir(dataset_dir) if isdir(join(dataset_dir, dd))]
        for data_dir in data_dirs:
            solution_file = join(data_dir, 'solution.json')
            data = json.load(open(solution_file, 'r'))
            num_side_grasps[i].append(data['stats']['num_side_grasps'])

    """ plot stats """
    plt.figure(figsize=(14, 4))
    for i, (k, v) in enumerate(num_side_grasps.items()):
        ax = plt.subplot(1, 5, i + 1)
        unit = 1
        bins = np.arange(math.floor(np.min(v)), math.ceil(np.max(v)) + 1)
        if len(bins) > 10:
            unit = len(bins) // 10
            bins = bins[::unit]
        _ = ax.hist(v, bins - 0.5 * unit)
        ax.set_xlabel('Number of Side Grasps', fontsize=16)
        ax.set_title(f"{k} Objects", fontsize=16)
        ax.set_xticks(bins)
    plt.suptitle(f"Number of side grasps in each problem", fontsize=20)
    plt.tight_layout()
    plt.savefig(join(RENDER_PATH, f'summarize_num_side_grasps.png'))
    plt.close()


def get_test_data_dirs(dataset_names):
    for dataset_name in dataset_names:
        dataset_dir = join(DATASET_PATH, dataset_name, 'raw')
        data_dirs = listdir(dataset_dir)
        data_dirs = sorted(data_dirs)
        for d in data_dirs:
            data_dir = join(dataset_dir, d)
            if not isdir(data_dir):
                continue
            yield data_dir


def clean_configurations(dataset_names):
    for data_dir in get_test_data_dirs(dataset_names):
        solution_json = join(data_dir, 'solution.json')
        result = pack_given_solution_json(solution_json, record_video=False, render_fps=0, check_placements=True)
        if not result:
            print(data_dir)
            # shutil.rmtree(data_dir)


def sample_tamp_skeleton(new_data):
    random.shuffle(new_data['placements'])


def collect_data(min_num_objects=5, max_num_objects=5, **kwargs):
    return build_and_test_data(test_robot_world, pack_given_world_json, world_name='TableToBoxWorld',
                               min_num_objects=min_num_objects, max_num_objects=max_num_objects, **kwargs)


def collect_test_data(**kwargs):
    return build_and_test_test_data(test_robot_world, pack_given_world_json,
                                    world_name='TableToBoxWorld', **kwargs)


#########################################################################################


def rejection_sampling_fn(solution_json, prediction_json, **kwargs):
    if not rejection_sample_given_solution_json(solution_json, prediction_json, **kwargs):
        return False
    return pack_given_solution_json(prediction_json, record_video=False, render_fps=0, check_placements=True)


def run_rejection_baseline(json_name=None):
    run_rejection_sampling_baseline(
        {i: f'TableToBoxWorld(100)_robot_box_test_{i}_object' for i in range(2, 7)},
        'TableToBoxWorld(100)_eval_m=smarter_robot_box',
        rejection_sampling_fn, input_mode='robot_box', json_name=json_name
    )


def run_batch_baseline():
    for json_name in range(2, 6):
        run_rejection_baseline(json_name=json_name)
        with open(join('logs2', 'run_solve_csp_2_log.txt'), 'a+') as f:
            f.write(f"python solve_csp_2.py -input_mode robot_box -json_name t=0_{json_name}\n")


#########################################################################################


def create_qualitative_data():
    """ load objects in robot data and add qualitative constraints """
    from worlds import RandomSplitWorld

    input_mode = 'robot_qualitative'

    ## for adding the test set
    data_dirs = []
    for num in [10]:  ## , 100
        for n in range(4, 6):
            data_dir = f'TableToBoxWorld({num})_{input_mode}_test_{n}_object'
            data_dirs.append(data_dir)

    # for adding the fine-tuning set
    data_dirs += [f'TableToBoxWorld(30)_{input_mode}_finetune']
    data_dirs += [f'TableToBoxWorld(200)_{input_mode}_finetune']

    ## for creating the sets
    for data_dir in data_dirs:
        data_dir = join(DATASET_PATH, data_dir, 'raw')
        world_jsons = [join(data_dir, f, 'world.json') for f in listdir(data_dir)]
        for i, world_json in enumerate(sorted(world_jsons)):
            qualitative_constraints_json = world_json.replace('world.json', 'qualitative_constraints.json')
            if isfile(qualitative_constraints_json):
                continue

            """ the correct extent is in solution.json """
            solution_json = world_json.replace('world.json', 'solution.json')
            solution_data = json.load(open(solution_json, 'r'))
            object_extents = [d['extent'] for d in solution_data['placements']]
            world_data = json.load(open(world_json, 'r'))
            rotations = world_data['fitted_theta']

            objects = world_data['objects']
            world_objects = {'bottom': v for v in objects.values() if v['label'] == 'bottom'}
            world_objects['bottom']['extents'] = solution_data['container']['tray_dim']

            tile_objects = {v['label']: v for v in objects.values() if '[' in v['label']}
            for i, (k, v) in enumerate(tile_objects.items()):
                tile_objects[k]['extents'] = object_extents[i]
            world_objects.update(tile_objects)

            # for j, obj in enumerate(objects.values()):
            #     if j == 0:
            #         continue
            #     vertices = np.asarray(obj['vertices'])[:, :2]
            #     xyz_min = np.min(vertices, axis=0)
            #     xyz_max = np.max(vertices, axis=0)
            #     print(nice(obj['extents']), nice(xyz_max - xyz_min))
            #     print(nice(obj['center']), nice((xyz_max + xyz_min)/2))
            #     print()

            w, l = world_objects['bottom']['extents'][:2]
            scale = min([w / 3, l / 2])
            constraints = compute_qualitative_constraints(world_objects, scale=scale, rotations=rotations)
            print(len(constraints), constraints)

            ## visualize the world
            world = RandomSplitWorld(w=w, l=l)
            world.construct_scene_from_objects({k: v for k, v in tile_objects.items() if k != 'bottom'}, list(rotations.values()))
            world.render(show=False, save=True, img_name=world_json.replace('world.json', 'world.png'))

            with open(qualitative_constraints_json, 'w') as f:
                json.dump({'qualitative_constraints': constraints}, f, indent=2)

        # return


#########################################################################################

def get_visualize_kwargs():
    return dict(record_video=True, render_fps=120, check_placements=True, save_key_poses=True, save_trajectory=True)


def visualize_trajectories():
    kwargs = get_visualize_kwargs()
    solution_json = join(DATASET_PATH, 'TableToBoxWorld(10000)_robot_box_train',
                         'raw', '230531_103823_i=380_n=5', 'solution.json')
    pack_given_solution_json(solution_json, **kwargs)

    # ## ------------ select multiple trajectories for gym visualization ----------------------
    # replay_jsons = [join(f, 'solution.json') for f in get_packing_replay_dirs(num_worlds=4)]
    # for i, solution_json in enumerate(replay_jsons):
    #     print('\n\n', '-'*20, '\n', i, solution_json)
    #     pack_given_solution_json(solution_json, **kwargs)


def generate_data():

    ## ------------ debug ----------------------
    # pack_given_world_json()
    # pack_given_solution_dir()
    # data_dir = collect_data(num_data=1, render=False, debug=False)

    ## ------------ generate data ----------------------
    input_mode = 'robot_box'
    # input_mode = 'robot_qualitative'
    collect_data(min_num_objects=3, max_num_objects=5, num_data=1, parallel=True)
    # collect_data(num_data=300, group='finetune', parallel=True, render=False, input_mode=input_mode)
    # collect_data(num_data=15, group='viz', parallel=True, render=True)
    # collect_test_data(num_data=10, min_num_objects=3, max_num_objects=6, parallel=True, input_mode=input_mode)
    # collect_test_data(num_data=15, min_num_objects=6, max_num_objects=6, parallel=True, render=True)

    ## ------------ visualize instance distribution ----------------------
    # summarize_data(data_name='TableToBoxWorld(10)_robot_box_train')
    # summarize_data(data_name='TableToBoxWorld(300)_robot_qualitative_finetune')
    # summarize_data(data_names=[f'TableToBoxWorld(100)_robot_box_test_{i}_object' for i in range(2, 7)])
    test_dir_names = [f'TableToBoxWorld(100)_robot_box_test_{i}_object' for i in range(2, 7)]
    # summarize_num_side_grasps(
    #     {i: f'TableToBoxWorld(100)_robot_box_test_{i}_object' for i in range(2, 7)})

    # clean_configurations(
    #     ['TableToBoxWorld(10000)_robot_box_train']
    #     #  + [f'TableToBoxWorld(100)_robot_box_test_{i}_object' for i in range(2, 4)]  ## 2, 8
    #     #  + ['TableToBoxWorld(12)_robot_box_test']
    # )


if __name__ == '__main__':

    ## ------------ generate data ----------------------
    generate_data()

    ## ------------ visualize trajectories ----------------------
    # visualize_trajectories()
    # create_qualitative_data()

    # ## ------------ integrated TAMP & CSP solver ----------------------
    # create_tamp_test_suites(test_dir_names, sample_tamp_skeleton)

    ## ------------ baselines ----------------------
    # run_batch_baseline()

    ## ------------ generate trajectories for gym visualization ----------------------
    # generate_gym_trajectories()
