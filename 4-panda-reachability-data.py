#! /usr/bin/env python3
import random
import os
from os.path import join, isfile, isdir
from tqdm import tqdm
import numpy as np
import time
import pandas as pd
import pybullet_planning as pp

from packing_models.assets import CATEGORIES_DIFFUSION_CSP, get_model_ids
from packing_models.bullet_utils import get_reachability_file
from demo_utils import create_robot_workstation


def sample_ik_solvable_pose(num_samples=500, num_scales=10, use_rrt=False, correction_mode=True):

    def pose_gen(z):
        r = random.uniform(0.3, 0.9)
        alpha = random.uniform(-1, 1) * np.pi
        theta = random.uniform(-1, 1) * np.pi
        pose = ((r * np.cos(alpha), r * np.sin(alpha), z), pp.quat_from_euler((0, 0, theta)))
        return pose, alpha

    c, robot = create_robot_workstation(robot_name='panda', plane_scale=2, fps=1200)
    cid = c.client_id
    robot.open_gripper_free()
    obj_kwargs = dict(static=True, pos=(2, 0), robot=robot)

    if use_rrt:
        columns = ['grasp_id', 'scale', 'x', 'y', 'z', 'theta', 'alpha', 'solved_ik', 'solved_rrt', 'solved']
    else:
        columns = ['grasp_id', 'scale', 'x', 'y', 'z', 'theta', 'alpha', 'solved']

    with c.disable_rendering():
        ## for each grasp of each asset of each category, sample 1k poses, save those that are IK solvable
        for cat in CATEGORIES_DIFFUSION_CSP:  ##
            model_ids = get_model_ids(cat)
            for model_id in model_ids:  ## ['101284', '101293', '101291']: ##
                file = get_reachability_file(f'{cat}_{model_id}', num_samples=num_samples, use_rrt=use_rrt)
                if isfile(file):
                    continue
                    data = pd.read_csv(file).values.tolist()
                    # if correction_mode:
                    #     data = [d for d in data if d[0] not in [2, 3]]
                else:
                    data = []
                start_time = time.time()
                reasons = defaultdict(int)

                for i in tqdm(range(num_scales), desc=f'{cat}_{model_id}'):
                    scale = str(i/(num_scales-1))
                    # if i == 0: scale = 'min'
                    # elif i == num_scales - 1: scale = 'max'

                    body, grasp_poses = load_packing_object(c, cat, model_id=model_id, scale=scale, **obj_kwargs)
                    scale = get_loaded_scale(cid, body)
                    z = get_pose(cid, body)[0][2]
                    for grasp_id, g in enumerate(grasp_poses):
                        # if correction_mode:
                        #     if grasp_id in [0, 1]:
                        #         continue
                        n = num_samples // len(grasp_poses) // num_scales
                        for k in range(n):
                            pose, alpha = pose_gen(z)
                            solved_ik, reason = exist_cfree_ik(c, robot, body, pose, g)

                            if reason is not None:
                                for rea in reason:
                                    reasons[rea] += 1

                            if use_rrt:
                                solved_rrt = False
                                if solved_ik:
                                    solved_rrt, _ = exist_rrt(c, robot, pick_q)
                                solved = solved_ik and solved_rrt
                                d = [grasp_id, scale, *pose[0], pose[1][2], alpha, solved_ik, solved_rrt, solved]
                            else:
                                d = [grasp_id, scale, *pose[0], pose[1][2], alpha, solved_ik]
                            data.append(d)
                    c.remove_body(body)

                df = pd.DataFrame(data, columns=columns)
                if isfile(file):
                    file = get_reachability_file(f'{cat}_{model_id}', num_samples=len(df), use_rrt=use_rrt)
                df.to_csv(file, index=False)

                reasons = dict({r: round(v / num_samples, 4) for r, v in reasons.items()})
                reasons = {k: v for k, v in sorted(reasons.items(), key=lambda item: item[1], reverse=True)}
                err(f'\t finished in {time.time() - start_time:.2f}s\t collisions:', reasons)


def visualize_all_ik_solvable_pose(**kwargs):

    def summarize_success(ddf, title):
        success = ddf.loc[ddf['solved'] == True]
        x = success.loc[:, ["x"]].abs()
        y = success.loc[:, ["y"]].abs()
        r = np.sqrt(np.array(x).reshape(-1) ** 2 + np.array(y).reshape(-1) ** 2)
        print_out = f'{title} {len(success)} / {len(ddf)} ({round(len(success) / len(ddf), 2)})'
        if len(success) > 0:
            print_out += f'\tx in [{round(x.min().item(), 3)}, {round(x.max().item(), 3)}]' + \
                  f'\ty in [{round(y.min().item(), 3)}, {round(x.max().item(), 3)}]' + \
                  f'\tr in [{round(r.min(), 3)}, {round(r.max(), 3)}]'
            scales = ddf.scale.unique()
            ## count success rate of each scale in success
            scale_total = len(ddf) // len(scales)
            scale_count = {s: len(success.loc[success['scale'] == s]) / scale_total for s in scales}
            scale_count = [(v, round(k, 3)) for k, v in sorted(scale_count.items(), key=lambda item: item[1])]
            print_out += f'\t{scale_count}'
        if 'solved_ik' in success.columns:
            solved_ik = ddf.loc[ddf['solved_ik'] == True]
            print_out += f'\t\t{len(success)} / {len(solved_ik)} ({round(len(success) / len(solved_ik), 2)}) ' \
                         f'solved IK and solved RRT'
        print(print_out)

    worked = ['Dispenser', 'Bowl', 'StaplerFlat', 'Eyeglasses', 'Pliers', 'Scissors', 'Camera', 'Bottle']
    for cat in CATEGORIES_DIFFUSION_CSP:
        if cat in worked:
            continue
        model_ids = get_model_ids(cat)
        for model_id in model_ids:
            file = get_reachability_file(f'{cat}_{model_id}', **kwargs)
            if not isfile(file):
                continue
            df = pd.read_csv(file)
            summarize_success(df, f'{cat}_{model_id}:')

            grasps = df.grasp_id.unique()
            for g in grasps:
                all_grasps = df.loc[df['grasp_id'] == g]
                summarize_success(all_grasps, f'\t\tgrasp {g}:')


if __name__ == '__main__':
    sample_ik_solvable_pose(num_samples=50000)
