import json
import os
import shutil
import copy
from pprint import pprint
import pybullet as p
import pybullet_planning as pp
import numpy as np
import os
from os import listdir
from os.path import join, dirname, abspath, isfile, basename, isdir
import math
import psutil
import time
import random
import sys
import functools
from tqdm import tqdm
err = functools.partial(print, flush=True, file=sys.stderr)

from pybullet_engine.client import BulletClient
from pybullet_engine.models.ur5.ur5_robot import UR5Robot
from pybullet_engine.models.panda.panda_robot import PandaRobot
from pybullet_engine.rotation_utils import compose_transformation
from pybullet_engine.algorithms.rrt import smooth_path

from config import DATASET_PATH, RENDER_PATH
from packing_models.assets import get_model_ids, load_asset_to_pdsketch, get_model_path, \
    get_instance_name
from packing_models.bullet_utils import get_grasp_poses, set_pose, take_bullet_image, set_pose, \
    dump_json, get_datetime, draw_aabb, get_aabb, draw_goal_pose, get_pose, aabb_from_extent_center, \
    get_closest_points, parallel_processing, images_to_gif, save_image, get_bodies, equal, create_attachment
from mesh_utils import DARKER_GREY, RAINBOW_COLOR_NAMES


## process name will be 'java' if ran inside the terminal in IDE
has_ffmpeg = psutil.Process(os.getpid()).name() in ['zsh', 'bash', 'python']
MODELS_DIR = abspath(join(dirname(__file__), 'pybullet_engine', 'models', 'assets'))
DEMO_DIR = abspath(join(dirname(__file__), 'demos'))
unit_quat = (0, 0, 0, 1)


def get_rainbow_colors():
    from mesh_utils import RAINBOW_COLORS
    return [(np.array(color) / 255).tolist() for color in RAINBOW_COLORS]


def get_color(color):
    return (np.array(color) / 255).tolist()


def adjust_pose(pose, offset):
    if offset is None or pose is None:
        return pose
    if len(offset) == 2:
        offset = (offset[0], offset[1], 0)
    if len(pose) == 2:
        return (pose[0][0] + offset[0], pose[0][1] + offset[1], pose[0][2] + offset[2]), pose[1]
    else:
        return pose[0] + offset[0], pose[1] + offset[1], pose[2] + offset[2]


class Saver(object):

    def save(self):
        pass

    def restore(self):
        raise NotImplementedError()

    def __enter__(self):
        # TODO: move the saving to enter?
        self.save()

    def __exit__(self, type, value, traceback):
        self.restore()


class VideoSaver(Saver):

    def __init__(self, path, client_id, enable=True):
        """ path shouldn't include "(" or ")" """
        if not enable:
            path = None
        self.path = path
        if path is None:
            self.log_id = None
        else:
            name, ext = os.path.splitext(path)
            assert ext == '.mp4'
            self.log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, fileName=path, physicsClientId=client_id)

    def restore(self):
        if self.log_id is not None:
            p.stopStateLogging(self.log_id)
            print('Saved', self.path)

#####################################################################################


def get_initial_pose_alphas(n):
    ## poses will end up following an almost circular path around the center
    d_alpha = 1 / 6
    alpha_max = d_alpha * n / 2
    return np.linspace(-alpha_max, alpha_max, n) * np.pi


def get_initial_poses(n):
    if isinstance(n, str):
        n = int(n.split('_')[2].split('=')[1])

    if n == 4:
        poses_0 = [
            (0.5, -0.3, 0.025), (0.5, 0.3, 0.025),
            (0.3, -0.3, 0.025), (0.3, 0.3, 0.025),
        ]
    else:  ## if n == 8:
        poses_0 = [
            (0.6, -0.3, 0.025), (0.6, 0.3, 0.025),
            (0.45, -0.3, 0.025), (0.45, 0.3, 0.025),
            (0.3, -0.3, 0.025), (0.3, 0.3, 0.025),
            (0.15, -0.3, 0.025), (0.15, 0.3, 0.025),
        ][:n]
    # else:
    #     poses_0 = [(0.3, 0, 0.025)] * n
    return poses_0


def smooth_move_qpos_trajectory_saved(c, robot, qt, debug=False, **kwargs):
    # kwargs['local_smoothing'] = False
    kwargs['timeout'] = 30  ## 0.5
    kwargs['atol'] = 0.03
    kwargs['gains'] = 0.6

    qt = qt.copy()
    qt.insert(0, robot.get_qpos())
    smooth_qt = [qt[0]]
    cspace = robot.get_configuration_space()
    for qpos1, qpos2 in zip(qt[:-1], qt[1:]):
        smooth_qt.extend(cspace.gen_path(qpos1, qpos2)[1][1:])

    with c.disable_rendering(False):
        ## smooth trajectory
        # cfree_pspace = robot.get_collision_free_problem_space()
        # smooth_qt = smooth_path(cfree_pspace, smooth_qt, use_smooth_path=True)

        ## save world states
        world_states = list()
        for qpos in smooth_qt:
            robot.set_qpos(qpos)
            world_states.append(robot.world.save_world())
        robot.set_qpos(smooth_qt[0])

    if debug:
        with robot.world.save_world_v2():
            robot.replay_qpos_trajectory(smooth_qt, verbose=True)
        print('Simulation smoothed trajectory finished.')
        input('Press enter to continue...')

    # NB(Jiayuan Mao @ 2023/05/26): Debug code: Save the trajectory to a temp file.
    # import time
    # import pickle
    # time_str = time.strftime("%Y%m%d-%H%M%S")
    # with open(f'/tmp/{time_str}.pkl', 'wb') as f:
    #     pickle.dump(smooth_qt, f)
    # print('Trajectory saved to', f'/tmp/{time_str}.pkl')

    # smooth_qt = qt
    world_states = robot.move_qpos_trajectory_v2(smooth_qt, return_world_states=True, **kwargs)

    return world_states


def get_pre_grasp(grasp, ratio=1.1):
    point, quat = grasp
    return [pt * ratio for pt in point], quat


def pick_and_place(robot, args, c, nice_traj=False, gif=False, debug=False,
                   camera_pose=(0.01, -1, 1), target_pose=(0, 0, 0), save_trajectory=False):
    """ return completion status and failure source """
    images = []
    world_states = []
    image_kwargs = dict(camera_pose=camera_pose, target_pose=target_pose, tf='rot180')
    pick_pose = args['pick_pose']
    place_pose = args['place_pose']
    if isinstance(robot, UR5Robot):
        robot.reach_and_pick(pick_pose[0], pick_pose[1])
        robot.reach_and_place(place_pose[0], place_pose[1])

        # target_state = c.w.get_body_state('box1')
        # robot.reach_and_pick(target_state.position, target_state.orientation)
    else:
        grasp = args['grasp_pose']
        body = args['body']
        place_pose = adjust_pose(place_pose, (0, 0, -0.005)) ## for more stable placement
        # ckwargs = dict(nr_smooth_iterations=100)

        """ GRASPING OBJECT """
        robot.open_gripper_free()
        pick_ee_pose = pp.multiply(pick_pose, grasp)
        # err('pick_ee_pose', pick_ee_pose)
        pick_q = robot.ikfast(pick_ee_pose[0], pick_ee_pose[1], error_on_fail=False)
        if pick_q is None: return None, 'pick_q'

        if nice_traj:
            with c.disable_rendering():
                try:
                    solved, pick_path = robot.rrt_collision_free(pick_q)
                except TypeError:
                    solved = False
            if not solved: return None, 'pick_path'
            ws = smooth_move_qpos_trajectory_saved(c, robot, pick_path)
            if not ws: return None, 'pick_path_smooth'

            world_states.extend(ws)
            robot.grasp()
            world_states.append(robot.world.save_world())

            actual_pick_pose = c.w.get_body_state_by_id(body)[:2]
            actual_ee_pose = robot.get_ee_pose()
            actual_grasp = pp.multiply(pp.invert(actual_pick_pose), actual_ee_pose)

            gripper_positions = [
                c.w.get_joint_state(f'panda/panda_finger_joint{i}').position for i in [1, 2]
            ]

        else:
            trials = 3
            with c.disable_rendering():
                while robot.is_colliding(pick_q) and trials > 0:
                    pick_q = robot.ikfast(pick_ee_pose[0], pick_ee_pose[1], error_on_fail=False)
                    trials -= 1
            collisions = robot.is_colliding(pick_q, return_contacts=True)
            if len(collisions) > 0:
                if debug:
                    err('\ncollision\t', [c.w.get_body_name(collisions[0].body_a), c.w.get_body_name(collisions[0].body_b)])
                # err('\npick_pose (computed)\t', pick_pose)
                # err('pick_pose (actual)\t', get_pose(c.client_id, body))
                return None, 'cfree pick_q'
            # robot.set_qpos(pick_q)
            actual_grasp = grasp

        if gif:
            images.append(take_bullet_image(c.client_id, **image_kwargs))

        """ PLACING DOWN """
        place_ee_pose = pp.multiply(place_pose, actual_grasp)
        place_q = robot.ikfast(place_ee_pose[0], place_ee_pose[1], error_on_fail=False)
        if place_q is None: return None, 'place_q'

        # ## -- to prevent bumping into objects
        # pre_grasp = get_pre_grasp(actual_grasp)
        # pre_place_ee_pose = pp.multiply(place_pose, pre_grasp)
        # pre_place_q = robot.ikfast(pre_place_ee_pose[0], pre_place_ee_pose[1], error_on_fail=False)
        # if pre_place_q is None: return None, 'pre_place_q'

        ## -- check if the body and the tray collided
        # set_pose(c.client_id, body, place_pose)
        # contacts = get_closest_points(c.client_id, body, c.w.get_body_index('container'))
        # # robot.open_gripper_free()
        # # create_attachment(c, body, c.w.get_body_index('container'), place_pose)
        # # world_states.append(robot.world.save_world())
        # # if save_trajectory:
        # #     return True, world_states
        # set_pose(c.client_id, body, pick_pose)

        # draw_pose(c.client_id, place_ee_pose, length=0.3, width=0.03)
        # c.wait_for_user()
        # return True

        ## TODO: debug robot
        # set_pose(c.client_id, body, place_pose)
        # robot.set_qpos(place_q)
        # # c.wait_for_user()
        # set_pose(c.client_id, body, pick_pose)
        # robot.set_qpos(pick_q)

        if nice_traj:

            with c.disable_rendering():
                try:
                    solved, place_path = robot.rrt_collision_free(place_q)
                except TypeError:
                    solved = False
            if not solved:
                # print('FAILED')
                # robot.set_qpos(pick_q)
                # robot.client.wait_for_user('wait...')
                return None, 'place_path'
            ws = smooth_move_qpos_trajectory_saved(c, robot, place_path)
            if not ws: return None, 'place_path_smooth'
            print('place_path_smooth', ws)
            world_states.extend(ws)

            # c.step(c.fps * 2)
            robot.open_gripper_free()
            world_states.append(robot.world.save_world())

            """ fix the object after being placed down """
            # if 'Bowl' not in c.w.get_body_name(body):
            #     create_attachment(c, body, c.w.get_body_index('container'))
            #     world_states.append((c.w.get_body_name(body), get_pose(c.client_id, body)))

            # print('actual_place_pose', get_pose(c.client_id, body)[0])

            # with c.disable_rendering():
            #     try:
            #         solved, pre_place_path = robot.rrt_collision_free(robot.get_home_qpos())
            #     except TypeError:
            #         solved = False
            # if not solved: return None, 'home_path'
            # ws = smooth_move_qpos_trajectory_saved(c, robot, pre_place_path)
            # if not ws: return None, 'home_smooth'
            # world_states.extend(ws)

            ws = robot.move_home(save_trajectory=save_trajectory)
            world_states.extend(ws)

            actual_place_pose = c.w.get_body_state_by_id(body)[:2]
        else:
            trials = 3
            with c.disable_rendering():
                while robot.is_colliding(place_q) and trials > 0:
                    place_q = robot.ikfast(place_ee_pose[0], place_ee_pose[1], error_on_fail=False)
                    trials -= 1
            collisions = robot.is_colliding(place_q, return_contacts=True)
            if len(collisions) > 0:
                if debug:
                    err('\ncollision\t', [c.w.get_body_name(collisions[0].body_a), c.w.get_body_name(collisions[0].body_b)])
                return None, 'cfree place_q'
            # robot.set_qpos(place_q)
            set_pose(c.client_id, body, place_pose)

        if gif:
            images.append(take_bullet_image(c.client_id, **image_kwargs))
    if gif:
        return True, images

    if save_trajectory:
        return True, world_states

    sim_data = {
        'pick_q': pick_q,
        'place_q': place_q,
    }
    if nice_traj:
        sim_data.update({
            'gripper_positions': gripper_positions,
            'actual_pick_pose': actual_pick_pose,
            'actual_place_pose': actual_place_pose
        })
    return True, sim_data


def create_robot_workstation(robot_name='ur5', tray_dim=None, tray_pose=None,
                             plane_pose=(1, 0, 0), plane_scale=2, render=False,
                             is_gui=True, fps=120, render_fps=None):
    c = BulletClient(is_gui=is_gui, fps=fps, render_fps=render_fps)

    with c.disable_rendering(not render):
        c.load_urdf('assets://plane/plane.urdf', (0, 0, -0.001), body_name='plane', scale=plane_scale)
        # c.load_urdf('assets://ur5/workspace.urdf', plane_pose, body_name='workspace', scale=plane_scale)
        if tray_dim is not None:  ## the tray or storage may be meshes
            c.load_urdf_template(
                'assets://container/container-template.urdf',
                {'DIM': tray_dim, 'HALF': tuple([d / 2 for d in tray_dim])},
                tray_pose, rgba=(0.5, 1.0, 0.5, 1.0), body_name='container', static=True
            )

        target = c.w.get_debug_camera().target
        if robot_name == 'ur5':
            robot = UR5Robot(c)
            c.w.set_debug_camera(1, 90, -60, target=target)
        elif robot_name == 'panda':
            robot = PandaRobot(c)
            c.w.set_debug_camera(1.4, -120, -30, target=target)
            c.w.set_debug_camera(1.3, -120, -80, target=target)
        else:
            robot = None
            c.w.set_debug_camera(1.4, 120, -30, target=target)

        # c.wait_for_user()
    return c, robot


def demo_runner(load_fn, video_name='demo.mp4', output_dir='demos', render=False, nice_traj=False,
                gif=False, mp4=True, save_key_poses=False, save_trajectory=False, solution_json=None, **kwargs):
    start_time = time.time()

    if save_trajectory:
        nice_traj = True
        gif = False

    c, robot = create_robot_workstation(render=render, **kwargs)
    with c.disable_rendering(not render):
        placements = load_fn(c, robot)
    if placements is None:
        err('Failed to compute valid placements')
        # c.wait_for_user()
        c.disconnect()
        return False
    elif isinstance(placements, tuple):
        placements, data = placements
    else:
        data = None

    # """ set all objects in placement poses to check if they float in air """
    # for args in placements:
    #     set_pose(c.client_id, args['body'], args['place_pose'])
    # c.wait_for_user()

    # NB(Jiayuan Mao @ 2023/05/26): Debug code: load a specific trajectory from file and simulate it.
    # import pickle
    # with open('/tmp/20230526-165624.pkl', 'rb') as f:
    #     a = pickle.load(f)
    # robot.set_qpos(a[0])
    # robot.move_qpos_trajectory_v2(a, atol=0.03, timeout=50, step_size=3)
    # exit()

    failed = False
    images = []
    sim_data = []
    video_path = join(DEMO_DIR, video_name.replace('.mp4', f"_n={len(placements)}.mp4"))
    enable = has_ffmpeg and mp4
    with VideoSaver(video_path, c.client_id, enable=enable): ## , robot.ignore_physics()
        for i, args in enumerate(placements):
            name = args['name']
            target_state = c.w.get_body_state(name)
            args['pick_pose'] = (target_state.position.tolist(), target_state.orientation.tolist())
            result, comments = pick_and_place(robot, args, c, nice_traj=nice_traj, gif=gif,
                                              save_trajectory=save_trajectory)
            if result is None:
                err(f'Failed to compute {comments} for {name}')
                failed = True
                break
            if isinstance(comments, list):
                images.extend(comments)
            elif isinstance(comments, dict):
                sim_data.append(comments)

    # while True:
    #     p.stepSimulation()
    #     time.sleep(1.0 / c.fps)
    c.disconnect()

    ## for generating animation
    if len(images) > 0:
        if save_trajectory:
            import pickle
            pkl_file = join(output_dir, 'world_states.pkl')
            if solution_json is not None:
                pkl_file = solution_json.replace('.json', '.pkl')
            with open(pkl_file, 'wb') as f:
                pickle.dump(images, f)
            # print(len(images))
        elif gif:
            gif_file = join(output_dir, video_name.replace('.mp4', '.gif'))
            if solution_json is not None:
                gif_file = solution_json.replace('.json', '.gif')
            images_to_gif(images, gif_file, repeat=15)

    ## for adding to the solution json with key poses
    if len(sim_data) > 0 and save_key_poses:
        data = json.load(open(join(output_dir, 'solution.json'), 'r'))
        for i, d in enumerate(sim_data):
            for k, v in d.items():
                if isinstance(v, np.ndarray):
                    sim_data[i][k] = v.tolist()
        pprint(sim_data)
        data['sim_data'] = sim_data
        with open(join(output_dir, 'solution.json'), 'w') as f:
            json.dump(data, f, indent=2)

    ## for generating pickle with trajectory

    if failed:
        return False

    if enable:
        shutil.move(video_path, join(output_dir, video_name))

    duration = round(time.time() - start_time, 3)
    err(f'success in {round(time.time() - start_time, 3)} sec!')

    if data is not None:
        import platform
        data['stats'].update({
            'duration': duration,
            'timestamp': get_datetime(),
            'hostname': platform.node(),
        })
        with open(join(output_dir, 'solution.json'), 'w') as f:
            json.dump(data, f, indent=2)
        # dump_json(data, join(output_dir, 'solution.json'))
        return len(placements)

    return True


##########################################################################################


def get_shape_pose(center, kw, kl, tray_pose):
    x = tray_pose[0] - kw * center[0]
    y = tray_pose[1] + kl * center[1]
    pose_g = (x, y, tray_pose[2])
    if len(center) == 4:
        theta = np.arccos(center[2])
        if center[3] > 0:
            theta = 2 * np.pi - theta
        quat = p.getQuaternionFromEuler((0, 0, -theta-np.pi/2))
        pose_g = (pose_g, quat)
    return pose_g


def get_tiles(json_file, tray_dim):
    objects = json.load(open(json_file, 'r'))["objects"]
    tray = [v for k, v in objects.items() if v["label"] == 'bottom'][0]

    # to normalize regarding wrt the tray
    kw = tray_dim[0] / tray['extents'][0]
    kl = tray_dim[1] / tray['extents'][1]

    objects = [v for k, v in objects.items() if v["label"].startswith('tile')]
    return objects, kw, kl


def triangle_generator(tray_dim, tray_pose):

    # models_dir = join(MODELS_DIR, 'TriangularRandomSplitWorld')
    # data_dir = join(DATASET_PATH, 'TriangularRandomSplitWorld[64]_(5)_diffuse_pairwise_test_6_split')
    # data_name = 'idx=0.json'
    # n = 6

    data_dir = join(DATASET_PATH, 'TriangularRandomSplitWorld[64]_(30000)_train_m=ULA_t=1000_diffuse_pairwise_image_id=j8lenp74')
    models_dir = data_dir
    data_name = 'denoised_t=eval_n=5_i=35_k=0.json'
    n = data_name

    poses_0 = get_initial_poses(n)

    objects, kw, kl = get_tiles(join(data_dir, data_name), tray_dim)
    for i, o in enumerate(objects):
        name = o['label']
        mesh_file = join(models_dir, data_name.split('.')[0], name.replace('triangle_', '') + '.obj')
        if not isfile(mesh_file):
            print('mesh file not found: {}'.format(mesh_file))
            continue
        color = [n / 255 for n in o['rgba']]
        pose_g = get_shape_pose(o['center'], kw, kl, tray_pose)
        # xyz = (-pose_g[0][0], -pose_g[0][1], i*0.2)
        # rpy = (math.pi, 0, -math.pi/2 + pose_g[1][-1])
        yield mesh_file, kw, color, poses_0[i], pose_g, name  ## , rpy, xyz


##########################################################


def load_packing_object(c, cat, model_id=None, model_idx=None, random_id=False, name=None,
                        floor=None, robot=None, faces=None, **kwargs):
    if floor is None:
        floor = c.w.get_body_index('plane')
    if model_id is None:
        if random_id:
            model_id = random.choice(get_model_ids(cat))
        else:
            if model_idx is None:
                model_idx = 0
            model_id = get_model_ids(cat)[model_idx]
    if name is None:
        name = f"{cat}_{model_id}"

    body = load_asset_to_pdsketch(c, cat, model_id, name=name, floor=floor, **kwargs)

    ## return grasp poses
    if robot is not None:
        path = get_model_path(cat, model_id)
        instance_name = get_instance_name(path)
        return body, get_grasp_poses(c, robot, body, instance_name, faces=faces)
    return body, model_id


##########################################################


# def get_hand_link_pose_at_grasp(c, robot, pose, grasp):
#     with c.disable_rendering():
#         pick_ee_pose = pp.multiply(pose, grasp)
#         pick_q = robot.ikfast(pick_ee_pose[0], pick_ee_pose[1], error_on_fail=False)
#         robot.set_qpos(pick_q)
#         link_pose = robot.p.getLinkState(robot.panda, robot.w.link_names[f'panda/panda_hand'].link_id)
#     return link_pose


def exist_ik(robot, pose, grasp):
    pick_ee_pose = pp.multiply(pose, grasp)
    pick_q = robot.ikfast(pick_ee_pose[0], pick_ee_pose[1], error_on_fail=False)
    solved_ik = pick_q is not None
    return solved_ik, pick_q


def set_gripper_pos(c, robot, qpos):
    c.w.set_batched_qpos_by_id(robot.panda, robot.gripper_joints, qpos)


def exist_cfree_ik(c, robot, body, pose, grasp, debug=True, collide_obj=True):

    def get_collision_pairs(collision):
        return list(set([(cl.a_name.replace('@link/panda/', ''),
                          cl.b_name.replace('@link/', '').replace('@body/', ''))
                         for cl in collision if 'panda' not in cl.b_name]))

    with c.disable_rendering(not debug):
        set_pose(c.client_id, body, pose)
        solved_ik, pick_q = exist_ik(robot, pose, grasp)
        if not solved_ik:
            return False, ['pick ik fail']
        colliding_before = robot.is_colliding(pick_q)
        link_pose = robot.p.getLinkState(robot.panda, robot.w.link_names[f'panda/panda_hand'].link_id)[:2]
        collision_before = [cl for cl in c.w.get_contact(robot.panda)] if colliding_before else None
        set_gripper_pos(c, robot, [robot.PANDA_GRIPPER_CLOSE] * 2)  ## close gripper
        colliding_after = robot.is_colliding(pick_q)
        collision_after = [cl for cl in c.w.get_contact(robot.panda)] if colliding_after else None
        set_gripper_pos(c, robot, [robot.PANDA_GRIPPER_OPEN] * 2)  ## open gripper

    if colliding_before:
        collided = get_collision_pairs(collision_before)
        if debug: err(collided)
        return False, collided

    if not collide_obj:
        return link_pose, None

    if colliding_after:
        collided = get_collision_pairs(collision_after)
        if collide_obj and len(collided) >= 1 and '/' in collided[0][1]:
            return link_pose, []

    return False, ['no object collision']


def exist_rrt(c, robot, pick_q):
    with c.disable_rendering():
        robot.set_qpos(robot.get_home_qpos())
        return robot.rrt_collision_free(pick_q)


def exist_ik_rrt(c, robot, body, pose, grasp):
    set_pose(c.client_id, body, pose)
    solved_ik, pick_q = exist_ik(robot, pose, grasp)
    if solved_ik:

        # if debug:
        #     robot.set_qpos(pick_q)
        # return True, None

        colliding = robot.is_colliding(pick_q)
        if colliding:
            return False, 'pick q collision'
        return True, None

        # solved_rrt, path = exist_rrt(c, robot, pick_q)
        # if not solved_rrt:
        #     return False, 'pick rrt'
    else:
        return False, 'pick ik'
    return True, None


##########################################################


def collect_data_process(builder_dn, tester_fn, output_dir, index=0, render=False, debug=False, **kwargs):
    """ index if process index for parallelization """

    for _ in range(10):  ## while True: ##
        data_dir = join(output_dir, str(get_datetime()) + f'_i={index}')
        builder_dn(render=False, world_dir=data_dir, **kwargs)
        result = tester_fn(data_dir, render=render, debug=debug)
        if not result:
            shutil.rmtree(data_dir)
        else:
            new_data_dir = data_dir + f'_n={result}'
            shutil.move(data_dir, new_data_dir)
            return new_data_dir


def process(inputs):
    ## balanced data
    if len(inputs) == 7:
        builder_dn, tester_fn, output_dir, index, min_num_objects, options, kwargs = inputs
        num_objects = index % options + min_num_objects
        min_num_objects = max_num_objects = num_objects
    else:
        builder_dn, tester_fn, output_dir, index, (min_num_objects, max_num_objects), kwargs = inputs
    return collect_data_process(builder_dn, tester_fn, output_dir, index=index,
                                min_num_objects=min_num_objects, max_num_objects=max_num_objects, **kwargs)


def build_and_test_data(builder_dn, tester_fn, world_name, num_data=0, break_data=None, group='train',
                        min_num_objects=3, max_num_objects=5, balance_data=True,
                        parallel=False, input_mode='robot_box', **kwargs):

    if num_data == 1:
        output_dir = join(DATASET_PATH, f'{world_name}({num_data})_{input_mode}_{group}')
        if isdir(output_dir):
            shutil.rmtree(output_dir)
    elif num_data == 0:
        output_dir = join(DATASET_PATH, world_name)
        num_data = 1
    else:
        output_dir = join(DATASET_PATH, f'{world_name}({num_data})_{input_mode}_{group}')
    output_dir = join(output_dir, 'raw')
    os.makedirs(output_dir, exist_ok=True)

    output_dirs = [join(output_dir, f) for f in listdir(output_dir) if isdir(join(output_dir, f))]
    removed = clean_data_dir(output_dirs)
    succeed = len(output_dirs) - removed

    num_data = num_data  ## min([succeed+30, num_data])
    if break_data:
        num_data = min([num_data, succeed+break_data])
    if balance_data:
        options = max_num_objects - min_num_objects + 1
        inputs = [(builder_dn, tester_fn, output_dir, i, min_num_objects, options, kwargs)
                  for i in range(succeed, num_data)]
    else:
        inputs = [(builder_dn, tester_fn, output_dir, i, (min_num_objects, max_num_objects), kwargs)
                  for i in range(succeed, num_data)]
    data_dirs = parallel_processing(process, inputs, parallel)

    # summarize_data(data_name=basename(output_dir))
    return data_dirs


def clean_data_dir(output_dirs):
    removed = 0
    for d in output_dirs:
        if 'n=' not in d:
            shutil.rmtree(d)
            removed += 1
    return removed


def build_and_test_test_data(builder_dn, tester_fn, min_num_objects=3, max_num_objects=5, **kwargs):
    for num_objects in range(min_num_objects, max_num_objects+1):
        build_and_test_data(builder_dn, tester_fn, group=f'test_{num_objects}_object',
                            min_num_objects=num_objects, max_num_objects=num_objects, **kwargs)


####################################################################################


robot_data_config = dict(plane_pose=(0, 0, 0), plane_scale=40, robot_name='panda', fps=120)
g_z_gap = 0 ## 0.01


def get_panda_ready(c, robot):
    robot.open_gripper_free()
    target = c.w.get_debug_camera().target
    c.w.set_debug_camera(1.4, 90, -60, target=target)


def pack_given_solution_json(solution_json=None, render=False, record_video=True, gif=True,
                             render_fps=0, check_placements=False, save_key_poses=False, save_trajectory=False):

    # gif = not record_video
    mp4 = nice_traj = is_gui = draw_pose = record_video
    if record_video:
        gif = False

    rainbow_colors = get_rainbow_colors()
    if solution_json is None:
        solution_json = join(DATASET_PATH, 'TableToBoxWorld', 'solution.json')
    if not solution_json.endswith('.json'):
        solution_json = join(solution_json, 'solution.json')
    output_dir = dirname(solution_json)
    mp4_name = basename(solution_json).replace('.json', '.mp4')

    data = json.load(open(solution_json, 'r'))

    tray_dim = data['container']['tray_dim']
    tray_pose = data['container']['tray_pose']

    def load_fn(c, robot):
        cid = c.client_id
        get_panda_ready(c, robot)
        for i, pc in enumerate(data['placements']):
            name = pc['name']
            if name.startswith('box'):
                w, l, h = pc['extent']
                body = pp.create_box(w, l, h, mass=30)
                c.w.notify_update(body, body_name=name, group='rigid')
                set_pose(cid, body, pc['pick_pose'])
            else:
                cat, model_id = name.split('_')[:2]
                body, _ = load_packing_object(c, cat, model_id=model_id, scale=pc['scale'], pos=pc['pick_pose'], name=name)

            ## some initial pose is too high
            pose = get_pose(cid, body)
            pose = (pose[0][0], pose[0][1], pose[0][2] - g_z_gap), pose[1]
            set_pose(cid, body, pose)

            pc['body'] = body

            if draw_pose:
                draw_aabb(cid, get_aabb(cid, body), color=rainbow_colors[i])
                draw_goal_pose(cid, body, pc['place_pose'], color=rainbow_colors[i])

        if check_placements and check_pairwise_collisions(c, data['placements']):
            return None

        return data['placements']

    return demo_runner(load_fn, tray_dim=tray_dim, tray_pose=tray_pose, render=render,
                       output_dir=output_dir, video_name=mp4_name, solution_json=solution_json,
                       gif=gif, mp4=mp4, nice_traj=nice_traj, is_gui=is_gui,
                       **robot_data_config, render_fps=render_fps,
                       save_key_poses=save_key_poses, save_trajectory=save_trajectory)


#################################################################################


def load_stability_shelf(c, thickness, shelf_attr):
    target = c.w.get_debug_camera().target
    c.w.set_debug_camera(2.5, 180, -20, target=target)

    lower_shelf = c.load_urdf_template(
        'assets://box/box-template.urdf', shelf_attr, [0, 0.125, -thickness / 2],
        body_name='lower_shelf', group='rigid', static=True
    )
    return lower_shelf


def run_simulation(mp4=True):
    sim_time = 600 ## if mp4 else 1000
    for k in range(sim_time):
        p.stepSimulation()
        if mp4:
            time.sleep(0.01)


def draw_vertical_tray(c, w, l, h):
    tray_dim = [w, l, h]
    tray_pose = [0, -1, w / 2]
    body = c.load_urdf_template(
        'assets://container/container-template.urdf',
        {'DIM': tray_dim, 'HALF': tuple([d / 2 for d in tray_dim])},
        tray_pose, rgba=(0.2, 0.2, 0.2, 0.4), body_name='container', static=True
    )
    tray_pose = [0, h / 2, w / 2]
    tray_pose = (tray_pose, pp.quat_from_euler([0, np.pi / 2, np.pi / 2]))
    set_pose(c.client_id, body, tray_pose)
    return body


def load_upper_shelf(c, shelf_space_z, shelf_attr, thickness=0.1, upper_dz=0.1):
    shelf_pose = [0, 0.125, shelf_space_z + upper_dz + thickness / 2]
    upper_shelf = c.load_urdf_template(
        'assets://box/box-template.urdf', shelf_attr, shelf_pose,
        body_name='upper_shelf', group='rigid', static=True
    )
    return upper_shelf


def draw_world_bb(c, bodies, l, h, thickness, shelf_attr, show_tray=False):
    cid = c.client_id
    w_new = 0
    for i, body in enumerate(bodies):
        aabb = get_aabb(cid, body)
        w_new = max(w_new, aabb[1][2])
    w_new += 0.02

    if show_tray:
        draw_vertical_tray(c, w_new, l, h)
    else:
        extent = [l, h, w_new]
        center = [0, h / 2, w_new / 2]
        # aabb = aabb_from_extent_center(extent, center)
        # draw_aabb(cid, aabb)
        load_upper_shelf(c, w_new, shelf_attr, thickness)
    return w_new


def check_pairwise_collisions(c, placements, debug=False):
    cid = c.client_id
    tray = c.w.get_body_index('container')

    for i, pc in enumerate(placements):
        set_pose(cid, pc['body'], pc['place_pose'])

    for i, pc in enumerate(placements):

        contacts = get_closest_points(cid, pc['body'], tray)
        if len(contacts) > 0:
            if debug:
                err(f'\ncheck_pairwise_collisions | colliding between tray', pc['name'])
            return True

        for j, pc2 in enumerate(placements):
            if i <= j: continue
            contacts = get_closest_points(cid, pc['body'], pc2['body'])
            if len(contacts) > 0:
                if debug:
                    err(f'\ncheck_pairwise_collisions | colliding between', pc['name'], pc2['name'])
                return True
            # box_aabb_2d = pp.aabb2d_from_aabb(get_aabb(cid, body))
            # if not pp.aabb_contains_aabb(box_aabb_2d, tray_aabb_2d):
            #     err(f'\n?? {name} out of tray')
            #     return None
            # set_pose(cid, body, pose)

    for i, pc in enumerate(placements):
        if 'pick_pose' not in pc:
            continue
        set_pose(cid, pc['body'], pc['pick_pose'])

    return False
    

def create_robot_prediction_json(features, prediction_json='prediction.json'):
    from data_utils import cat_from_model_id, grasp_from_id_scale, yaw_from_sn_cs
    features = features.detach().cpu().numpy()

    tray = features[0].tolist()
    h_tray = 0.1
    w_tray, l_tray = tray[3], tray[4]
    x_tray, y_tray = tray[6], tray[7] ## tray[13], tray[14]
    k_features = 21

    data = {
        'stats': get_datetime(),
        'container': {
            'tray_dim': [w_tray, l_tray, h_tray],
            'tray_pose': [x_tray, y_tray, h_tray / 2 + 0.006]
        },
        'placements': []
    }

    names = []
    print()
    for i, f in enumerate(features[1:]):
        f = f.tolist()
        w, l, h, w0, l0, h0, x0, y0, model_id, scale, g1, g2, g3, g4, g5, grasp_id, x, y, z, sn, cs = f[:k_features]
        yaw = yaw_from_sn_cs(sn, cs)
        w, l, h = w * w0, l * l0, h * h0
        x, y, z = x * w0 / 2 + x0, y * l0 / 2 + y0, z * h0 + g_z_gap

        pick_pose = f[k_features:k_features+7]
        pick_pose = (pick_pose[:3], pick_pose[3:])

        model_id = int(model_id)
        cat = cat_from_model_id(model_id)
        name = f'{cat}_{model_id}'
        grasp_pose = grasp_from_id_scale(model_id, grasp_id, scale)

        ## repeated names
        if name in names:
            name = f'{name}_{i}'
        names.append(name)
        print(name, yaw)

        data['placements'].append({
            'name': name,
            'scale': scale,
            'extent': [w, l, h],
            'pick_pose': pick_pose,
            'place_pose': [[x, y, z], pp.quat_from_euler([0, 0, yaw])],
            'grasp_pose': grasp_pose,
            'grasp_id': grasp_id,
        })

    json.dump(data, open(prediction_json, 'w'), indent=2)


def render_robot_world_from_graph(x, prediction_json, record_video=False, **kwargs):
    create_robot_prediction_json(x, prediction_json=prediction_json)
    return pack_given_solution_json(prediction_json, record_video=record_video,
                                    check_placements=True, **kwargs)


#################################################################################


def take_stability_image(cid, png_image):
    image_kwargs = dict(camera_pose=(0.02, -2.5, 1), target_pose=(0, 0, 0))
    image = take_bullet_image(cid, **image_kwargs)
    image = image[20:280, 200:640-200, :]
    save_image(image, png_image)


def check_exist_bridges(supports):
    all_supported = []
    for (i, j) in supports:
        if i in all_supported:
            return True
        all_supported.append(i)
    return False


def get_dissemble_order(supports):
    order = []
    while len(supports) > 0:
        for k in range(len(supports)):
            i, j = supports[k]
            if i not in [s[1] for s in supports]:
                order.append(i)
                supports = [s for s in supports if s[0] != i]
                break
    return order


def check_all_feasibility(c, supports, bodies, debug=False, mp4=True):

    ## --------- intermediate configurations should be stable too when objects taken out one by one
    order = get_dissemble_order(supports)
    if len(order) != len(bodies):
        if debug:
            err('FAILED\tnot cover all blocks')
        return None

    if not check_intermediate_stable(c, bodies, order, debug=debug, mp4=mp4):
        if debug:
            err('FAILED\tnot intermediate stable')
        return None

    return order


def get_support_structure(c, bodies, debug=False):
    supports = get_support_structure_by_contact(c, bodies, debug=debug)
    above = get_above_structure(c, bodies, debug=debug)
    new_support = [a for a in above if a not in supports]
    if debug and len(new_support) > 0:
        err('supports\t', supports)
        err('above\t', above)
        err('new_support\t', new_support)
    supports = supports + new_support

    heights = {i+1: get_aabb(c.client_id, body)[1][2] for i, body in enumerate(bodies)}
    supports = sorted(supports, key=lambda x: heights[x[0]], reverse=True)
    return supports


def get_above_structure(c, bodies, debug=True):
    """ get support using aabb overlap """
    colors = RAINBOW_COLOR_NAMES
    supports = []
    for i, body in enumerate(bodies):
        for j, body2 in enumerate(bodies):
            if i == j:
                continue
            (mx1, _, mz1), (mx2, _, mz2) = get_aabb(c.client_id, body)
            (nx1, _, nz1), (nx2, _, nz2) = get_aabb(c.client_id, body2)
            left_right_overlap = (mx1 < nx2 and mx2 > nx1) or (nx1 < mx2 and nx2 > mx1)
            if equal(mz1, nz2) and left_right_overlap:
                if debug:
                    err(colors[i], colors[j], 'above')
                supports.append([i+1, j+1])
    return supports


def get_support_structure_by_contact(c, bodies, debug=False):
    """ get support using contact normal """
    colors = ['[tray]'] + RAINBOW_COLOR_NAMES
    supports = []

    lower_shelf = c.w.get_body_index('lower_shelf')
    z_shelf = get_aabb(c.client_id, lower_shelf)[1][2]
    if debug: print()
    for i, body in enumerate(bodies):

        z_lower = get_aabb(c.client_id, body)[0][2]
        if abs(z_lower - z_shelf) < 0.01:
            if debug:
                err(colors[i], '[shelf]')
            supports.append([i+1, 0])

        for j, body2 in enumerate(bodies):
            if i == j:
                continue
            ct = get_closest_points(c.client_id, body, body2)
            if len(ct) > 0 and ct[0].contactNormalOnB[2] > 0:
                if debug:
                    err(colors[i], colors[j], ct[0].contactNormalOnB)
                supports.append([i+1, j+1])
            elif debug:
                if len(ct) > 0:
                    for cct in ct:
                        print(colors[i], colors[j], cct.contactNormalOnB)
                else:
                    print('\t', colors[i], colors[j], 'no contact')

    if debug:
        err('\n' + '\n'.join([str([colors[ss] for ss in s]) for s in supports]))
    return supports


def stability_given_solution_json(solution_json=None, render=False,
                                  debug=True, mp4=False, png=True):
    rainbow_colors = get_rainbow_colors()
    data = json.load(open(solution_json, 'r'))
    w, l, h = data['container']['shelf_extent']
    tray_pose = data['container']['shelf_pose']

    thickness = 0.1
    shelf_attr = dict({'DIM': [1, 0.25, thickness], 'LATERAL_FRICTION': 1.0, 'MASS': 0.2,
                       'COLOR': get_color(DARKER_GREY)})

    c = BulletClient(is_gui=render, fps=120, render_fps=0)
    lower_shelf = load_stability_shelf(c, thickness, shelf_attr)

    bodies = []
    for i, tile in enumerate(data['placements']):
        bw, bl, bh = tile['extents']
        bx, by, bz = tile['centroid']
        theta = tile['theta']
        b_dim = [bw, bh, bl]
        b_pose = [bx, bz, by]
        body = c.load_urdf_template(
            'assets://box/box-template.urdf',
            {'DIM': b_dim, 'LATERAL_FRICTION': 1.0, 'MASS': 0.2, 'COLOR': rainbow_colors[i]},
            b_pose, body_name=f"tile_{i}", group='rigid', static=False
        )
        set_pose(c.client_id, body, (b_pose, pp.quat_from_euler([0, theta, 0])))
        bodies.append(body)
    tmp_upper_shelf = load_upper_shelf(c, l, shelf_attr, thickness)
    animation = run_simulation(mp4=mp4)
    c.remove_body(tmp_upper_shelf)
    load_upper_shelf(c, l, shelf_attr, thickness)
    # z2 = draw_world_bb(c, bodies, l, h, thickness, shelf_attr)

    if png:
        take_stability_image(c.client_id, solution_json.replace('.json', '.png'))

    ## final configuration is stable
    if not check_stable(c, bodies):
        err('FAILED!\tnot stable')
        c.disconnect()
        return None

    ## support structure is the same as given
    supports_original = data['supports']
    supports = get_support_structure(c, bodies, debug=False)
    if set([tuple(s) for s in supports_original]) != set([tuple(s) for s in supports]):
        if debug:
            err('FAILED\tnot the same support structure')
            # err('original', supports_original)
            # err('current', supports)
        c.disconnect()
        return None

    ## top of stack should not be higher than the top of the shelf
    top_z = max([get_aabb(c.client_id, body)[1][2] for body in bodies])
    if top_z > l:
        if debug:
            err('FAILED\theight violated the upper limit')
        c.disconnect()
        return None

    ## intermediate configuration is stable
    with c.disable_rendering():
        order = check_all_feasibility(c, supports, bodies, debug=debug, mp4=mp4)
    if order is None:
        c.disconnect()
        return None

    # c.wait_for_user()
    c.disconnect()
    if debug:
        err('SUCCEED')
    return True


def check_stable(c, bodies):
    for body in bodies:
        pose = get_pose(c.client_id, body)
        if pose[0][2] < 0:
            return False
    return True


def check_intermediate_stable(c, bodies, order, debug=True, mp4=False):
    # if debug: err(f'order ({len(order)})', order)
    original_bodies = bodies.copy()
    for i in order:
        c.remove_body(original_bodies[i-1])
        run_simulation(mp4=mp4)
        bodies.remove(original_bodies[i-1])
        # if debug: err(f'\tremoving  {original_bodies[i-1]}', bodies)
        if not check_stable(c, bodies):
            # if debug: err(f'\t\tintermediate not stable')
            return False
    return True


def create_stability_prediction_json(features, world_dims, supports,
                                     prediction_json='prediction.json'):
    from data_utils import yaw_from_sn_cs
    features = features.detach().cpu().numpy()
    supports = supports.detach().cpu().numpy().T.tolist()

    h0 = 0.25
    w0, l0 = world_dims
    x0, y0 = 0, l0/2

    data = {
        'stats': get_datetime(),
        'container': {
            'shelf_extent': [w0, l0, h0],
            'shelf_pose': [x0, y0, h0 / 2]
        },
        'placements': [],
        'supports': supports,
    }

    for i, f in enumerate(features[1:]):
        f = f.tolist()
        w, l, x, y, sn, cs = f
        yaw = yaw_from_sn_cs(sn, cs)
        w, l, h = w * w0, l * l0, h0
        x, y, z = x * w0 / 2 + x0, y * l0 / 2 + y0, h0/2

        data['placements'].append({
            'extents': [w, l, h], 'centroid': [x, y, z], 'theta': yaw,
        })

    json.dump(data, open(prediction_json, 'w'), indent=2)


def render_stability_world_from_graph(x, prediction_json, world_dims, supports, **kwargs):
    create_stability_prediction_json(x, world_dims, supports, prediction_json=prediction_json)
    return stability_given_solution_json(prediction_json, **kwargs)


#######################################################################################


def create_tamp_test_suites(test_data_names, sample_tamp_skeleton_fn, **kwargs):
    for test_data_name in test_data_names:
        create_tamp_test_suite(test_data_name, sample_tamp_skeleton_fn, **kwargs)


def create_tamp_test_suite(test_data_name, sample_tamp_skeleton_fn, **kwargs):
    data_dir = join(DATASET_PATH, test_data_name, 'raw')
    data_dirs = listdir(data_dir)
    random.shuffle(data_dirs)
    for i, d in enumerate(data_dirs[:10]):
        solution_json = join(data_dir, d, 'solution.json')
        create_tamp_test_data(solution_json, sample_tamp_skeleton_fn, index=i, **kwargs)


def create_tamp_test_data(solution_json, sample_tamp_skeleton_fn, index=0, num_per_batch=50):
    """ given a solution.json, create variations in placement sequence """
    n = int(basename(dirname(solution_json)).split('=')[-1])
    data_dir = dirname(dirname(dirname(solution_json))) + f'_all_n={n}_i={index}'
    num = data_dir.split('(')[1].split(')')[0]
    data_dir = data_dir.replace(f'({num})', f'({num_per_batch})')
    if isdir(data_dir):
        shutil.rmtree(data_dir)
    data_dir = join(data_dir, 'raw')
    os.makedirs(data_dir, exist_ok=True)
    data = json.load(open(solution_json, 'r'))
    for i in range(num_per_batch):
        new_data = copy.deepcopy(data)
        new_data['original'] = solution_json
        new_data_dir = join(data_dir, f'{i}')
        os.makedirs(new_data_dir, exist_ok=True)
        sample_tamp_skeleton_fn(new_data)
        json.dump(new_data, open(join(new_data_dir, 'solution.json'), 'w'), indent=2)


def run_rejection_sampling_baseline(dataset_names, output_name, rejection_sampling_fn,
                                    input_mode='robot_box', json_name=None):
    prefix = 'denoised_t=0'
    if json_name is not None:
        prefix += f'_{json_name}'

    output_dir = join(RENDER_PATH, output_name)
    if not isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    log = {}
    for n, dataset_name in tqdm(dataset_names.items()):
        success_list = []
        succeeded_graph_indices = []
        success_rounds = {}
        if input_mode in ['robot_box', 'stability_flat']:
            dataset_dir = join(DATASET_PATH, dataset_name, 'raw')
            data_dirs = listdir(dataset_dir)
        elif input_mode in ['diffuse_pairwise', 'qualitative']:
            dataset_dir = join(RENDER_PATH, dataset_name)
            data_dirs = [f for f in listdir(dataset_dir) if '.json' in f]
        data_dirs = sorted(data_dirs)
        for i, d in tqdm(enumerate(data_dirs), desc=f'\t[{n}] {dataset_name}'):
            if input_mode in ['robot_box', 'stability_flat']:
                data_dir = join(dataset_dir, d)
                solution_json = join(data_dir, 'solution.json')
            elif input_mode in ['diffuse_pairwise', 'qualitative']:
                solution_json = join(dataset_dir, d)
            for k in tqdm(range(10), desc=f'\t\t[{i}] {solution_json}'):
                prediction_json = f"{prefix}_n={n}_i={i}_k={k}.json"
                prediction_json = join(output_dir, prediction_json)
                success = rejection_sampling_fn(solution_json, prediction_json, input_mode=input_mode)

                if success:
                    success_list.append((i, k))
                    succeeded_graph_indices.append(i)
                    success_rounds[i] = k
                    break
        top1 = round(len(succeeded_graph_indices) / len(data_dirs), 3)
        log[n] = {
            'problems': data_dirs, 'percentage_solved': top1,
            'success_list': success_list, 'success_rounds': success_rounds,
            'succeeded_graph_indices': succeeded_graph_indices
        }
        with open(join(output_dir, f'{prefix}.json'), 'w') as f:
            json.dump(log, f)


def rejection_sample_given_solution_json(solution_json, prediction_json, num_samples=50, input_mode='robot_box'):
    from worlds import RandomSplitWorld

    data = json.load(open(solution_json, 'r'))
    if input_mode == 'robot_box':
        w, l, _ = tray_dim = data['container']['tray_dim']
        tray_pose = data['container']['tray_pose']
        placements = data['placements']
        tray_world = RandomSplitWorld(w=w, l=l, grid_size=0.5)
    elif input_mode == 'stability_flat':
        w, l, _ = tray_dim = data['container']['shelf_extent']
        # w, l, _ = tray_dim = (3, 2, 0.1)
        tray_pose = (0, 0, 0.125)
        placements = data['placements']
        tray_world = RandomSplitWorld(w=w, l=l, h=0.2, orthographic=False, grid_size=0.5)
    elif input_mode in ['diffuse_pairwise', 'qualitative']:
        w, l, _ = tray_dim = (3, 3, 0.1)
        if input_mode == 'qualitative':
            w, l, _ = tray_dim = (3, 2, 0.1)
        tray_pose = (0, 0, 0)
        placements = [v for v in data['objects'].values() if 'tile_' in v['label']]
        names = [k for k, v in data['objects'].items() if 'tile_' in v['label']]
        tray_world = RandomSplitWorld(w=w, l=l, h=0.2, orthographic=False, grid_size=0.5)

    tray_aabb = aabb_from_extent_center(tray_dim, tray_pose)

    objects = tray_world.generate_json()['objects']
    rotations_so_far = {}
    for i, pc in enumerate(placements):
        n_samples = 0
        name = f'tile_{i}'

        extents = pc['extents'] if input_mode == 'qualitative' else None
        x, y, z, yaw = sample_pose_in_tray(tray_aabb, extents=extents, input_mode=input_mode)

        if input_mode == 'robot_box':
            obj = {name: {'label': pc['name'], 'centroid': (x, y, z), 'extents': pc['extent'], 'shape': 'box'}}
        elif input_mode == 'stability_flat':
            ww, ll, hh = pc['extents']
            if abs(yaw) - np.pi / 2 < 0.01:
                ll, ww, hh = pc['extents']
            obj = {name: {'label': name, 'centroid': (x, y, z), 'extents': [ww, ll, hh], 'shape': 'box'}}
        elif input_mode == 'diffuse_pairwise':
            vertices = pc['vertices_centered']
            # print('vertices', i, vertices)
            new_vertices = [v[:2] for v in vertices]
            new_vertices = np.array(new_vertices)
            cs = np.cos(yaw)
            sn = np.sin(yaw)
            R = np.array([[cs, -np.abs(sn)], [np.abs(sn), cs]])
            new_vertices = (R @ new_vertices.T).T ## + np.array([x, y])
            vertices = [[v[0], v[1], vertices[i][2] * 100] for i, v in enumerate(new_vertices)]
            # R = tf.rotation_matrix(rotations_so_far[pc['label']], (0, 0, 1))
            obj = {name: {'label': name, 'centroid': (x, y, z), 'shape': 'arbitrary_triangle',
                          'vertices_centered': vertices, 'faces': pc['faces'], 'color': pc['color']}}
        elif input_mode == 'qualitative':
            obj = {name: {'label': name, 'centroid': (x, y, z), 'extents': pc['extents'], 'shape': 'box'}}

        combo = copy.deepcopy(objects)
        combo.update(obj)
        rotations = copy.deepcopy(rotations_so_far)
        rotations.update({obj[name]['label']: yaw})
        if input_mode in ['diffuse_pairwise', 'stability_flat']:
            rotations = {}
        while not ensure_cfree_in_world(combo, rotations, debug=False):
            x, y, z, yaw = sample_pose_in_tray(tray_aabb, extents=extents, input_mode=input_mode)
            if input_mode == 'robot_box':
                obj = {name: {'label': pc['name'], 'centroid': (x, y, z), 'extents': pc['extent'], 'shape': 'box'}}
            elif input_mode == 'stability_flat':
                ww, ll, hh = pc['extents']
                if abs(yaw) - np.pi / 2 < 0.01:
                    ll, ww, hh = pc['extents']
                obj = {name: {'label': name, 'centroid': (x, y, z), 'extents': [ww, ll, hh], 'shape': 'box'}}
            elif input_mode == 'diffuse_pairwise':
                vertices = pc['vertices_centered']
                new_vertices = [v[:2] for v in vertices]
                new_vertices = np.array(new_vertices)
                cs = np.cos(yaw)
                sn = np.sin(yaw)
                R = np.array([[cs, -np.abs(sn)], [np.abs(sn), cs]])
                new_vertices = (R @ new_vertices.T).T ## + np.array([x, y])
                vertices = [[v[0], v[1], vertices[i][2]*100] for i, v in enumerate(new_vertices)]

                obj = {name: {'label': name, 'centroid': (x, y, z), 'shape': 'arbitrary_triangle',
                              'vertices_centered': vertices, 'faces': pc['faces'], 'color': pc['color']}}
            elif input_mode == 'qualitative':
                obj = {name: {'label': name, 'centroid': (x, y, z), 'extents': pc['extents'], 'shape': 'box'}}

            combo = copy.deepcopy(objects)
            combo.update(obj)
            rotations = copy.deepcopy(rotations_so_far)
            rotations.update({obj[name]['label']: yaw})
            if input_mode in ['diffuse_pairwise', 'stability_flat']:
                rotations = {}
            n_samples += 1
            if n_samples > num_samples:
                # print(f"Failed to sample a pose for {name}")
                return False
        rotations_so_far[obj[name]['label']] = yaw
        objects.update(obj)
        if input_mode == 'robot_box':
            pc['place_pose'] = ((x, y, z), pp.quat_from_euler((0, 0, yaw)))
        elif input_mode == 'stability_flat':
            pc['centroid'] = [x, y, z]
            pc['extents'] = obj[name]['extents']
            pc['theta'] = yaw
            pc['label'] = name
        elif input_mode == 'diffuse_pairwise':
            data['objects'][names[i]]['centroid'] = [x, y, z]
            data['objects'][names[i]]['vertices_centered'] = obj[name]['vertices_centered']
        elif input_mode == 'qualitative':
            data['objects'][names[i]]['centroid'] = data['objects'][names[i]]['center'] = [x, y, z]

    json.dump(data, open(prediction_json, 'w'), indent=2)

    if input_mode == 'diffuse_pairwise' and False:
        import transformations as tf
        for i, pc in enumerate(placements):
            x, y, z = pc['centroid']
            # R = tf.rotation_matrix(rotations_so_far[pc['label']], (0, 0, 1))
            tray_world.add_shape('triangle', size=pc['vertices_centered'], x=x, y=y, z=z, color=pc['color'])
        tray_world.render(img_name=prediction_json.replace('.json', '.png'), show=False)

    elif input_mode == 'qualitative':
        import transformations as tf

        for i, pc in enumerate(placements):
            x, y, z = pc['centroid']
            R = tf.rotation_matrix(rotations_so_far[pc['label']], (0, 0, 1))
            tray_world.add_shape('box', size=pc['extents'], x=x, y=y, z=z, color=pc['color'], R=R)
        tray_world.render(img_name=prediction_json.replace('.json', '.png'), show=False)
        tray_world_constraints = set([tuple(ct) for ct in tray_world.generate_json(input_mode='qualitative')['constraints']])
        given_constraints = set([tuple(ct) for ct in data['constraints']])
        return tray_world_constraints == given_constraints

    elif input_mode == 'stability_flat':
        import transformations as tf

        for i, pc in enumerate(placements):
            x, y, z = pc['centroid']
            tray_world.add_shape('box', size=pc['extents'], x=x, y=y, z=z)
        tray_world.render(img_name=prediction_json.replace('.json', '.png'), show=False)

    return True


def ensure_cfree_in_world(objects, rotations, debug=True):
    from collisions import check_collisions_in_scene
    from data_monitor import plot_boxes
    cfree = [('north', 'east'), ('south', 'east'), ('north', 'west'), ('south', 'west')]
    collisions = check_collisions_in_scene(objects, rotations)
    collisions = [c for c in collisions if c not in cfree and c[0] != 'bottom' and c[1] != 'bottom']
    result = len(collisions) == 0
    if debug and not result:
        num_tiles = len([k for k, v in objects.items() if 'tile' in v['label']])
        keys = ['label', 'extents', 'centroid']
        rectangles = [[objects['geometry_0'][k] for k in keys]]
        rectangles += [[objects[f'tile_{n}'][k] for k in keys] for n in range(num_tiles)]
        plot_boxes(rectangles, png_name=None, title='collisions')
    return result


def sample_pose_in_tray(tray_aabb, extents=None, input_mode='robot_box'):
    x = np.random.uniform(tray_aabb[0][0], tray_aabb[1][0])
    y = np.random.uniform(tray_aabb[0][1], tray_aabb[1][1])
    z = (tray_aabb[0][2] + tray_aabb[1][2]) / 2
    if extents is not None:
        w, l = [s/2 for s in extents[:2]]
        if tray_aabb[0][0] + w < tray_aabb[1][0] - w:
            x = np.random.uniform(tray_aabb[0][0] + w, tray_aabb[1][0] - w)
        if tray_aabb[0][1] + l < tray_aabb[1][1] - l:
            y = np.random.uniform(tray_aabb[0][1] + l, tray_aabb[1][1] - l)
    yaw = 0
    if input_mode == 'diffuse_pairwise':
        yaw = np.random.uniform(0, 2*np.pi)
    elif input_mode in ['robot_box', 'stability_flat', 'qualitative'] and np.random.uniform() < 0.5:
        yaw = np.pi / 2
    return x, y, z, yaw
