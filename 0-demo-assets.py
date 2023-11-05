from os.path import join, dirname
import time
import json
import numpy as np
import pybullet as p
from os.path import isfile
import pybullet_planning as pp
import open3d as o3d

from pybullet_engine.client import BulletClient
from pybullet_engine.models.ur5.ur5_robot import UR5Robot

from demo_utils import create_robot_workstation, load_packing_object
from packing_models.bullet_utils import save_pointcloud_to_ply, set_joint_positions, get_joints, \
    get_aabb, get_grasp_sides, load_floating_gripper, save_grasp_db
from packing_models.assets import get_model_ids, get_pointcloud_path, load_asset_to_pdsketch, \
    CATEGORIES_BOX, CATEGORIES_TALL, get_cat_models, get_grasp_db_file, get_instance_name, \
    CATEGORIES_NON_CONVEX, CATEGORIES_SIDE_GRASP, CATEGORIES_FOLDED_CONTAINER, get_model_path, \
    CATEGORIES_BANDU, CATEGORIES_DIFFUSION_CSP


def main(save_pointcloud=False):
    # c = BulletClient(is_gui=True, width=1960, height=1440)
    c = BulletClient(is_gui=True)

    with c.disable_rendering():
        floor = c.load_urdf('assets://plane/plane.urdf', (0, 0, -0.001), body_name='plane')
        if not save_pointcloud:
            robot = UR5Robot(c)  ## faces +x axis
    target = c.w.get_debug_camera().target
    c.w.set_debug_camera(1.5, 90, -45, target=target)

    ## add assets
    all_categories = [CATEGORIES_SIDE_GRASP, CATEGORIES_FOLDED_CONTAINER]
    all_categories = [CATEGORIES_BOX, CATEGORIES_NON_CONVEX]
    all_categories = [CATEGORIES_BANDU]
    if save_pointcloud or True:
        all_categories = [CATEGORIES_DIFFUSION_CSP]

    n_max = 4

    y_spacing = 1.5
    for k, categories in enumerate(all_categories):
        gap = 0.3, 0.2
        if categories == CATEGORIES_FOLDED_CONTAINER:
            gap = 0.3, 0.6
        if categories == CATEGORIES_BANDU:
            gap = 0.1, 0.2
        n = 0
        for i, category in enumerate(categories):
            ids = get_model_ids(category)
            n_per_row = min(n_max, len(ids))
            for j, model_id in enumerate(ids):
                if save_pointcloud:
                    x, y = 0, 0
                else:
                    if len(ids) <= n_max:
                        x = (i + 1) * gap[0]
                        y = (j - n_per_row / 2) * gap[1]
                    else: ## len(ids) > n_max:
                        x = (n // n_max + 1) * gap[0]
                        y = (n % n_max - n_per_row / 2) * gap[1]
                    y += y_spacing * (k + 1 - len(all_categories) / 2)
                    n += 1

                scale = 1 if save_pointcloud else None
                draw_bb = False if save_pointcloud else True
                body = load_asset_to_pdsketch(c, category, model_id, pos=(x, y), floor=floor,
                                              draw_bb=draw_bb, scale=scale)

                if save_pointcloud:
                    pcd_path = get_pointcloud_path(category, model_id)
                    save_pointcloud_to_ply(c, body, pcd_path)
                    c.remove_body(body)

    if not save_pointcloud:
        c.wait_for_user()


def demo_save_pointcloud():
    category = 'Stapler'
    model_id = '102990'

    ## save pcd by loading into bullet, may take 20 sec
    c = BulletClient(is_gui=True)
    floor = c.load_urdf('assets://plane/plane.urdf', (0, 0, -0.001), body_name='plane')
    body = load_asset_to_pdsketch(c, category, model_id, pos=(0, 0), floor=floor,
                                  draw_bb=False, scale=1)
    print('Loading point cloud ...')
    pcd = c.world.get_pointcloud(body, zero_center=False, points_per_geom=100)
    print('Loading finished.')

    o3d.visualization.draw_geometries([o3d.geometry.PointCloud(
        points=o3d.utility.Vector3dVector(pcd),
    )])


def save_panda_gripper_pointcloud():
    c = BulletClient(is_gui=True)
    feg_file = 'assets://franka_description/robots/hand.urdf'
    body = c.load_urdf(feg_file, pos=(0, 0, 0), static=True, body_name='body_name')
    joints = get_joints(c.client_id, body)
    set_joint_positions(c.client_id, body, joints[:2], [0.04] * 2)

    pcd_file = join(dirname(__file__), 'packing_models', 'models', 'franka_description', 'hand.ply')
    save_pointcloud_to_ply(c, body, pcd_file, points_per_geom=400)

    # pcd = c.world.get_pointcloud(body, zero_center=True, points_per_geom=400)
    # o3d.visualization.draw_geometries([o3d.geometry.PointCloud(
    #     points=o3d.utility.Vector3dVector(pcd),
    # )])


def check_grasp_sides():
    c, robot = create_robot_workstation(robot_name='panda', is_gui=True)
    kwargs = dict(pos=(0.4, 0), draw_bb=False, robot=robot)

    changed = False
    db_file = get_grasp_db_file(robot)
    db = json.load(open(db_file, 'r'))

    cat_models = get_cat_models()
    # cat_models = ['Camera_102852', 'Bowl_7001', 'Bowl_7002']
    # cat_models = [s.split('_') for s in cat_models]

    for cat, model_id in cat_models:
        ## load objects and see which sides it intersects with
        with c.disable_rendering():
            body, grasps = load_packing_object(c, cat, model_id=model_id, **kwargs)
        instance_name = get_instance_name(get_model_path(cat, model_id))
        if 'grasp_sides' not in db[instance_name]:
            changed = True
            db[instance_name]['grasp_sides'] = get_grasp_sides(c, f"{cat}_{model_id}", robot, body, grasps)
        c.remove_body(body)
    if changed:
        save_grasp_db(db, db_file)

    c.wait_for_user()


if __name__ == '__main__':
    main()
    # demo_save_pointcloud()
    # save_panda_gripper_pointcloud()
    # check_grasp_sides()

