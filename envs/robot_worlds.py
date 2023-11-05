import math
import random
import pybullet_planning as pp


from mesh_utils import RAINBOW_COLORS, BLACK
from data_utils import cat_from_model_id, print_tensor, r, grasp_from_id_scale
from render_utils import export_gif

from mesh_utils import regions_to_meshes, assets_to_meshes, load_panda_meshes
from builders import get_tray_splitting_gen

from worlds import RandomSplitWorld


class TableToBoxWorld(RandomSplitWorld):
    """ 1. randomly sample a dim and pose of box
        2. sample a splitting arrangement of the box
        3. fit in objects, with orientation options
        4. save as json and pt
        5. render in trimesh and in pybullet
    """
    def __init__(self, w=None, l=None, h=None, grid_size=0.5, **kwargs):
        if w is None:
            w = random.uniform(0.3, 0.5)
            l = random.uniform(0.3, 0.6)
            self.x = random.uniform(0.2, 0.4)
            self.y = random.uniform(-0.1, 0.1)
        else:
            grid_size = min([w, l]) / 6
        t = min([w, l]) / 10
        self.fitted_theta = []
        self.gripper_poses = []
        super(TableToBoxWorld, self).__init__(w=w, l=l, h=0.1, t=t, orthographic=False,
                                              grid_size=grid_size, **kwargs)

    def sample_scene(self, min_num_objects=2, max_num_objects=6):
        """ 1. first get region boxes from `get_tray_splitting_gen`
            2. for each region, try out each asset to find a fit
            2. if no collision, optionally rotate mesh by pi, add to the scene
        """
        from assets import get_packing_assets, fit_object_assets
        h_tile = 0.01

        max_depth = math.ceil(math.log2(max_num_objects)) + 1
        gen = get_tray_splitting_gen(num_samples=40, min_num_regions=min_num_objects,
                                     max_num_regions=max_num_objects, max_depth=max_depth)
        models = get_packing_assets()
        fitted_assets = []
        fitted_regions = []
        fitted_theta = []
        while True:
            regions = next(gen(self.w, self.l))
            if regions is None:
                return None
            for region in regions:
                result = fit_object_assets(region, models, self.w, self.l, h_tile)
                if result is None:
                    continue
                fitted_assets.append(result)
                fitted_regions.append(region)
                fitted_theta.append(result[-1])
            if min_num_objects <= len(fitted_regions) <= max_num_objects:
                print(f'sample_scene.fitted {len(fitted_regions)} assets')
                break

        ## add rectangular tiles for visualization
        meshes = regions_to_meshes(fitted_regions, self.w, self.l, h_tile, max_offset=0, min_offset_perc=0)
        self.tiles.extend(meshes)

        ## add fitted assets
        meshes = assets_to_meshes(fitted_assets)
        self.tiles.extend(meshes)
        self.fitted_theta = {meshes[i].metadata['label']: fitted_theta[i] for i in range(len(meshes))}

        return True

    def generate_json(self, **kwargs):
        world = {'fitted_theta': self.fitted_theta}
        return super().generate_json(world=world, **kwargs)

    def construct_scene_from_graph_data(self, nodes, labels=None, predictions=None, verbose=False,
                                        phase='truth', draw_markers=False, only_last_grasp=False):
        """
        geom = [w*w0, l*l0, h*h0, mobility_id, scale]
        pose = [x*w0/2, y*l0/2, z*h0, yaw] + [grasp_id]
        """
        import numpy as np
        w, l = nodes[0, 1:3]
        if verbose:
            print_tensor('nodes', nodes)
            print_tensor('predictions', predictions)

        fitted_assets = []
        for i in range(1, nodes.shape[0]):

            t, bw, bl, bh, model_id, scale, x, y, z, yaw, grasp_id = nodes[i][:11]
            # hand_pose = nodes[i][11:11+7]
            # hand_pose = (hand_pose[:3].tolist(), hand_pose[3:].tolist())
            # grasp_pose = nodes[i][11+7:11+7+7]
            # grasp_pose = (grasp_pose[:3].tolist(), grasp_pose[3:].tolist())

            ## add the box shadow
            color = RAINBOW_COLORS[i - 1]
            if draw_markers:
                self.add_shape('box', height=0.01, size=(bw, bl), x=x, y=y, z=-0.01, color=color)

            ## add the object asset
            model_id = int(model_id)
            cat = cat_from_model_id(model_id)

            pose_original = ((x, y, z), pp.quat_from_euler((0, 0, yaw)))
            pose = ((x, y, z), xyzw_to_wxyz(pp.quat_from_euler((0, 0, yaw))))
            fitted_assets.append([(cat, model_id), scale, (bw, bl, bh), pose, yaw])

            """ only render the last grasp """
            if only_last_grasp and i != len(nodes) - 1:
                continue

            ## add the gripper
            grasp_original = grasp_from_id_scale(model_id, grasp_id, scale)
            TR = ((0, 0, 0.1), (0, 0, 1, 0))
            hand_pose_new = (x, y, z), _ = pp.multiply(pose_original, grasp_original, pp.invert(TR))
            if draw_markers:
                self.add_shape('box', height=0.02, size=(0.02, 0.02), x=x, y=y, z=z, color=color)

            self.gripper_poses.append(hand_pose_new)
            pick_ee_pose = (hand_pose_new[0], xyzw_to_wxyz(hand_pose_new[1]))
            self.tiles.append(load_panda_meshes(pick_ee_pose))

            ## debug
            prediction = None
            if verbose:
                print(f"{i}\t nodes: {r(nodes[i])}\t -> (predictions = {r(prediction)})\t | labels: {r(labels[i])}")

        meshes = assets_to_meshes(fitted_assets)
        self.tiles.extend(meshes)


def xyzw_to_wxyz(q):
    """ pybullet and trimesh uses different convention for quaternions """
    return q[3], q[0], q[1], q[2]
