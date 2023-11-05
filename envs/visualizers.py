import trimesh
import numpy as np
import os
from os import listdir
from typing import Iterable, Tuple

from config import *
from worlds import TrayWorld, ShapeSettingWorld, RandomSplitWorld
from render_utils import *


class WorldVisualizer(object):

    def __init__(self, world_class: TrayWorld.__class__,
                 resolution: Tuple[int, int] = (1600, 1200),
                 world_args: dict = {},
                 world_sampler_args: dict = {},
                 render_args: dict = {}):
        self.name = self.__class__.__name__
        self.world_class = world_class
        self.resolution = resolution
        self.orthographic = True

        self.world_args = world_args
        self.world_sampler_args = world_sampler_args
        self.render_args = render_args

    def set_camera_pose(self, scene: trimesh.Scene, **kwargs):
        assert NotImplementedError

    def render(self, world: TrayWorld, debug: bool = False, **kwargs) -> np.array:
        scene = world.get_scene()
        self.set_camera_pose(scene, **kwargs)
        img = world.render(show=debug, array=True, topdown=False,
                           resolution=self.resolution, show_axis=False, **self.render_args)
        return img

    def render_gif(self,
                   gif_name: str = None,
                   num_frames: int = 10,
                   pause: int = 1,
                   debug: bool = False,
                   save_pngs: bool = False):
        images = []
        num_frames = 1 if debug else num_frames
        for t in range(num_frames):
            world = self.world_class(**self.world_args)
            world.sample_scene(**self.world_sampler_args)
            img = self.render(world, debug=debug, counter=t)
            images.append(img)
            if gif_name is None:
                gif_name = f"{self.name}_{world.name}"
        export_gif(images, join(RENDER_PATH, f'{gif_name}.gif'), pause=pause, save_pngs=save_pngs)
        # files = [join(RENDER_PATH, f) for f in listdir(RENDER_PATH) if world.name in f and '.png' in f]
        # for f in files:
        #     os.remove(f)


class TopDownVisualizer(WorldVisualizer):
    """ look at the tray from topdown """
    def __init__(self, **kwargs):
        super(TopDownVisualizer, self).__init__(**kwargs)
        self.orthographic = True

    def set_camera_pose(self, scene: trimesh.Scene, **kwargs):
        adjust_camera_topdown(scene, resolution=self.resolution)


class OrbitVisualizer(WorldVisualizer):
    """ rotate around the tray with it in the center
    """
    def __init__(self, theta: float = 45, delta: float = 45, h: float = 0.5, **kwargs):
        """
            theta is the angle between the camera ray and z-axis
            rotate with delta increments (in degrees)
        """
        super(OrbitVisualizer, self).__init__(**kwargs)
        self.theta = theta
        self.delta = delta
        self.world_args['h'] = h
        self.world_args['orthographic'] = False

    def set_camera_pose(self, scene: trimesh.Scene, **kwargs):
        adjust_camera_topdown(scene, resolution=self.resolution)

    # def set_camera_pose(self, scene: trimesh.Scene, counter: int = 0):
    #     """ TODO: not working yet """
    #     from scipy.spatial.transform import Rotation
    #
    #     dx = np.deg2rad(self.theta)
    #     dz = np.deg2rad(counter * self.delta)
    #     rotation = np.eye(3) @ Rotation.from_euler('XYZ', [dx, np.pi, dz]).as_matrix()
    #     transform = np.eye(4)
    #     transform[:3, :3] = rotation
    #
    #     z = 30
    #     r = z * np.tan(dx)
    #     x = r * np.cos(dz)
    #     y = r * np.sin(dz)
    #     transform[0:3, 3] = np.array([x, y, z])
    #     print(f'--------- {counter} ----------')
    #     print(np.round(transform, 2))
    #     print()
    #     adjust_camera_tilted(scene, transform, resolution=self.resolution)


def test_top_down_shape_setting():
    renderer = TopDownVisualizer(world_class=ShapeSettingWorld, resolution=(1600, 1200))
    renderer.render_gif(debug=False, num_frames=10, pause=1)


def test_orbit_shape_setting():
    world_sampler_args = dict(case=0)
    renderer = OrbitVisualizer(world_class=ShapeSettingWorld, resolution=(1600, 1200),
                               theta=45, delta=45, world_sampler_args=world_sampler_args)
    renderer.render_gif(debug=False, num_frames=10, pause=1, save_pngs=True)


def test_orbit_random_split():
    renderer = OrbitVisualizer(world_class=RandomSplitWorld, resolution=(1920, 1080),
                               theta=45, delta=45)
    renderer.render_gif(debug=False, num_frames=10, pause=1)


def test_random_split_grids():
    render_args = dict(show_grid=True)
    renderer = TopDownVisualizer(world_class=RandomSplitWorld, resolution=(1920, 1080),
                                 render_args=render_args)
    renderer.render_gif(debug=False, num_frames=10, pause=1)


if __name__ == '__main__':
    # test_top_down_shape_setting()
    # test_orbit_shape_setting()
    test_orbit_random_split()
    # test_random_split_grids()