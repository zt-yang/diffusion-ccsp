#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : client.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/17/2020
#
# This file is part of HACL-PyTorch.
# Distributed under terms of the MIT license.

import time
import six
import os.path as osp
import tempfile
import threading
import collections
import functools
import contextlib
from typing import Any, Optional, Union, Tuple, List, Dict

import pybullet as p
import pybullet_data
import jacinle
import jacinle.io as io

from .world import BulletWorld

__all__ = ['BulletClient']


class BulletP(object):
    def __init__(self, client_id=None):
        self.client_id = client_id

    def set_client_id(self, client_id):
        self.client_id = client_id

    def __getattr__(self, item):
        assert self.client_id is not None
        func = getattr(p, item)
        if callable(func):
            return functools.partial(func, physicsClientId=self.client_id)
        return func


class MouseEvent(collections.namedtuple('_MouseEvent', ['eventType', 'mousePosX', 'mousePosY', 'buttonIndex', 'buttonState'])):
    pass


class BulletClient(object):
    DEFAULT_ENGINE_PARAMETERS = {'numSolverIterations': 10}
    DEFAULT_FPS = 120
    DEFAULT_GRAVITY = (0, 0, -9.8)
    DEFAULT_ASSETS_ROOT = osp.join(osp.dirname(osp.abspath(__file__)), 'models', 'assets')

    def __init__(
        self,
        assets_root: Optional[str] = None,
        is_gui: bool = False,
        realtime_gui: bool = False,
        fps: Optional[int] = None,
        render_fps: Optional[int] = None,
        gravity: Optional[Union[Tuple[float], float]] = None,
        engine_parameters: Optional[Dict[str, Any]] = None,
        connect: bool = True,
        client_id: int = -1,
        save_video: Optional[str] = None,
        width: Optional[int] = 960,
        height: Optional[int] = 960,
    ):
        if not is_gui:
            render_fps = 0

        self.is_gui = is_gui
        self.realtime_gui = realtime_gui
        self.fps = fps if fps is not None else type(self).DEFAULT_FPS
        self.render_fps = render_fps if render_fps is not None else self.fps
        self.gravity = canonize_gravity(gravity if gravity is not None else type(self).DEFAULT_GRAVITY)
        self.engine_parameters = engine_parameters
        self.client_id = None
        self.assets_root = assets_root if assets_root is not None else type(self).DEFAULT_ASSETS_ROOT
        self.save_video = save_video

        self.w = BulletWorld()
        self.p = BulletP()
        self.width = width
        self.height = height

        if client_id == -1:
            if connect:
                self.connect()
        else:
            self.client_id = client_id
            self.w.set_client_id(self.client_id)
            self.p.set_client_id(self.client_id)

    @property
    def world(self):
        """Alias for `self.w`."""
        return self.w

    @contextlib.contextmanager
    def with_fps(self, fps: Optional[int] = None, render_fps: Optional[int] = None):
        current_fps, current_render_fps = self.fps, self.render_fps
        if fps is not None:
            self.fps = fps
        if render_fps is not None:
            self.render_fps = render_fps
        elif fps is not None:
            self.render_fps = fps
        yield
        self.fps, self.render_fps = current_fps, current_render_fps

    def connect(self, suppress_warnings: bool = True):
        if suppress_warnings:
            with jacinle.suppress_stdout():
                self._connect()
        else:
            self._connect()

    def _connect(self):
        options = ''
        if self.save_video:
            options += f'--mp4="{self.save_video}" --mp4fps=60'
        if self.width is not None:
            options += ' --width={}'.format(self.width)
        if self.height is not None:
            options += ' --height={}'.format(self.height)
        self.client_id = p.connect(p.GUI if self.is_gui else p.DIRECT, options=options)

        if self.save_video or (self.is_gui and self.realtime_gui):
            p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 1, physicsClientId=self.client_id)

        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
        if self.engine_parameters is not None:
            p.setPhysicsEngineParameter(physicsClientId=self.client_id, **self.engine_parameters)
        else:
            p.setPhysicsEngineParameter(physicsClientId=self.client_id, **type(self).DEFAULT_ENGINE_PARAMETERS)

        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0, physicsClientId=self.client_id)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1, physicsClientId=self.client_id)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 1, physicsClientId=self.client_id)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 1, physicsClientId=self.client_id)

        # Disable the GUI (e.g., synthetic camera views and parameters).
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.client_id)
        # p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1, physicsClientId=self.client_id)

        p.setPhysicsEngineParameter(enableFileCaching=0, physicsClientId=self.client_id)

        if self.assets_root is not None:
            file_io = p.loadPlugin('fileIOPlugin', physicsClientId=self.client_id)
            if file_io >= 0:
                p.executePluginCommand(file_io, textArgument=self.assets_root, intArgs=[p.AddFileIOAction], physicsClientId=self.client_id)
            else:
                raise RuntimeError('pybullet: cannot load FileIO!')
            p.setAdditionalSearchPath(self.assets_root, physicsClientId=self.client_id)

        # NB(Jiayuan Mao @ 10/04): also add the temp dir to the asset path so that we can load JIT URDF files.
        p.setAdditionalSearchPath(tempfile.gettempdir(), physicsClientId=self.client_id)

        p.setGravity(*self.gravity, physicsClientId=self.client_id)
        p.setTimeStep(1.0 / self.fps, physicsClientId=self.client_id)
        self.w.set_client_id(self.client_id)
        self.p.set_client_id(self.client_id)

    def is_connected(self):
        return p.isConnected(physicsClientId=self.client_id)

    def has_gui(self):
        return p.getConnectionInfo(physicsClientId=self.client_id)['connectionMethod'] == p.GUI

    def reset_world(self):
        p.resetSimulation(physicsClientId=self.client_id)
        p.setGravity(*self.gravity, physicsClientId=self.client_id)
        p.setTimeStep(1.0 / self.fps, physicsClientId=self.client_id)

        # Should also remember to reset the world record.
        self.w = BulletWorld()
        self.w.set_client_id(self.client_id)

    def disconnect(self):
        p.disconnect(physicsClientId=self.client_id)

    @contextlib.contextmanager
    def disable_rendering(self, disable_rendering: bool = True, reset: bool = False, suppress_warnings: bool = True):
        if reset:
            self.reset_world()

        with jacinle.cond_with(
            jacinle.suppress_stdout(),
            suppress_warnings
        ):
            if disable_rendering:
                p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=self.client_id)
            yield
            if disable_rendering:
                p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.client_id)

    @contextlib.contextmanager
    def disable_world_update(self):
        """Temproary disable the world update. Specifically, when loading a new model, the world object `self.w` will not be updated.
        This function also disables rendering of the pybullet debug renderer.
        Thus, this functionality is useful when loading a large number of models."""
        with self.disable_rendering():
            yield

    def step(self, steps=1):
        clock = None
        if self.render_fps > 0:
            clock = jacinle.Clock(1 / self.render_fps)
        for i in range(steps):
            p.stepSimulation(physicsClientId=self.client_id)
            if self.render_fps > 0:
                clock.tick()

    def perform_collision_detection(self):
        p.performCollisionDetection(physicsClientId=self.client_id)

    def load_urdf(self, xml_path, pos=(0, 0, 0), quat=(0, 0, 0, 1), body_name: Optional[str] = None, group: Optional[str] = '__UNSET__', static=False, scale: float = 1.0, rgba=None, notify_world_update=True) -> int:
        xml_path = self._canonize_asset_path(xml_path)
        pos, quat = canonize_default_pos_and_quat(pos, quat)
        ret = p.loadURDF(xml_path, pos, quat, useFixedBase=static, globalScaling=scale, physicsClientId=self.client_id, flags=p.URDF_USE_SELF_COLLISION)
        if notify_world_update:
            if group == '__UNSET__':
                group = 'fixed' if static else 'rigid'
            self.w.notify_update(ret, body_name=body_name, group=group)
        if rgba is not None:
            self.w.change_visual_color(ret, rgba=rgba)
        return ret

    def load_urdf_template(self, xml_path: str, replaces: Dict[str, Any], pos=None, quat=None, return_xml=False,
                           **kwargs) -> int:
        xml_path = self._canonize_asset_path(xml_path)
        with open(xml_path) as f:
            xml_content = f.read()
        for k, v in sorted(replaces.items(), key=lambda x: len(x[0]), reverse=True):
            if isinstance(v, (tuple, list)):
                for i in range(len(v)):
                    xml_content = xml_content.replace(k + str(i), str(v[i]))
            else:
                xml_content = xml_content.replace(k, str(v))

        if return_xml:
            return xml_content
        with io.tempfile('w', '.xml') as f:
            f.write(xml_content)
            f.flush()
            return self.load_urdf(f.name, pos=pos, quat=quat, **kwargs)

    def load_mjcf(self, xml_path, pos=(0, 0, 0), quat=(0, 0, 0, 1), body_name=None, group='__UNSET__', static=False, notify_world_update=True) -> int:
        xml_path = self._canonize_asset_path(xml_path)
        pos, quat = canonize_default_pos_and_quat(pos, quat)
        ret = p.loadMJCF(xml_path, pos, quat, useFixedBase=static, physicsClientId=self.client_id, flags=p.MJCF_COLORS_FROM_FILE)
        if notify_world_update:
            if group == '__UNSET__':
                group = 'fixed' if static else 'rigid'
            self.w.notify_update(ret, body_name=body_name, group=group)
        return ret

    def loads_mjcf(self, xml_content, pos=None, quat=None, save_to=None, **kwargs) -> int:
        if not isinstance(xml_content, six.string_types):
            xml_content = io.dumps_xml(xml_content)

        if save_to is not None:
            with open(save_to, 'w') as f:
                f.write(xml_content)

        with io.tempfile('w', '.xml') as f:
            f.write(xml_content)
            f.flush()
            return self.load_mjcf(f.name, pos=pos, quat=quat, **kwargs)

    def _canonize_asset_path(self, path):
        return path.replace('assets://', self.assets_root + '/')

    def remove_body(self, body_id):
        return p.removeBody(body_id, physicsClientId=self.client_id)

    def get_mouse_events(self) -> List[MouseEvent]:
        return list(MouseEvent(*event) for event in self.p.getMouseEvents())

    def update_viewer(self):
        self.get_mouse_events()

    def wait_for_duration(self, duration):
        t0 = time.time()
        while time.time() - t0 <= duration:
            self.update_viewer()

    def wait_forever(self):
        while True:
            self.update_viewer()

    def wait_for_user(self, message='Press enter to continue...'):
        import platform
        if self.has_gui() and platform.system() == 'Darwin':
            # OS X doesn't multi-thread the OpenGL visualizer
            return self._threaded_input(message)
        return input(message)

    def timeout(self, duration: float):
        return jacinle.timeout(duration, fps=self.fps)

    def _threaded_input(self, *args, **kwargs):
        # OS X doesn't multi-thread the OpenGL visualizer
        data = []
        thread = threading.Thread(target=lambda: data.append(input(*args, **kwargs)), args=[])
        thread.start()
        try:
            while thread.is_alive():
                self.update_viewer()
        finally:
            thread.join()
        return data[-1]


def canonize_gravity(gravity):
    if isinstance(gravity, (int, float)):
        return (0, 0, gravity)
    else:
        gravity = tuple(gravity)
        assert len(gravity) == 3
        return gravity


def canonize_default_pos_and_quat(pos, quat):
    if pos is None:
        pos = (0, 0, 0)
    if quat is None:
        quat = (0, 0, 0, 1)
    return pos, quat

