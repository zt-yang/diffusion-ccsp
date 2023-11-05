#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : world.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/17/2020
#
# This file is part of HACL-PyTorch.
# Distributed under terms of the MIT license.

import time
import inspect
import collections
import six
from typing import Any, Optional, Union, Iterable, Tuple, List, Dict

import numpy as np
import open3d as o3d
import pybullet as p

from .camera import CameraConfig
from .rotation_utils import quat_mul, quat_conjugate, rotate_vector_batch, compose_transformation

__all__ = ['BodyState', 'JointState', 'LinkState', 'DebugCameraState', 'JointInfo', 'ContactInfo', 'BulletSaver', 'GroupSaver', 'BodyStateSaver', 'JointStateSaver', 'BodyFullStateSaver', 'WorldSaver', 'BulletWorld']


class _NameToIdentifier(object):
    def __init__(self):
        self.string_to_int = dict()
        self.int_to_string = dict()

    def add_int_to_string(self, i, s):
        self.int_to_string[i] = s
        self.string_to_int[s] = i

    def __getitem__(self, item):
        if isinstance(item, six.string_types):
            return self.string_to_int[item]
        return self.int_to_string[item]

    def __len__(self):
        return len(self.int_to_string)

    def __iter__(self):
        yield from self.int_to_string.items()


GlobalIdentifier = collections.namedtuple('GlobalIdentifier', ['type', 'body_id', 'joint_or_link_id'])
JointIdentifier = collections.namedtuple('JointIdentifier', ['body_id', 'joint_id'])
LinkIdentifier = collections.namedtuple('LinkIdentifier', ['body_id', 'link_id'])


class BodyState(collections.namedtuple('_BodyState', ['position', 'orientation', 'linear_velocity', 'angular_velocity'])):
    @property
    def pos(self):
        return self.position

    @property
    def quat_xyzw(self):
        return self.orientation

    @property
    def quat_wxyz(self):
        return (self.orientation[3], self.orientation[0], self.orientation[1], self.orientation[2])

    def get_transformation(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.position, self.orientation

    def get_7dpose(self) -> np.ndarray:
        return np.concatenate([self.position, self.orientation])


class JointState(collections.namedtuple('_JointState', ['position', 'velocity'])):
    pass


class LinkState(collections.namedtuple('_LinkState', ['position', 'orientation'])):
    @property
    def pos(self):
        return self.position

    @property
    def quat_xyzw(self):
        return self.orientation

    @property
    def quat_wxyz(self):
        return (self.orientation[3], self.orientation[0], self.orientation[1], self.orientation[2])

    def get_transformation(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.array(self.position), np.array(self.orientation)

    def get_7dpose(self) -> np.ndarray:
        return np.concatenate([self.position, self.orientation])


class DebugCameraState(collections.namedtuple('_DebugCameraState', [
    'width',
    'height',
    'viewMatrix',
    'projectionMatrix',
    'cameraUp',
    'cameraForward',
    'horizontal',
    'vertical',
    'yaw',
    'pitch',
    'distance',
    'target',
])):

    @property
    def dist(self) -> float:
        return self.distance


class JointInfo(collections.namedtuple('_JointInfo', [
    'joint_index',
    'joint_name',
    'joint_type',
    'qindex',
    'uindex',
    'flags',
    'joint_damping',
    'joint_friction',
    'joint_lower_limit',
    'joint_upper_limit',
    'joint_max_force',
    'joint_max_velocity',
    'link_name',
    'joint_axis',
    'parent_frame_pos',
    'parent_frame_orn',
    'parent_index',
])):
    pass


class CollisionShapeData(collections.namedtuple('_CollisionShapeInfo', [
    'object_unique_id',
    'link_index',
    'shape_type',
    'dimensions',
    'filename',
    'local_pos',
    'local_orn',
    'world_pos',  # the position relative to the world frame
    'world_orn',  # the orientation relative to the world frame
])):
    pass


class ContactInfo(collections.namedtuple('_ContactInfo', [
    'world',
    'contact_flag',
    'body_a',
    'body_b',
    'link_a',
    'link_b',
    'position_on_a',
    'position_on_b',
    'contact_normal_on_b',
    'contact_distance',
    'contact_normal_force',
    'lateral_friction_1',
    'lateral_friction_dir1',
    'lateral_friction_2',
    'lateral_friction_dir2',
])):
    @property
    def body_a_name(self):
        return self.world.body_names[self.body_a]

    @property
    def link_a_name(self):
        return self.world.link_names[self.body_a, self.link_a]

    @property
    def body_b_name(self):
        return self.world.body_names[self.body_b]

    @property
    def link_b_name(self):
        return self.world.link_names[self.body_b, self.link_b]

    @property
    def a_name(self):
        if self.link_a != -1:
            return '@link/' + self.link_a_name
        else:
            return '@body/' + self.body_a_name

    @property
    def b_name(self):
        if self.link_b != -1:
            return '@link/' + self.link_b_name
        else:
            return '@body/' + self.body_b_name

    def hash(self):
        return (self.body_a, self.body_b, self.link_a, self.link_b)


class ConstraintInfo(collections.namedtuple('_ConstraintInfo', [
    'parent_body',
    'parent_joint',
    'child_body',
    'child_link',
    'constraint_type',
    'joint_axis',
    'parent_frame_pos',
    'child_frame_pos',
    'parent_frame_orn',
    'child_frame_orn',
    'max_force',
    'gear_ratio',
    'gear_aux_link',
    'relative_position_target',
    'erp'
])):
    pass


class BulletSaver(object):
    def __init__(self, world: 'BulletWorld'):
        self.world = world

    def save(self):
        pass

    def restore(self):
        raise NotImplementedError()

    def __enter__(self):
        self.save()
        return self

    def __exit__(self, type, value, traceback):
        self.restore()


class GroupSaver(BulletSaver):
    def __init__(self, savers: Iterable[BulletSaver]):
        savers = tuple(savers)
        assert len(savers) > 0, 'savers should not be empty'
        super().__init__(savers[0].world)
        self.savers = savers

    def save(self):
        for saver in self.savers:
            saver.save()

    def restore(self):
        for saver in self.savers:
            saver.restore()


class BodyStateSaver(BulletSaver):
    def __init__(self, world: 'BulletWorld', body_id: int, state: Optional[BodyState] = None, save: bool = True):
        super().__init__(world)
        self.body_id = body_id
        self.body_name = self.world.body_names.int_to_string.get(body_id, None)

        if save:
            self.save(state)

    def save(self, state: Optional[BodyState] = None):
        if state is None:
            state = self.world.get_body_state_by_id(self.body_id)
        self.state = state

    def restore(self):
        self.world.set_body_state_by_id(self.body_id, self.state)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.body_name or self.body_id)


class JointStateSaver(BulletSaver):
    def __init__(self, world: 'BulletWorld', body_id: int, joint_ids: Optional[List[int]] = None, states: Optional[List[JointState]] = None, save: bool = True):
        super().__init__(world)
        self.body_id = body_id
        self.body_name = self.world.body_names.int_to_string.get(body_id, None)

        if save:
            self.save(joint_ids, states)

    def save(self, joint_ids: Optional[List[int]] = None, states: Optional[List[JointState]] = None):
        if joint_ids is None:
            joint_ids = self.world.get_free_joints(self.body_id)
        self.joint_ids = joint_ids
        if states is None:
            states = [self.world.get_joint_state_by_id(self.body_id, joint_id) for joint_id in joint_ids]
        self.states = states

    def restore(self):
        for joint_id, state in zip(self.joint_ids, self.states):
            self.world.set_joint_state_by_id(self.body_id, joint_id, state)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.body_name or self.body_id)


class BodyFullStateSaver(BulletSaver):
    """Save and restore the full state of a body. That is, including both the state of the base link and all joints."""

    def __init__(self, world: 'BulletWorld', body_id: int, save: bool = True):
        super().__init__(world)
        self.body_id = body_id
        self.body_name = self.world.body_names.int_to_string.get(body_id, None)

        self.pose_saver = BodyStateSaver(world, body_id, save=save)
        self.joint_saver = JointStateSaver(world, body_id, save=save)

    def save(self, body_state: Optional[BodyState] = None, joint_ids: Optional[List[int]] = None, joint_states: Optional[List[JointState]] = None):
        self.pose_saver.save(body_state)
        self.joint_saver.save(joint_ids, joint_states)

    def restore(self):
        self.pose_saver.restore()
        self.joint_saver.restore()

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.body_name or self.body_id)


class WorldSaver(BulletSaver):
    def __init__(self, world: 'BulletWorld', body_ids: Optional[List[int]] = None, save: bool = True):
        super().__init__(world)
        self.body_savers = list()
        self.body_ids = body_ids

        if save:
            self.save()

    def save(self):
        if len(self.body_savers) == 0:
            if self.body_ids is None:
                self.body_ids = list(self.world.body_names.int_to_string.keys())
            self.body_savers = [BodyFullStateSaver(self.world, body_id, save=True) for body_id in self.body_ids]
        else:
            for body_saver in self.body_savers:
                body_saver.save()

    def restore(self):
        for body_saver in self.body_savers:
            body_saver.restore()


class WorldSaverV2(BulletSaver):
    def __init__(self, world: 'BulletWorld', save: bool = True):
        super().__init__(world)

        self.saved_id = None
        if save:
            self.save()

    def save(self):
        self.saved_id = p.saveState(physicsClientId=self.world.client_id)

    def restore(self):
        if self.saved_id is None:
            raise RuntimeError('No state has been saved.')
        p.restoreState(self.saved_id, physicsClientId=self.world.client_id)
        p.removeState(self.saved_id, physicsClientId=self.world.client_id)
        self.saved_id = None


def _geometry_cache(type_id: int, geom_name_template: str):
    def wrapper(func):
        sig = inspect.signature(func)

        def wrapped_func(self: 'BulletWorld', *args, **kwargs):
            bound_args = sig.bind(self, *args, **kwargs)
            bound_args.apply_defaults()
            geom_name = geom_name_template.format(**bound_args.arguments)

            if (type_id, geom_name) in self.cached_geometries:
                return self.cached_geometries[type_id, geom_name]
            else:
                value = func(self, *args, **kwargs)
                self.cached_geometries[(type_id, geom_name)] = value
                return value
        return wrapped_func
    return wrapper


class BulletWorld(object):
    def __init__(self, client_id=None):
        self.client_id = client_id

        self.body_names = _NameToIdentifier()
        self.link_names = _NameToIdentifier()
        self.joint_names = _NameToIdentifier()
        self.global_names = _NameToIdentifier()
        self.body_base_link: Dict[str, str] = dict()
        self.body_groups: Dict[str, List[int]] = dict()

        self.cached_geometries: Dict[Tuple[int, str], Any] = dict()

        if self.client_id is not None:
            self.refresh_names()

    def set_client_id(self, client_id):
        self.client_id = client_id
        self.refresh_names()

    @property
    def nr_bodies(self):
        return len(self.body_names)

    @property
    def nr_joints(self):
        return len(self.joint_names)

    @property
    def nr_links(self):
        return len(self.link_names)

    def notify_update(self, body_id, body_name=None, group=None):
        body_info = p.getBodyInfo(body_id, physicsClientId=self.client_id)
        base_link_name = body_info[0].decode('utf-8')
        if body_name is None:
            body_name = body_info[1].decode('utf-8')
        self.body_base_link[body_name] = base_link_name

        self.body_names.add_int_to_string(body_id, body_name)
        self.link_names.add_int_to_string(LinkIdentifier(body_id, -1), body_name + '/' + base_link_name)
        self.global_names.add_int_to_string(GlobalIdentifier('body', body_id, -1), body_name)
        self.global_names.add_int_to_string(GlobalIdentifier('link', body_id, -1), body_name + '/' + base_link_name)

        for j in range(p.getNumJoints(body_id, physicsClientId=self.client_id)):
            joint_info = JointInfo(*p.getJointInfo(body_id, j, physicsClientId=self.client_id))
            joint_name = joint_info[1].decode('utf-8')
            link_name = joint_info[12].decode('utf-8')
            self.joint_names.add_int_to_string(JointIdentifier(body_id, j), joint_name)
            self.joint_names.add_int_to_string(JointIdentifier(body_id, j), body_name + '/' + joint_name)
            self.global_names.add_int_to_string(GlobalIdentifier('joint', body_id, j), joint_name)
            self.global_names.add_int_to_string(GlobalIdentifier('joint', body_id, j), body_name + '/' + joint_name)

            self.link_names.add_int_to_string(LinkIdentifier(body_id, j), link_name)
            self.link_names.add_int_to_string(LinkIdentifier(body_id, j), body_name + '/' + link_name)
            self.global_names.add_int_to_string(GlobalIdentifier('link', body_id, j), link_name)
            self.global_names.add_int_to_string(GlobalIdentifier('link', body_id, j), body_name + '/' + link_name)

        if group is not None:
            if group not in self.body_groups:
                self.body_groups[group] = list()
            self.body_groups[group].append(body_id)

    def refresh_names(self):
        self.global_names = _NameToIdentifier()
        self.body_names = _NameToIdentifier()
        self.joint_names = _NameToIdentifier()
        self.link_names = _NameToIdentifier()
        self.body_base_link = dict()

        for i in range(p.getNumBodies(physicsClientId=self.client_id)):
            body_info = p.getBodyInfo(i, physicsClientId=self.client_id)
            base_link_name, body_name = body_info[0].decode('utf-8'), body_info[1].decode('utf-8')
            self.body_base_link[body_name] = base_link_name

            self.body_names.add_int_to_string(i, body_name)
            self.global_names.add_int_to_string(('body', i), body_name)
            self.link_names.add_int_to_string((i, -1), body_name + '/' + base_link_name)
            self.global_names.add_int_to_string(('link', i, -1), body_name + '/' + base_link_name)

        for body_id in range(self.nr_bodies):
            body_name = self.body_names[body_id]
            for j in range(p.getNumJoints(body_id, physicsClientId=self.client_id)):
                joint_info = JointInfo(*p.getJointInfo(body_id, j, physicsClientId=self.client_id))
                joint_name = joint_info[1].decode('utf-8')
                link_name = joint_info[12].decode('utf-8')
                self.joint_names.add_int_to_string((body_id, j), joint_name)
                self.joint_names.add_int_to_string((body_id, j), body_name + '/' + joint_name)
                self.global_names.add_int_to_string(('joint', body_id, j), joint_name)
                self.global_names.add_int_to_string(('joint', body_id, j), body_name + '/' + joint_name)

                self.link_names.add_int_to_string((body_id, j), link_name)
                self.link_names.add_int_to_string((body_id, j), body_name + '/' + link_name)
                self.global_names.add_int_to_string(('link', body_id, j), link_name)
                self.global_names.add_int_to_string(('link', body_id, j), body_name + '/' + link_name)

    def get_body_name(self, body_id: int) -> str:
        return self.body_names.int_to_string[body_id]

    def get_link_name(self, body_id: int, link_id: int) -> str:
        return self.link_names.int_to_string[(body_id, link_id)]

    def get_body_index(self, body_name: str) -> int:
        return self.body_names.string_to_int[body_name]

    def get_link_index(self, link_name: str) -> LinkIdentifier:
        return self.link_names.string_to_int[link_name]

    def get_link_index_with_body(self, body_id: int, link_name: str) -> int:
        body_name = self.body_names.int_to_string[body_id]
        return self.link_names.string_to_int[body_name + '/' + link_name].link_id

    def get_xpos(self, name, type=None):
        info = self.get_state(name, type)
        assert isinstance(info, (BodyState, LinkState))
        return info.position

    def get_xquat(self, name, type=None):
        info = self.get_state(name, type)
        assert isinstance(info, (BodyState, LinkState))
        return info.orientation

    def get_xmat(self, name, type=None):
        return p.getMatrixFromQuaternion(self.get_xquat(name, type=type))

    def get_qpos(self, name):
        info = self.get_joint_state(name)
        return info.position

    def get_qpos_by_id(self, body_id, joint_id):
        info = self.get_joint_state_by_id(body_id, joint_id)
        return info.position

    def set_qpos(self, name, qpos):
        return p.resetJointState(*self.joint_names[name], qpos, physicsClientId=self.client_id)

    def set_qpos_by_id(self, body_id, joint_id, qpos):
        return p.resetJointState(body_id, joint_id, qpos, physicsClientId=self.client_id)

    def get_batched_qpos(self, names, numpy=True):
        rv = [self.get_qpos(name) for name in names]
        if numpy:
            return np.array(rv)
        return rv

    def get_batched_qpos_by_id(self, body_id, joint_ids, numpy=True):
        rv = [self.get_qpos_by_id(body_id, joint_id) for joint_id in joint_ids]
        if numpy:
            return np.array(rv)
        return rv

    def set_batched_qpos(self, names, qpos):
        for name, q in zip(names, qpos):
            self.set_qpos(name, q)

    def set_batched_qpos_by_id(self, body_index, joint_indices, qpos):
        for index, q in zip(joint_indices, qpos):
            p.resetJointState(body_index, index, q, physicsClientId=self.client_id)

    def get_state(self, name: str, type: Optional[str] = None) -> Union[BodyState, LinkState, JointState]:
        if ':' in name:
            type, name = name.split(':')

        try:
            if type is None:
                record = self.global_names[name]
                type, args = record[0], record[1:]
            elif type == 'body':
                args = (self.body_names[type],)
            elif type == 'joint':
                args = self.joint_names[type]
            elif type == 'link':
                args = self.link_names[type]
            else:
                raise ValueError('Unknown object type: {}.'.format(type))
        except KeyError:
            raise ValueError('Unknown name: {}.'.format(name))

        if type == 'body':
            return self.get_body_state_by_id(*args)
        elif type == 'joint':
            return self.get_joint_state_by_id(*args)
        elif type == 'link':
            return self.get_link_state_by_id(*args)
        else:
            raise ValueError('Unknown object type: {}.'.format(type))

    def __getitem__(self, item):
        return self.get_state(item)

    def get_debug_camera(self) -> DebugCameraState:
        return DebugCameraState(*p.getDebugVisualizerCamera(physicsClientId=self.client_id))

    def set_debug_camera(self, distance: float, yaw: float, pitch: float, target: Tuple[float, float, float]):
        p.resetDebugVisualizerCamera(distance, yaw, pitch, target, physicsClientId=self.client_id)

    def change_visual_color(self, body_id, rgba, link_id=None):
        if link_id is None:
            for i_body_id, link_id in self.link_names.int_to_string:
                if i_body_id == body_id:
                    p.changeVisualShape(body_id, link_id, rgbaColor=rgba, physicsClientId=self.client_id)
        else:
            p.changeVisualShape(body_id, link_id, rgbaColor=rgba, physicsClientId=self.client_id)

    def get_body_state_by_id(self, body_id: int) -> BodyState:
        state = p.getBasePositionAndOrientation(body_id, physicsClientId=self.client_id)
        vel = p.getBaseVelocity(body_id, physicsClientId=self.client_id)
        return BodyState(np.array(state[0]), np.array(state[1]), np.array(vel[0]), np.array(vel[1]))

    def get_body_state(self, body_name: str) -> BodyState:
        return self.get_body_state_by_id(self.body_names[body_name])

    def set_body_state_by_id(self, body_id: int, state: BodyState):
        p.resetBasePositionAndOrientation(body_id, tuple(state.position), tuple(state.orientation), physicsClientId=self.client_id)
        p.resetBaseVelocity(body_id, tuple(state.linear_velocity), tuple(state.angular_velocity), physicsClientId=self.client_id)

    def set_body_state(self, body_name: str, state: BodyState):
        self.set_body_state_by_id(self.body_names[body_name], state)

    def set_body_state2_by_id(self, body_id: int, position: np.ndarray, orientation: np.ndarray):
        p.resetBasePositionAndOrientation(body_id, tuple(position), tuple(orientation), physicsClientId=self.client_id)

    def set_body_state2(self, body_name: str, position: np.ndarray, orientation: np.ndarray):
        self.set_body_state2_by_id(self.body_names[body_name], position, orientation)

    def get_joint_info_by_id(self, body_id: int, joint_id: int) -> JointInfo:
        info = p.getJointInfo(body_id, joint_id, physicsClientId=self.client_id)
        return JointInfo(*info)

    def get_joint_info_by_body(self, body_id: int) -> List[JointInfo]:
        return [self.get_joint_info_by_id(body_id, i) for i in range(p.getNumJoints(body_id, physicsClientId=self.client_id))]

    def get_joint_info(self, joint_name: str) -> JointInfo:
        return self.get_joint_info_by_id(*self.joint_names[joint_name])

    def get_joint_state_by_id(self, body_id: int, joint_id: int) -> JointState:
        state = p.getJointState(body_id, joint_id, physicsClientId=self.client_id)
        return JointState(state[0], state[1])

    def get_joint_state(self, joint_name: str) -> JointState:
        return self.get_joint_state_by_id(*self.joint_names[joint_name])

    def set_joint_state_by_id(self, body_id: int, joint_id: int, state: JointState):
        p.resetJointState(body_id, joint_id, state.position, state.velocity, physicsClientId=self.client_id)

    def set_joint_state(self, joint_name: str, state: JointState):
        self.set_joint_state_by_id(*self.joint_names[joint_name], state)

    def get_link_state_by_id(self, body_id: int, link_id: int, fk=False) -> LinkState:
        if link_id == -1:
            state = p.getBasePositionAndOrientation(body_id, physicsClientId=self.client_id)
            return LinkState(np.array(state[0]), np.array(state[1]))
        state = p.getLinkState(body_id, link_id, computeForwardKinematics=fk, physicsClientId=self.client_id)
        return LinkState(np.array(state[0]), np.array(state[1]))

    def get_link_state(self, link_name: str, fk=False) -> LinkState:
        return self.get_link_state_by_id(*self.link_names[link_name], fk=fk)

    def get_collision_shape_data(self, body_name: str) -> List[CollisionShapeData]:
        body_id = self.body_names[body_name]
        return self.get_collision_shape_data_by_id(body_id)

    def get_collision_shape_data_by_id(self, body_id: int) -> List[CollisionShapeData]:
        output_data = list()
        for i in range(-1, p.getNumJoints(body_id, physicsClientId=self.client_id)):
            link_state = self.get_link_state_by_id(body_id, i, fk=True)

            data_list = p.getCollisionShapeData(body_id, i, physicsClientId=self.client_id)
            for shape_data in data_list:
                world_pos, world_orn = compose_transformation(link_state.position, link_state.orientation, shape_data[5], shape_data[6])
                # world_pos, world_orn = p.multiplyTransforms(link_state.position, link_state.orientation, shape_data[5], shape_data[6], physicsClientId=self.client_id)
                output_data.append(CollisionShapeData(*shape_data, world_pos, world_orn))
        return output_data

    def get_free_joints(self, body_id: int) -> List[int]:
        """Get a list of indices corresponding to all non-fixed joints in the body."""
        all_joints = list()
        for joint_id in range(p.getNumJoints(body_id, physicsClientId=self.client_id)):
            if self.get_joint_info_by_id(body_id, joint_id).joint_type != p.JOINT_FIXED:
                all_joints.append(joint_id)
        return all_joints

    def get_constraint(self, constraint_id: int) -> ConstraintInfo:
        return ConstraintInfo(*p.getConstraintInfo(constraint_id, physicsClientId=self.client_id))

    def get_contact(self, a: Optional[Union[int, LinkIdentifier, str]] = None, b: Optional[Union[int, LinkIdentifier, str]] = None, update: bool = False) -> List[ContactInfo]:
        if update:
            p.performCollisionDetection(physicsClientId=self.client_id)

        kwargs = dict(physicsClientId=self.client_id)

        def update_kwargs(name, value):
            if value is not None:
                if isinstance(value, int):
                    kwargs['body' + name] = value
                elif isinstance(value, (tuple, list)):
                    kwargs['body' + name], kwargs['linkIndex' + name] = value
                else:
                    assert isinstance(value, six.string_types)
                    value = self.global_names[value]
                    if value[0] == 'body':
                        kwargs['body' + name] = value[1]
                    elif value[0] == 'link':
                        kwargs['body' + name], kwargs['linkIndex' + name] = value[1:]
                    else:
                        raise ValueError('get_contact API only allows the specification of body or link.')

        update_kwargs('A', a)
        update_kwargs('B', b)
        while True:
            contacts = p.getContactPoints(**kwargs)
            if contacts is not None:
                break
            # TODO(Jiayuan Mao @ 2023/04/13): sometimes PyBullet returns None, not sure why.
            time.sleep(0.001)
        return [ContactInfo(self, *c) for c in contacts]

    def update_contact(self):
        p.performCollisionDetection(physicsClientId=self.client_id)

    def is_collision_free(self, a: Union[int, LinkIdentifier, str], b: Union[int, LinkIdentifier, str]) -> bool:
        return len(self.get_contact(a, b)) == 0

    def save_world(self) -> WorldSaver:
        return WorldSaver(self)

    def save_world_v2(self) -> WorldSaverV2:
        return WorldSaverV2(self)

    def save_body(self, body_identifier: Union[str, int]) -> BodyFullStateSaver:
        if isinstance(body_identifier, int):
            return BodyFullStateSaver(self, body_identifier)
        else:
            return BodyFullStateSaver(self, self.body_names[body_identifier])

    def save_bodies(self, body_identifiers: List[Union[str, int]]) -> GroupSaver:
        return GroupSaver([self.save_body(b) for b in body_identifiers])

    def render_image(self, config: CameraConfig, image_size: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        view_matrix, proj_matrix = config.get_view_and_projection_matricies(image_size)
        image_size = image_size if image_size is not None else config.image_size
        znear, zfar = config.zrange

        # Render with OpenGL camera settings.
        _, _, color, depth, segm = p.getCameraImage(
            width=image_size[1],
            height=image_size[0],
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            shadow=1,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )

        # Get color image.
        color_image_size = (image_size[0], image_size[1], 4)
        color = np.asarray(color, dtype=np.uint8).reshape(color_image_size)
        color = color[:, :, :3]  # remove alpha channel

        # Get depth image.
        depth_image_size = (image_size[0], image_size[1])
        zbuffer = np.asarray(depth).reshape(depth_image_size)
        depth = zbuffer
        # depth = (zfar + znear - (2. * zbuffer - 1.) * (zfar - znear))
        # depth = (2. * znear * zfar) / depth

        # Get segmentation image.
        segm = np.uint8(segm).reshape(depth_image_size)

        return color, depth, segm

    def get_pointcloud(self, body_id: int, points_per_geom: int = 1000, zero_center: bool = True) -> np.ndarray:
        """Get the point cloud of a body."""

        all_pcds = list()

        body_state = self.get_body_state_by_id(body_id)
        for shape_info in self.get_collision_shape_data_by_id(body_id):
            if shape_info.shape_type == p.GEOM_BOX:
                pcd = self.get_pointcloud_box(shape_info.dimensions, points_per_geom)
            # elif shape_info.shape_type == client.p.GEOM_SPHERE:
            #     return get_point_cloud_sphere(shape_info.dimensions, points_per_geom)
            elif shape_info.shape_type == p.GEOM_MESH:
                pcd = self.get_pointcloud_mesh(shape_info.filename, shape_info.dimensions[0], points_per_geom)
            else:
                raise ValueError(f'Unsupported shape type: {shape_info.shape_type}.')

            if zero_center:
                pos = shape_info.world_pos - body_state.pos
                orn = quat_mul(quat_conjugate(body_state.quat_xyzw), shape_info.world_orn)
            else:
                pos, orn = shape_info.world_pos, shape_info.world_orn

            pcd = self.transform_pcd(pcd, pos, orn)
            all_pcds.append(pcd)

        if len(all_pcds) == 0:
            return np.zeros((0, 3), dtype=np.float32)
        return np.concatenate(all_pcds, axis=0)

    @_geometry_cache(type_id=0, geom_name_template='box_{dimensions[0]}_{dimensions[1]}_{dimensions[2]}_{points_per_geom}')
    def get_pointcloud_box(self, dimensions, points_per_geom=1000) -> np.ndarray:
        """Get a point cloud for a box."""

        total_volume = dimensions[0] * dimensions[1] * dimensions[2]
        density = points_per_geom / total_volume
        linear_density = density ** (1. / 3.)

        x, y, z = np.meshgrid(
            np.linspace(-dimensions[0] / 2, dimensions[0] / 2, int(np.ceil(dimensions[0] * linear_density))),
            np.linspace(-dimensions[1] / 2, dimensions[1] / 2, int(np.ceil(dimensions[1] * linear_density))),
            np.linspace(-dimensions[2] / 2, dimensions[2] / 2, int(np.ceil(dimensions[2] * linear_density))),
        )
        return np.stack([x, y, z], axis=-1).reshape(-1, 3)

    @_geometry_cache(type_id=0, geom_name_template='mesh_{mesh_file}_{mesh_scale}_{points_per_geom}')
    def get_pointcloud_mesh(self, mesh_file, mesh_scale, points_per_geom=1000) -> np.ndarray:
        """Get a point cloud for a mesh."""
        mesh = o3d.io.read_triangle_mesh(mesh_file)
        pcd = mesh.sample_points_poisson_disk(points_per_geom)
        return np.asarray(pcd.points, dtype=np.float32) * mesh_scale

    def transform_pcd(self, raw_pcd, pos, quat_xyzw):
        return rotate_vector_batch(raw_pcd, quat_xyzw) + pos

    def get_mesh(self, body_id: int, zero_center: bool = True, verbose: bool = False, mesh_filename: Optional[str] = None, mesh_scale: float = 1.0) -> o3d.geometry.TriangleMesh:
        """Get the point cloud of a body."""

        base_mesh = o3d.geometry.TriangleMesh()
        body_state = self.get_body_state_by_id(body_id)
        for shape_info in self.get_collision_shape_data_by_id(body_id):
            if shape_info.shape_type == p.GEOM_BOX:
                mesh = self.get_mesh_box(shape_info.dimensions)
            # elif shape_info.shape_type == client.p.GEOM_SPHERE:
            #     return get_point_cloud_sphere(shape_info.dimensions, points_per_geom)
            elif shape_info.shape_type == p.GEOM_MESH:
                if mesh_filename is not None:
                    mesh = self.get_mesh_mesh(mesh_filename, mesh_scale)
                else:
                    mesh = self.get_mesh_mesh(shape_info.filename, shape_info.dimensions[0])
            else:
                raise ValueError(f'Unsupported shape type: {shape_info.shape_type}.')

            if zero_center:
                pos = shape_info.world_pos - body_state.pos
                orn = quat_mul(quat_conjugate(body_state.quat_xyzw), shape_info.world_orn)
            else:
                pos, orn = shape_info.world_pos, shape_info.world_orn

            if verbose:
                print(shape_info)
                print('Zero-center:', zero_center)
                print(f'Transforming the mesh: pos: {pos}, orn: {orn}')

            mesh = self.transform_mesh(mesh, pos, orn)
            base_mesh += mesh

        return base_mesh

    @_geometry_cache(type_id=1, geom_name_template='box_{dimensions[0]}_{dimensions[1]}_{dimensions[2]}')
    def get_mesh_box(self, dimensions) -> o3d.geometry.TriangleMesh:
        """Get a triangle mesh for a box primitive."""
        mesh = o3d.geometry.TriangleMesh.create_box(dimensions[0], dimensions[1], dimensions[2])
        mesh = mesh.translate([-dimensions[0] / 2, -dimensions[1] / 2, -dimensions[2] / 2])
        return mesh

    @_geometry_cache(type_id=1, geom_name_template='mesh_{mesh_file}_{mesh_scale}')
    def get_mesh_mesh(self, mesh_file, mesh_scale) -> o3d.geometry.TriangleMesh:
        """Get a triangle mesh for a mesh primitive."""
        mesh = o3d.io.read_triangle_mesh(mesh_file)
        mesh = mesh.scale(mesh_scale, np.array([0, 0, 0], dtype=np.float64))
        return mesh

    def transform_mesh(self, mesh: o3d.geometry.TriangleMesh, pos, quat_xyzw):
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_quaternion([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        mesh = o3d.geometry.TriangleMesh(mesh)
        mesh = mesh.rotate(rotation_matrix, center=(0, 0, 0))
        mesh = mesh.translate(pos)
        return mesh

    def transform_mesh2(self, mesh: o3d.geometry.TriangleMesh, pos: np.ndarray, quat_xyzw: np.ndarray, current_pos: np.ndarray, current_quat_xyzw: np.ndarray):
        rotation_matrix1 = o3d.geometry.get_rotation_matrix_from_quaternion([current_quat_xyzw[3], -current_quat_xyzw[0], -current_quat_xyzw[1], -current_quat_xyzw[2]])
        rotation_matrix2 = o3d.geometry.get_rotation_matrix_from_quaternion([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        mesh = o3d.geometry.TriangleMesh(mesh)
        mesh.translate(-current_pos)
        mesh.rotate(rotation_matrix1, center=np.array([0, 0, 0]))
        mesh.rotate(rotation_matrix2, center=np.array([0, 0, 0]))
        mesh.translate(pos)
        return mesh
