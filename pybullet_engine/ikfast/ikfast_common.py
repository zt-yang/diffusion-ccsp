#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : ikfast_common.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/26/2022
#
# This file is part of HACL-PyTorch.
# Distributed under terms of the MIT license.

import itertools
from typing import Optional, Iterable, Tuple, List
from collections import namedtuple

import random
import numpy as np
import numpy.random as npr
from jacinle.logging import get_logger
from pybullet_engine.rotationlib import mat2quat, quat2mat
from pybullet_engine.world import BulletWorld
from pybullet_engine.rotation_utils import quat_conjugate, quat_mul

# TODO: lookup robot & tool in dictionary and use if exists

logger = get_logger(__file__)


class IKFastWrapper(object):
    def __init__(
        self,
        world: BulletWorld, module,
        body_id, joint_ids: List[int], free_joint_ids: List[int],
        use_xyzw: bool = True,  # PyBullet uses xyzw.
        max_attempts: int = 1000
    ):
        self.world = world
        self.module = module

        self.body_id = body_id
        self.joint_ids = joint_ids
        self.free_joint_ids = free_joint_ids

        self.use_xyzw = use_xyzw
        self.max_attempts = max_attempts

        joint_info = [self.world.get_joint_info_by_id(self.body_id, joint_id) for joint_id in joint_ids]
        self.joints_lower = np.array([info.joint_lower_limit for info in joint_info])
        self.joints_upper = np.array([info.joint_upper_limit for info in joint_info])
        self.free_joints_lower = np.array([info.joint_lower_limit for info in joint_info if info.joint_index in free_joint_ids])
        self.free_joints_upper = np.array([info.joint_upper_limit for info in joint_info if info.joint_index in free_joint_ids])

        assert len(self.free_joint_ids) + 6 == len(self.joint_ids)

    def fk(self, qpos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pos, mat = self.module.get_fk(qpos)
        quat = mat2quat(mat)
        if self.use_xyzw:
            return pos, quat[[1, 2, 3, 0]]
        return pos, quat

    def ik_internal(self, pos: np.ndarray, quat: np.ndarray, sampled: Optional[np.ndarray] = None) -> List[np.ndarray]:
        if self.use_xyzw:
            quat = quat[[3, 0, 1, 2]]
        mat = quat2mat(quat)
        if sampled is None:
            solutions = self.module.get_ik(mat.tolist(), pos.tolist())
        else:
            solutions = self.module.get_ik(mat.tolist(), pos.tolist(), list(sampled))

        if solutions is None:
            return list()
        return [np.array(solution) for solution in solutions]

    def get_current_joint_positions(self) -> np.ndarray:
        return np.array([self.world.get_joint_state_by_id(self.body_id, joint_id).position for joint_id in self.joint_ids])

    def get_current_free_joint_positions(self) -> np.ndarray:
        return np.array([self.world.get_joint_state_by_id(self.body_id, joint_id).position for joint_id in self.free_joint_ids])

    def gen_ik(self, pos: np.ndarray, quat: np.ndarray, last_qpos: Optional[np.ndarray], max_attempts: Optional[int] = None, max_distance: float = float('inf'), verbose: bool = False) -> Iterable[np.ndarray]:
        if last_qpos is None:
            current_joint_positions = self.get_current_joint_positions()
            current_free_joint_positions = self.get_current_free_joint_positions()
        else:
            current_joint_positions = last_qpos
            current_free_joint_positions = [last_qpos[i] for i, joint_id in enumerate(self.joint_ids) if joint_id in self.free_joint_ids]

        generator = itertools.chain([current_free_joint_positions], gen_uniform_sample_joints(self.free_joints_lower, self.free_joints_upper))
        if max_attempts is not None:
            generator = itertools.islice(generator, max_attempts)
        else:
            generator = itertools.islice(generator, self.max_attempts)

        succeeded = False
        for sampled in generator:
            solutions = self.ik_internal(pos, quat, sampled)
            random.shuffle(solutions)
            for solution in solutions:
                if check_joint_limits(solution, self.joints_lower, self.joints_upper):
                    if distance_fn(solution, current_joint_positions) < max_distance:
                        succeeded = True

                        # fk_pos, fk_quat = self.fk(solution.tolist())
                        # print('query (inside): ', pos, quat, 'solution: ', solution, 'fk', fk_pos, fk_quat, 'fk_diff', np.linalg.norm(fk_pos - pos), quat_mul(quat_conjugate(fk_quat), quat)[3])

                        yield solution
                    elif verbose:
                        print(f'IK solution is too far from current joint positions: {solution} vs {current_joint_positions}')

        if not succeeded and max_attempts is None:
            logger.warning(f'Failed to find IK solution for {pos} {quat} after {self.max_attempts} attempts.')


def check_joint_limits(qpos: np.ndarray, lower_limits: np.ndarray, upper_limits: np.ndarray) -> bool:
    return np.all(np.logical_and(qpos >= lower_limits, qpos <= upper_limits))


def uniform_sample_joints(lower_limits: np.ndarray, upper_limits: np.ndarray) -> np.ndarray:
    return np.array([npr.uniform(lower, upper) for lower, upper in zip(lower_limits, upper_limits)])


def gen_uniform_sample_joints(lower_limits: np.ndarray, upper_limits: np.ndarray) -> Iterable[np.ndarray]:
    while True:
        yield uniform_sample_joints(lower_limits, upper_limits)


def random_select_solution(solutions: List[np.ndarray]) -> np.ndarray:
    return random.choice(solutions)


def distance_fn(qpos1: np.ndarray, qpos2: np.ndarray) -> float:
    return np.linalg.norm(np.array(qpos1) - np.array(qpos2))


def closest_select_solution(solutions: List[np.ndarray], current_qpos: np.ndarray) -> np.ndarray:
    return min(solutions, key=lambda qpos: distance_fn(qpos, current_qpos))

