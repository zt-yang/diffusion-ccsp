#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : ur5_robot.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 08/08/2022
#
# This file is part of HACL-PyTorch.
# Distributed under terms of the MIT license.

import time
import contextlib
import numpy as np
import pybullet as p
import jacinle

from typing import Optional, Tuple, Union
from pybullet_engine.client import BulletClient
from pybullet_engine.models.robot import Robot, Gripper, GripperObjectIndices, RobotActionPrimitive
from pybullet_engine.models.ur5.ur5_gripper import UR5GripperType
from pybullet_engine.rotation_utils import compose_transformation

__all__ = ['UR5Robot', 'UR5ReachAndPick', 'UR5ReachAndPlace', 'UR5PlanarMove']

logger = jacinle.get_logger(__file__)

g_debug_options = jacinle.FileOptions(__file__, show_timeout_warning=True)


class UR5Robot(Robot):
    UR5_FILE = 'assets://ur5/ur5.urdf'
    UR5_JOINT_HOMES = np.array([-1, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi

    def __init__(
        self,
        client: BulletClient,
        body_name='ur5',
        gripper_type: str = 'suction',
        gripper_objects: Optional[GripperObjectIndices] = None,
        pos: Optional[np.ndarray] = None,
        quat: Optional[np.ndarray] = None,
    ):
        super().__init__(client, body_name, gripper_objects)

        self.gripper_type = UR5GripperType.from_string(gripper_type)

        self.ur5 = client.load_urdf(type(self).UR5_FILE, pos=pos, quat=quat, static=True, body_name=body_name, group=None)

        ur5_joints = client.w.get_joint_info_by_body(self.ur5)
        self.ur5_joints = [joint.joint_index for joint in ur5_joints if joint.joint_type == p.JOINT_REVOLUTE]
        self.ur5_ee_tip = 10

        gripper_cls = self.gripper_type.get_class()
        self.gripper: Optional[Gripper] = None
        if gripper_cls is not None:
            self.gripper = gripper_cls(client, self.ur5, 9, self.gripper_objects)

        self.reset_home_qpos()
        self.ee_home_pos, self.ee_home_quat = self.get_ee_pose()

        self.reach_and_pick = UR5ReachAndPick(self)
        self.reach_and_place = UR5ReachAndPlace(self)
        self.planar_move = UR5PlanarMove(self)

        self._ignore_physics = False

    @contextlib.contextmanager
    def ignore_physics(self, ignore_physics: bool = True):
        old_ignore_physics = self._ignore_physics
        self._ignore_physics = ignore_physics
        yield
        self._ignore_physics = old_ignore_physics

    def set_ignore_physics(self, ignore_physics: bool = True):
        self._ignore_physics = ignore_physics

    def get_robot_body_id(self) -> int:
        return self.ur5

    def get_qpos(self) -> np.ndarray:
        return np.array([self.client.p.getJointState(self.ur5, i)[0] for i in self.ur5_joints], dtype=np.float32)

    def set_qpos(self, qpos: np.ndarray) -> None:
        self.world.set_batched_qpos_by_id(self.ur5, self.ur5_joints, qpos)

    def get_home_qpos(self) -> np.ndarray:
        return type(self).UR5_JOINT_HOMES

    def reset_home_qpos(self):
        self.client.w.set_batched_qpos_by_id(self.ur5, self.ur5_joints, type(self).UR5_JOINT_HOMES)
        if self.gripper is not None:
            self.gripper.release()

    def get_ee_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        state = self.world.get_link_state_by_id(self.ur5, self.ur5_ee_tip)
        return state.position, state.orientation

    def get_ee_home_pos(self) -> np.ndarray:
        return self.ee_home_pos

    def get_ee_home_quat(self) -> np.ndarray:
        return self.ee_home_quat

    def get_gripper_state(self) -> Optional[bool]:
        return self.gripper.activated if self.gripper is not None else None

    def fk(self, qpos: np.ndarray, link_name_or_id: Optional[Union[str, int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        with self.world.save_body(self.ur5):
            self.set_qpos(qpos)
            state = self.world.get_link_state_by_id(self.ur5, self.ur5_ee_tip)
            return state.position, state.orientation

    def ik_pybullet(self, pos: np.ndarray, quat: np.ndarray, force: bool = False, verbose: bool = False) -> Optional[np.ndarray]:
        """Calculate joint configuration with inverse kinematics."""
        joints = self.client.p.calculateInverseKinematics(
            bodyUniqueId=self.ur5,
            endEffectorLinkIndex=self.ur5_ee_tip,
            targetPosition=pos,
            targetOrientation=quat,
            lowerLimits=[-3 * np.pi / 2, -2.3562, -17, -17, -17, -17],
            upperLimits=[-np.pi / 2, 0, 17, 17, 17, 17],
            jointRanges=[np.pi, 2.3562, 34, 34, 34, 34],  # * 6,
            restPoses=type(self).UR5_JOINT_HOMES.tolist(),
            maxNumIterations=100,
            residualThreshold=1e-5,
        )
        joints = np.float32(joints)
        joints[2:] = (joints[2:] + np.pi) % (2 * np.pi) - np.pi
        return joints

    def ikfast(
        self,
        pos: np.ndarray,
        quat: np.ndarray,
        last_qpos: Optional[np.ndarray] = None,
        max_attempts: int = 1000,
        max_distance: float = float('inf'),
        error_on_fail: bool = True,
        verbose: bool = False
    ) -> Optional[np.ndarray]:
        raise NotImplementedError('IKFast is not implemented for UR5.')

    def move_qpos(self, targj, speed=0.01, timeout=10):
        """Move UR5 to target joint configuration."""
        for _ in jacinle.timeout(timeout):
            currj = [self.client.p.getJointState(self.ur5, i)[0] for i in self.ur5_joints]
            currj = np.asarray(currj)
            diffj = targj - currj

            # pos, quat = self.client.w.get_link_state_by_id(self.ur5, self.ur5_ee_tip)

            if all(np.abs(diffj) < 5e-2):
                return True

            # Move with constant velocity
            norm = np.linalg.norm(diffj)
            v = diffj / norm if norm > 0 else 0
            stepj = currj + v * min(speed, norm)

            if self._ignore_physics:
                self.set_qpos(stepj)
                self.gripper.sync_gripper_qpos()
                time.sleep(1 / self.client.fps)
            else:
                gains = np.ones(len(self.ur5_joints))
                self.client.p.setJointMotorControlArray(
                    bodyIndex=self.ur5,
                    jointIndices=self.ur5_joints,
                    controlMode=p.POSITION_CONTROL,
                    targetPositions=stepj,
                    positionGains=gains
                )
                self.client.step()

        if g_debug_options.show_timeout_warning:
            logger.warning(f'{self.body_name}: move qpos exceeded {timeout} second timeout.')
        return False

    def activate_gripper(self):
        if self.gripper is not None:
            return self.gripper.activate()

    def release_gripper(self):
        if self.gripper is not None:
            return self.gripper.release()


class UR5ReachAndPick(RobotActionPrimitive):
    robot: UR5Robot

    def __init__(self, robot: UR5Robot, height=0.32, speed=0.01):
        super().__init__(robot)
        self.height = height
        self.speed = speed

    def __call__(self, pos: np.ndarray, quat: np.ndarray, timeout: float = 10, speed: Optional[float] = None) -> bool:
        """Execute reach and pick primitive.

        Returns:
            success: if the pick was successful.
        """
        if speed is None:
            speed = self.speed

        # Execute picking primitive.
        prepick_pos, prepick_quat = compose_transformation(pos, quat, (0, 0, self.height), (0, 0, 0, 1))
        postpick_pos, postpick_quat = compose_transformation(pos, quat, (0, 0, self.height), (0, 0, 0, 1))

        with self.robot.client.with_fps(30):
            succ = self.robot.move_pose(prepick_pos, prepick_quat, speed=self.speed)
        if not succ:
            return False

        pos_delta = np.array([0, 0, -0.001], dtype=np.float32)
        target_pos = prepick_pos
        for _ in jacinle.timeout(timeout, self.client.fps):
            if self.robot.gripper.detect_contact():
                break
            target_pos = target_pos + pos_delta
            succ = self.robot.move_pose(target_pos, prepick_quat, speed=self.speed)
            if not succ:
                return False

        self.robot.activate_gripper()

        succ = self.robot.move_pose(postpick_pos, postpick_quat, self.speed)
        pick_success = self.robot.gripper.check_grasp()
        if not succ or not pick_success:
            return False

        return True


class UR5ReachAndPlace(RobotActionPrimitive):
    robot: UR5Robot

    def __init__(self, robot: UR5Robot, height=0.32, speed=0.01):
        super().__init__(robot)
        self.height = height
        self.speed = speed

    def __call__(self, pos: np.ndarray, quat: np.ndarray, release: bool = True, timeout: float = 10, speed: Optional[float] = None) -> bool:
        """Execute reach and place primitive.

        Returns:
            success: if the place was successful.
        """
        if speed is None:
            speed = self.speed

        self.robot.activate_gripper()

        preplace_pos, preplace_quat = compose_transformation(pos, quat, (0, 0, self.height), (0, 0, 0, 1))
        postplace_pos, postplace_quat = compose_transformation(pos, quat, (0, 0, self.height), (0, 0, 0, 1))

        target_pos = preplace_pos
        pos_delta = np.array([0, 0, -0.001], dtype=np.float32)

        for _ in jacinle.timeout(timeout, self.client.fps):
            if self.robot.gripper.detect_contact():
                break
            target_pos = target_pos + pos_delta
            succ = self.robot.move_pose(target_pos, preplace_quat, self.speed)
            if not succ:
                return False

        succ = True
        if release:
            self.robot.release_gripper()
            succ = self.robot.move_pose(postplace_pos, postplace_quat, self.speed)

        return succ


class UR5PlanarMove(RobotActionPrimitive):
    robot: UR5Robot

    def __init__(self, robot: UR5Robot, speed=0.01):
        super().__init__(robot)
        self.speed = speed

    def __call__(self, pos_xy: np.ndarray, speed=None, timeout: float = 10) -> bool:
        """Execute planar move primitive.

        Returns:
            success: if the move was successful.
        """

        if speed is None:
            speed = self.speed

        robot = self.robot
        link_state = self.robot.client.w.get_link_state_by_id(robot.ur5, robot.ur5_ee_tip)

        target_pos = np.array([pos_xy[0], pos_xy[1], link_state.position[2]], dtype=np.float32)
        target_quat = link_state.orientation

        for _ in jacinle.timeout(timeout, self.client.fps):
            current_link_state = self.robot.client.w.get_link_state_by_id(robot.ur5, robot.ur5_ee_tip)
            current_pos = current_link_state.position

            diff_pos = target_pos - current_pos
            if all(np.abs(diff_pos) < 3e-2):
                return True

            norm = np.linalg.norm(diff_pos)
            v = diff_pos / norm if norm > 0 else 0
            step_pos = current_pos + v * speed
            succ = self.robot.move_pose(step_pos, target_quat, self.speed)
            if not succ:
                return False

        return True
