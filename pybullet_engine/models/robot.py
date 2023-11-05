#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : robot.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 08/24/2022
#
# This file is part of HACL-PyTorch.
# Distributed under terms of the MIT license.

"""This file contains basic classes for defining robots and their action primitives."""

import contextlib

import numpy as np
from typing import Optional, Union, Iterable, Tuple, List, Dict, Callable

from jacinle.logging import get_logger
from jacinle.utils.enum import JacEnum
from pybullet_engine.client import BulletClient
from pybullet_engine.models.base import BulletModel

logger = get_logger(__file__)

__all__ = ['Gripper', 'GripperObjectIndices', 'IKMethod', 'Robot', 'RobotActionPrimitive']


class Gripper(BulletModel):
    """Base gripper class."""

    def __init__(self, client: BulletClient):
        super().__init__(client)
        self.activated = False

    def step(self):
        """This function can be used to create gripper-specific behaviors."""
        return

    def activate(self, objects):
        return

    def release(self):
        return


GripperObjectIndices = Dict[str, List[int]]


class IKMethod(JacEnum):
    PYBULLET = 'pybullet'
    IKFAST = 'ikfast'


class Robot(BulletModel):
    def __init__(self, client: BulletClient, body_name: Optional[str] = None, gripper_objects: Optional[GripperObjectIndices] = None, current_interface='pybullet', ik_method: Union[str, IKMethod] = 'pybullet'):
        super().__init__(client)
        self.body_name = body_name if body_name is not None else self.__class__.__name__
        self.gripper_objects = gripper_objects
        if self.gripper_objects is None:
            self.gripper_objects = self.client.w.body_groups

        self.current_interface = current_interface
        self.primitive_actions: Dict[Tuple[str, str], Callable] = dict()
        self.ik_method = IKMethod.from_string(ik_method)
        self.warnings_suppressed = False

    @contextlib.contextmanager
    def suppress_warnings(self):
        self.warnings_suppressed = True
        yield
        self.warnings_suppressed = False

    def register_action(self, name: str, func: Callable, interface: Optional[str] = None) -> None:
        """Register an action primitive.

        Args:
            name (str): Name of the action primitive.
            func (Callable): Function that implements the action primitive.
            interface (Optional[str], optional): Interface of the action primitive. Defaults to None.
                If None, the action primitive is registered for the current interface.

        Raises:
            ValueError: If the action primitive is already registered.
        """
        if interface is None:
            interface = self.current_interface
        if (name, interface) in self.primitive_actions:
            raise ValueError(f'Action primitive {name} for interface {interface} already registered.')
        self.primitive_actions[(interface, name)] = func

    def do(self, action_name: str, *args, **kwargs) -> bool:
        """Execute an action primitive.

        Args:
            action_name (str): Name of the action primitive.

        Returns:
            bool: True if the action primitive is successful.
        """
        assert (self.current_interface, action_name) in self.primitive_actions, f'Action primitive {action_name} for interface {self.current_interface} not registered.'
        return self.primitive_actions[(self.current_interface, action_name)](*args, **kwargs)

    def set_ik_method(self, ik_method: Union[str, IKMethod]):
        self.ik_method = IKMethod.from_string(ik_method)

    def get_robot_body_id(self) -> int:
        """Get the pybullet body ID of the robot.

        Returns:
            int: Body ID of the robot.
        """

        raise NotImplementedError()

    def get_qpos(self) -> np.ndarray:
        """Get the current joint configuration.

        Returns:
            np.ndarray: Current joint configuration.
        """
        raise NotImplementedError()

    def set_qpos(self, qpos: np.ndarray) -> None:
        """Set the joint configuration.

        Args:
            qpos (np.ndarray): Joint configuration.
        """
        raise NotImplementedError()

    def set_qpos_with_holding(self, qpos: np.ndarray) -> None:
        """Set the joint configuration with the gripper holding the object.

        Args:
            qpos (np.ndarray): Joint configuration.
        """
        return self.set_qpos(qpos)

    def get_home_qpos(self) -> np.ndarray:
        """Get the home joint configuration."""
        raise NotImplementedError()

    def reset_home_qpos(self):
        """Reset the home joint configuration."""
        raise NotImplementedError()

    def get_ee_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the pose of the end effector.

        Returns:
            Tuple[np.ndarray, np.ndarray]: 3D position and 4D quaternion of the end effector.
        """
        raise NotImplementedError()

    def get_ee_home_pos(self) -> np.ndarray:
        """Get the home position of the end effector."""
        raise NotImplementedError()

    def get_ee_home_quat(self) -> np.ndarray:
        """Get the home orientation of the end effector."""
        raise NotImplementedError()

    def get_gripper_state(self) -> Optional[bool]:
        """Get the gripper state.

        Returns:
            Optional[bool]: True if the gripper is activated.
        """
        raise NotImplementedError()

    def fk(self, qpos: np.ndarray, link_name_or_id: Optional[Union[str, int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Forward kinematics.

        Args:
            qpos (np.ndarray): Joint configuration.
            link_name_or_id (Union[str, int], optional): name or id of the link. If not specified, the pose of the end effector is returned.

        Returns:
            Tuple[np.ndarray, np.ndarray]: 3D position and 4D quaternion of the end effector.
        """
        raise NotImplementedError()

    def ik(self, pos: np.ndarray, quat: np.ndarray, force: bool = False, max_distance: float = float('inf'), max_attempts: int = 1000, verbose: bool = False) -> Optional[np.ndarray]:
        """Inverse kinematics.

        Args:
            pos (np.ndarray): 3D position of the end effector.
            quat (np.ndarray): 4D quaternion of the end effector.
            force (bool, optional): Whether to force the IK solver to return a solution. Defaults to False.
                If set, the IK solve may return a solution even if the end effector is not at the given pose.
                This function is useful for debugging and for moving towards a certain direction.
            max_distance (float, optional): Maximum distance between the last qpos and the solution (Only used for IKFast). Defaults to float('inf').
            max_attempts (int, optional): Maximum number of attempts (only used for IKFast). Defaults to 1000.
            verbose (bool, optional): Whether to print debug information.

        Returns:
            np.ndarray: Joint configuration.
        """

        if self.ik_method is IKMethod.PYBULLET:
            assert max_distance == float('inf'), 'max_distance is not supported in PyBullet IK'
            return self.ik_pybullet(pos, quat, force=force, verbose=verbose)
        elif self.ik_method is IKMethod.IKFAST:
            assert force is False, 'force is not supported in ik_fast'
            return self.ikfast(pos, quat, max_distance=max_distance, max_attempts=max_attempts, verbose=verbose)

    def ik_pybullet(self, pos: np.ndarray, quat: np.ndarray, force: bool = False, verbose: bool = False) -> Optional[np.ndarray]:
        """Inverse kinematics using pybullet.

        Args:
            pos (np.ndarray): 3D position of the end effector.
            quat (np.ndarray): 4D quaternion of the end effector.
            force (bool, optional): Whether to force the IK solver to return a solution. Defaults to False.
                If set, the IK solve may return a solution even if the end effector is not at the given pose.
                This function is useful for debugging and for moving towards a certain direction.
            verbose (bool, optional): Whether to print debug information.

        Returns:
            np.ndarray: Joint configuration.
        """
        raise NotImplementedError()

    def ikfast(self, pos: np.ndarray, quat: np.ndarray, last_qpos: Optional[np.ndarray] = None, max_attempts: int = 1000, max_distance: float = float('inf'), error_on_fail: bool = True, verbose: bool = False) -> Optional[np.ndarray]:
        """Inverse kinematics using IKFast.

        Args:
            pos (np.ndarray): 3D position of the end effector.
            quat (np.ndarray): 4D quaternion of the end effector.
            last_qpos (Optional[np.ndarray], optional): Last joint configuration. Defaults to None.
                If None, the current joint configuration is used.
            max_attempts (int, optional): Maximum number of IKFast attempts. Defaults to 1000.
            max_distance (float, optional): Maximum distance between the target pose and the end effector. Defaults to float('inf').
            error_on_fail (bool, optional): Whether to raise an error if the IKFast solver fails. Defaults to True.
            verbose (bool, optional): Whether to print debug information.

        Returns:
            np.ndarray: Joint configuration.
        """
        raise NotImplementedError(f'IKFast is not supported for this {self.__class__.__name__}.')

    def move_qpos(self, target_qpos: np.ndarray, speed: float = 0.01, timeout: float = 10.0, local_smoothing: bool = True) -> bool:
        """Move the robot to the given joint configuration.

        Args:
            target_qpos (np.ndarray): Target joint configuration.
            speed (float, optional): Speed of the movement. Defaults to 0.01.
            timeout (float, optional): Timeout of the movement. Defaults to 10.

        Returns:
            bool: True if the movement is successful.
        """
        raise NotImplementedError()

    def move_home(self, timeout: float = 10.0, save_trajectory: bool = False) -> bool:
        """Move the robot to the home configuration."""
        return self.move_qpos(self.get_home_qpos(), timeout=timeout, save_trajectory=save_trajectory)

    def move_pose(self, pos, quat, speed=0.01, force: bool = False, verbose: bool = False) -> bool:
        """Move the end effector to the given pose.

        Args:
            pos (np.ndarray): 3D position of the end effector.
            quat (np.ndarray): 4D quaternion of the end effector.
            speed (float, optional): Speed of the movement. Defaults to 0.01.
            force (bool, optional): Whether to force the IK solver to return a solution. Defaults to False.
                If set, the IK solve may return a solution even if the end effector is not at the specified pose.

        Returns:
            bool: True if the movement is successful.
        """

        if verbose:
            logger.info(f'{self.body_name}: Moving to pose: {pos}, {quat}.')
        target_qpos = self.ik(pos, quat, force=force)
        rv = self.move_qpos(target_qpos, speed)
        return rv

    def move_qpos_trajectory(self, qpos_trajectory: Iterable[np.ndarray], speed: float = 0.01, timeout: float = 10, first_timeout: float = None, local_smoothing: bool = True, verbose: bool = False) -> bool:
        """Move the robot along the given joint configuration trajectory.

        Args:
            qpos_trajectory (Iterable[np.ndarray]): Joint configuration trajectory.
            speed (float, optional): Speed of the movement. Defaults to 0.01.
            timeout (float, optional): Timeout of the movement. Defaults to 10.
            first_timeout (float, optional): Timeout of the first movement. Defaults to None (same as timeout).

        Returns:
            bool: True if the movement is successful.
        """

        this_timeout = timeout if first_timeout is None else first_timeout
        for target_qpos in qpos_trajectory:
            if verbose:
                logger.info(f'{self.body_name}: Moving to qpos: {target_qpos}.')
            rv = self.move_qpos(target_qpos, speed, this_timeout, local_smoothing=local_smoothing)
            this_timeout = timeout
            # if verbose:
            #     logger.info(f'{self.body_name}: Moved to qpos:  {self.get_qpos()}.')
        if not rv:
            return False
        return True

    def replay_qpos_trajectory(self, qpos_trajectory: Iterable[np.ndarray], verbose: bool = True):
        """Replay a joint confifguartion trajectory. This function is useful for debugging a generated joint trajectory.

        Args:
            qpos_trajectory: joint configuration trajectory.
            verbose: whether to print debug information.
        """
        for i, target_qpos in enumerate(qpos_trajectory):
            if verbose:
                logger.info(f'{self.body_name}: Moving to qpos: {target_qpos}. Trajectory replay {i + 1}/{len(qpos_trajectory)}.')
            self.set_qpos_with_holding(target_qpos)
            self.client.wait_for_duration(0.1)
            # input('Press enter to continue.')


class RobotActionPrimitive(object):
    def __init__(self, robot: Robot):
        self.robot = robot

    def __call__(self, *args, **kwargs) -> bool:
        raise NotImplementedError()

    @property
    def client(self):
        return self.robot.client

    @property
    def p(self):
        return self.robot.client.p

    @property
    def world(self):
        return self.robot.client.w

    @property
    def w(self):
        return self.robot.client.w

    @property
    def warnings_suppressed(self):
        return self.robot.warnings_suppressed