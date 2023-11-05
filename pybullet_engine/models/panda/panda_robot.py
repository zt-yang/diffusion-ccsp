#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : panda_robot.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 08/23/2022
#
# This file is part of HACL-PyTorch.
# Distributed under terms of the MIT license.

import itertools
import contextlib
import numpy as np
import pybullet as p
from typing import Any, Optional, Union, Iterable, Tuple, Dict

import jacinle
from jacinle.logging import get_logger

from pybullet_engine.algorithms.space import BoxConfigurationSpace, CollisionFreeProblemSpace
from pybullet_engine.algorithms.rrt import birrt
from pybullet_engine.range_utils import Range
from pybullet_engine.interpolation_utils import gen_cubic_spline, get_next_target_cubic_spline
from pybullet_engine.interpolation_utils import gen_linear_spline, get_next_target_linear_spline

from pybullet_engine.client import BulletClient
from pybullet_engine.world import BodyFullStateSaver, ConstraintInfo
from pybullet_engine.models.robot import Robot, GripperObjectIndices, RobotActionPrimitive
from pybullet_engine.rotation_utils import rotate_vector, quat_mul, quat_conjugate

logger = get_logger(__file__)

np.set_printoptions(precision=5, suppress=True)

__all__ = ['PandaRobot']


class PandaRobot(Robot):
    # PANDA_FILE = 'assets://franka_panda/panda.urdf'
    # PANDA_JOINT_HOMES = np.array([0.98, 0.458, 0.31, -2.24, -0.30, 2.66, 2.32])
    PANDA_FILE = 'assets://franka_description/robots/panda_arm_hand.urdf'
    PANDA_JOINT_HOMES = np.array([-0.45105, -0.38886, 0.45533, -2.19163, 0.13169, 1.81720, 0.51563])

    PANDA_GRIPPER_HOME = 0.02
    PANDA_GRIPPER_OPEN = 0.04
    PANDA_GRIPPER_CLOSE = 0.00

    def __init__(
        self,
        client: BulletClient,
        body_name='panda',
        gripper_objects: Optional[GripperObjectIndices] = None,
        use_magic_gripper: bool = True,
        pos: Optional[np.ndarray] = None,
        # pos: Optional[np.ndarray] = (0., 0.172, 0.),
        quat: Optional[np.ndarray] = None,
        # quat: Optional[np.ndarray] = (0, 0, -0.707, 0.707),
        # quat: Optional[np.ndarray] = (0, 0, -0.707, 0.707),
        # quat: Optional[np.ndarray] = (0, 0, 0.707, 0.707),
        # quat: Optional[np.ndarray] = (0, 0, 1, 0),
        ik_fast: bool = False,
    ):
        super().__init__(client, body_name, gripper_objects)

        self.panda = client.load_urdf(type(self).PANDA_FILE, pos=pos, quat=quat, static=True, body_name=body_name, group=None)

        all_joints = client.w.get_joint_info_by_body(self.panda)
        self.panda_joints = [joint.joint_index for joint in all_joints if joint.joint_type == p.JOINT_REVOLUTE]
        self.gripper_joints = [joint.joint_index for joint in all_joints if joint.joint_type == p.JOINT_PRISMATIC]
        assert set(self.gripper_joints) == {9, 10}
        self.panda_ee_tip = 11
        self.ik_fast = ik_fast
        self.ik_fast_wrapper = None

        # for joint_index in self.panda_joints:
        #     self.client.p.changeDynamics(self.panda, joint_index, linearDamping=0.0, angularDamping=0.0)

        # Create a constraint to keep the fingers centered
        # https://github.com/bulletphysics/bullet3/blob/daadfacfff365852ffc96f373c834216a25b11e5/examples/pybullet/gym/pybullet_robots/panda/panda_sim_grasp.py#L46-L54

        c = self.client.p.createConstraint(
            self.panda, 9, self.panda, 10,
            jointType=p.JOINT_GEAR, jointAxis=[1, 0, 0],
            parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0]
        )
        self.client.p.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)

        joint_info = [self.client.w.get_joint_info_by_id(self.panda, i) for i in self.panda_joints]
        self.panda_joints_lower = np.array([joint.joint_lower_limit for joint in joint_info])
        self.panda_joints_upper = np.array([joint.joint_upper_limit for joint in joint_info])

        # State variable: gripper_activated. True if the gripper is activated.
        # For all control actions, make sure to call self._set_gripper_control before steping the simulation.
        self.griper_activated = None

        self.use_magic_gripper = use_magic_gripper
        self.gripper_constraint = None

        self.reset_home_qpos()
        self.ee_home_pos, self.ee_home_quat = self.get_ee_pose()

        self.reach_two_stage = PandaReachTwoStage(self)
        self.grasp = PandaGrasp(self)
        self.planar_push = PandaPlanarPush(self)
        self.push_two_stage = PandaPushTwoStage(self)
        self.pick_and_place = PandaPickAndPlace(self)

        self._cspace = None
        self._cfree_pspace = None

        self.register_action('move_pose', self.move_pose)
        self.register_action('move_home', self.move_home)
        self.register_action('move_home_cfree', self.move_home_cfree)

        self.register_action('grasp', self.grasp)
        self.register_action('open_gripper_free', self.open_gripper_free)
        self.register_action('close_gripper_free', self.close_gripper_free)

        self.register_action('reach_two_stage', self.reach_two_stage)
        self.register_action('planar_push', self.planar_push)
        self.register_action('push_two_stage', self.push_two_stage)
        self.register_action('pick_and_place', self.pick_and_place)
        self.register_action('reach_two_stage', self.reach_two_stage, interface='franka')
        self.register_action('planar_push', self.planar_push, interface='franka')
        self.register_action('pick_and_place', self.pick_and_place, interface='franka')

    def get_robot_body_id(self) -> int:
        return self.panda

    def get_qpos(self) -> np.ndarray:
        return self.client.w.get_batched_qpos_by_id(self.panda, self.panda_joints)

    def set_qpos(self, qpos: np.ndarray) -> None:
        self.client.w.set_batched_qpos_by_id(self.panda, self.panda_joints, qpos)

    def set_qpos_with_holding(self, qpos: np.ndarray) -> None:
        self.set_qpos(qpos)
        if self.gripper_constraint is not None:
            # TODO(Jiayuan Mao @ 2023/03/02): should consider the actual link that the gripper is attached to, and self-collision, etc.
            constraint = self.p.getConstraintInfo(self.gripper_constraint)
            other_body_id = constraint[2]
            parent_frame_pos = constraint[6]
            parent_frame_orn = constraint[8]

            body_pose = self.p.getLinkState(self.panda, self.w.link_names[f'{self.body_name}/panda_leftfinger'].link_id)
            obj_to_body = (parent_frame_pos, parent_frame_orn)
            obj_pose = p.multiplyTransforms(body_pose[0], body_pose[1], *obj_to_body)

            # print('other_body_id:', other_body_id, 'pose:', self.world.get_body_state_by_id(other_body_id).get_transformation())

            self.w.set_body_state2_by_id(other_body_id, obj_pose[0], obj_pose[1])

    def get_home_qpos(self):
        return type(self).PANDA_JOINT_HOMES

    def reset_home_qpos(self):
        self.client.w.set_batched_qpos_by_id(self.panda, self.panda_joints, type(self).PANDA_JOINT_HOMES)
        self.client.w.set_batched_qpos_by_id(self.panda, self.gripper_joints, [type(self).PANDA_GRIPPER_HOME] * 2)
        self.gripper_activated = None

    def get_ee_pose(self, fk=False) -> Tuple[np.ndarray, np.ndarray]:
        state = self.client.w.get_link_state_by_id(self.panda, self.panda_ee_tip, fk=fk)
        return np.array(state.position), np.array(state.orientation)

    def set_ee_pose(self, pos: np.ndarray, quat: np.ndarray) -> bool:
        qpos = self.ik(pos, quat)
        if qpos is None:
            return False
        self.set_qpos(qpos)
        return True

    def get_ee_home_pos(self):
        return self.ee_home_pos

    def get_ee_home_quat(self):
        return self.ee_home_quat

    def get_gripper_state(self) -> Optional[bool]:
        return self.gripper_activated

    def fk(self, qpos: np.ndarray, link_name_or_id: Optional[Union[str, int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        with BodyFullStateSaver(self.client.w, self.panda):
            self.client.w.set_batched_qpos_by_id(self.panda, self.panda_joints, qpos)
            if link_name_or_id is None:
                return self.get_ee_pose(fk=True)
            elif isinstance(link_name_or_id, str):
                state = self.client.w.get_link_state(link_name_or_id, fk=True)
                return np.array(state.position), np.array(state.orientation)
            elif isinstance(link_name_or_id, int):
                state = self.client.w.get_link_state_by_id(self.panda, link_name_or_id, fk=True)
                return np.array(state.position), np.array(state.orientation)
            else:
                raise TypeError(f'link_name_or_id must be str or int, got {type(link_name_or_id)}.')

    def ik_pybullet(self, pos: np.ndarray, quat: np.ndarray, force: bool = False, verbose: bool = False) -> Optional[np.ndarray]:
        joints = self.client.p.calculateInverseKinematics(
            bodyUniqueId=self.panda,
            endEffectorLinkIndex=self.panda_ee_tip,
            targetPosition=pos,
            targetOrientation=quat,
            lowerLimits=self.panda_joints_lower,
            upperLimits=self.panda_joints_upper,
            jointRanges=self.panda_joints_upper - self.panda_joints_lower,
            restPoses=type(self).PANDA_JOINT_HOMES,
            maxNumIterations=100,
            residualThreshold=1e-5,
        )

        if verbose:
            print('IK (solution, lower, upper):\n', np.stack([joints[:7], self.panda_joints_lower, self.panda_joints_upper], axis=0), sep='')

        if np.all(self.panda_joints_lower[:7] <= joints[:7]) and np.all(joints[:7] <= self.panda_joints_upper[:7]):
            qpos = np.array(joints[:7])
            fk_pos, fk_quat = self.fk(qpos)
            if np.allclose(fk_pos, pos, atol=1e-2) and np.abs(quat_mul(quat, quat_conjugate(fk_quat))[3]) > 0.99:
                return qpos

        if force:
            return np.array(joints[:7])

        return None

    def _init_ikfast(self):
        if self.ik_fast_wrapper is None:
            from pybullet_engine.ikfast.ikfast_common import IKFastWrapper
            import pybullet_engine.ikfast.franka_panda.ikfast_panda_arm as ikfast_module
            self.ik_fast_wrapper = IKFastWrapper(
                self.world, ikfast_module,
                body_id=self.panda,
                joint_ids=self.panda_joints,
                free_joint_ids=[self.panda_joints[-1]],
                use_xyzw=True
            )

    def ikfast(self, pos: np.ndarray, quat: np.ndarray, last_qpos: Optional[np.ndarray] = None, max_attempts: int = 1000, max_distance: float = float('inf'), error_on_fail: bool = True, verbose: bool = False) -> Optional[np.ndarray]:
        self._init_ikfast()

        if verbose:
            print('Solving IK for pos:', pos, 'quat:', quat)

        pos_delta = [0, 0, 0.1]
        quat_delta = (0.0, 0.0, 0.9238795325108381, 0.38268343236617297)
        inner_quat = quat_mul(quat, quat_conjugate(quat_delta))
        inner_pos = np.array(pos) - rotate_vector(pos_delta, quat)

        body_state = self.world.get_body_state_by_id(self.panda)
        # TODO(Jiayuan Mao @ 2022/12/27): solve when orientation != identity.
        inner_pos = inner_pos - np.array(body_state.position)
        assert np.allclose(body_state.orientation, [0, 0, 0, 1])

        try:
            ik_solution = list(itertools.islice(self.ik_fast_wrapper.gen_ik(inner_pos, inner_quat, last_qpos=last_qpos, max_attempts=max_attempts, max_distance=max_distance, verbose=False), 1))[0]
        except IndexError:
            if error_on_fail:
                raise
            return None

        if verbose:
            print('IK (solution, lower, upper):\n', np.stack([ik_solution, self.panda_joints_lower, self.panda_joints_upper], axis=0), sep='')
            print('FK:', self.fk(ik_solution))

        # fk_pos, fk_quat = self.fk(ik_solution, 'panda/tool_link')
        # link8_pos, link8_quat = self.fk(ik_solution, 'panda/panda_link8')
        # print('query (outer):', pos, quat, 'solution:', ik_solution, 'fk', fk_pos, fk_quat, 'fk_diff', np.linalg.norm(fk_pos - pos), quat_mul(quat_conjugate(fk_quat), quat)[3])
        # print('query (outer):', 'link8 IN', link8_pos, link8_quat)
        # print('query (outer):', 'linkT IN', fk_pos, fk_quat)
        # print('query (outer):', 'linkT OT', link8_pos + rotate_vector((0, 0, 0.1), link8_quat), quat_mul(link8_quat, quat_delta))
        # print(self.w.get_joint_info_by_id(self.panda, 11))
        # print(self.w.get_joint_state_by_id(self.panda, 11))

        return ik_solution

    def get_configuration_space(self) -> BoxConfigurationSpace:
        if self._cspace is None:
            ranges = list()
            for lower, upper in zip(self.panda_joints_lower, self.panda_joints_upper):
                ranges.append(Range(lower, upper))
            self._cspace = BoxConfigurationSpace(ranges, 0.02)

        return self._cspace

    def is_colliding(self, q: Optional[np.ndarray] = None, return_contacts: bool = False):
        if q is not None:
            self.set_qpos_with_holding(q)
        contacts = self.world.get_contact(self.panda, update=True)

        ignore_bodies = [self.panda]
        if self.gripper_constraint is not None:
            # TODO(Jiayuan Mao @ 2023/03/02): should consider the actual link that the gripper is attached to, and self-collision, etc.
            constraint = self.p.getConstraintInfo(self.gripper_constraint)
            other_body_id = constraint[2]
            contacts += self.world.get_contact(other_body_id, update=True)
            ignore_bodies.append(other_body_id)
            # print('other_body_id:', other_body_id, 'pose:', self.world.get_body_state_by_id(other_body_id).get_transformation())

        filtered_contacts = list()
        for c in contacts:
            if c.body_b not in ignore_bodies:
                filtered_contacts.append(c)
                # jacinle.log_function.print('Detected collision between', c.body_a_name, 'and', c.body_b_name)
                if not return_contacts:
                    return True

        return False if not return_contacts else filtered_contacts

    def get_collision_free_problem_space(self) -> CollisionFreeProblemSpace:
        if self._cfree_pspace is None:
            cspace = self.get_configuration_space()

            def is_colliding(q):
                self.set_qpos(q)
                contacts = self.world.get_contact(self.panda, update=True)

                ignore_bodies = [self.panda]
                if self.gripper_constraint is not None:
                    # TODO(Jiayuan Mao @ 2023/03/02): should consider the actual link that the gripper is attached to, and self-collision, etc.
                    constraint = self.p.getConstraintInfo(self.gripper_constraint)
                    other_body_id = constraint[2]
                    parent_frame_pos = constraint[6]
                    parent_frame_orn = constraint[8]

                    body_pose = self.p.getLinkState(self.panda, self.w.link_names[f'{self.body_name}/panda_leftfinger'].link_id)
                    obj_to_body = (parent_frame_pos, parent_frame_orn)
                    obj_pose = p.multiplyTransforms(body_pose[0], body_pose[1], *obj_to_body)

                    # print('other_body_id:', other_body_id, 'pose:', self.world.get_body_state_by_id(other_body_id).get_transformation())

                    self.w.set_body_state2_by_id(other_body_id, obj_pose[0], obj_pose[1])

                    contacts += self.world.get_contact(other_body_id, update=True)
                    ignore_bodies.append(other_body_id)

                    # print('other_body_id:', other_body_id, 'pose:', self.world.get_body_state_by_id(other_body_id).get_transformation())

                for c in contacts:
                    if c.body_b not in ignore_bodies:
                        # jacinle.log_function.print('Detected collision between', c.body_a_name, 'and', c.body_b_name)
                        return True
                return False

            self._cfree_pspace = CollisionFreeProblemSpace(cspace, is_colliding)

        return self._cfree_pspace

    def rrt_collision_free(self, pos1: np.ndarray, pos2: Optional[np.ndarray] = None, smooth_fine_path: bool = False, disable_renderer: bool = True, **kwargs):
        if pos2 is None:
            pos2 = pos1
            pos1 = self.get_qpos()

        cfree_pspace = self.get_collision_free_problem_space()
        with self.world.save_world_v2(), jacinle.cond_with(self.client.disable_rendering(suppress_warnings=False), disable_renderer):
            path = birrt(cfree_pspace, pos1, pos2, smooth_fine_path=smooth_fine_path, **kwargs)
            if path[0] is not None:
                return True, path[0]
            return False, None

    def move_qpos(self, target_qpos, speed=0.01, timeout=10, gains=1, local_smoothing: bool = True,
                  save_trajectory: bool = False) -> bool:
        last_qpos = None
        world_states = []
        for _ in self.client.timeout(timeout):
            world_states.append(self.world.save_world())

            current_qpos = self.client.w.get_batched_qpos_by_id(self.panda, self.panda_joints)
            diff = target_qpos - current_qpos
            norm = np.linalg.norm(diff, ord=2)

            rel_diff, rel_norm = None, 1e9
            if last_qpos is not None:
                rel_diff = last_qpos - current_qpos
                rel_norm = np.linalg.norm(rel_diff, ord=1)
            else:
                last_qpos = current_qpos
                rel_norm = 1e9

            # pos, quat = self.get_ee_pose()
            # print('  current pos', pos, 'quat', quat)
            # print('  current/target joint state\n', np.stack([current_qpos, target_qpos], axis=0), sep='')
            # print('  diff', diff, 'norm', norm)
            # if last_qpos is not None:
            #     print('  rel_diff', rel_diff, 'rel_norm', rel_norm)

            if norm < 0.01:
                if save_trajectory:
                    return world_states
                return True

            # if rel_norm < 0.001:
            #     return False

            if local_smoothing:
                # If speed is bigger than norm, we clip it to norm.
                step_qpos = current_qpos + diff / norm * min(speed, norm)
            else:
                step_qpos = target_qpos

            vector_gains = np.ones(len(self.panda_joints)) * gains
            self.client.p.setJointMotorControlArray(
                bodyIndex=self.panda,
                jointIndices=self.panda_joints,
                controlMode=p.POSITION_CONTROL,
                targetPositions=step_qpos,
                positionGains=vector_gains,
            )
            self._set_gripper_control()
            self.client.step()

        if not self.warnings_suppressed:
            logger.warning(f'{self.body_name}: Moving timeout ({timeout}s).')
        return False

    def move_qpos_trajectory_v2(
        self, qpos_trajectory: Iterable[np.ndarray],
        step_size: float = 1, gains: float = 0.3,
        atol: float = 0.03, timeout: float = 20,
        verbose: bool = False,
        return_world_states: bool = False,
    ):
        qpos_trajectory = np.array(qpos_trajectory)
        qpos_trajectory_dedup = list()
        last_qpos = None
        for qpos in qpos_trajectory:
            if qpos is None:
                continue
            if last_qpos is None or np.linalg.norm(qpos - last_qpos, ord=2) > 0.01:
                qpos_trajectory_dedup.append(qpos)
                last_qpos = qpos
        qpos_trajectory = np.array(qpos_trajectory_dedup)

        # spl = gen_cubic_spline(qpos_trajectory)
        spl = gen_linear_spline(qpos_trajectory)
        prev_qpos = None
        prev_qpos_not_moving = 0
        next_id = None

        world_states = []
        for _ in self.client.timeout(timeout):
            current_qpos = self.client.w.get_batched_qpos_by_id(self.panda, self.panda_joints)
            # next_target = get_next_target_cubic_spline(spl, current_qpos, step_size, qpos_trajectory)
            next_id, next_target = get_next_target_linear_spline(
                spl, current_qpos, step_size,
                minimum_x=next_id - step_size + 0.2 if next_id is not None else None
            )

            if verbose:
                print('this step size', np.linalg.norm(next_target - current_qpos, ord=1))
                print('next_id', next_id)
                print('this_target (lower, current, next, upper)\n', np.stack([
                    self.panda_joints_lower,
                    current_qpos,
                    next_target,
                    self.panda_joints_upper,
                ]), sep='')

            last_norm = np.linalg.norm(qpos_trajectory[-1] - current_qpos, ord=1)

            if verbose:
                print('last_norm', last_norm)
            if prev_qpos is not None:
                last_moving_dist = np.linalg.norm(prev_qpos - current_qpos, ord=1)
                if last_moving_dist < 0.001:
                    prev_qpos_not_moving += 1
                else:
                    prev_qpos_not_moving = 0
                if prev_qpos_not_moving > 10:
                    if last_norm < atol * 10:
                        if return_world_states:
                            return world_states
                        return True
                    else:
                        if not self.warnings_suppressed:
                            logger.warning(f'{self.body_name}: No progress for 10 steps.')
                        return False
            prev_qpos = current_qpos

            if last_norm < atol:
                if return_world_states:
                    return world_states
                return True

            vector_gains = np.ones(len(self.panda_joints)) * gains
            self.client.p.setJointMotorControlArray(
                bodyIndex=self.panda,
                jointIndices=self.panda_joints,
                controlMode=p.POSITION_CONTROL,
                targetPositions=next_target,
                positionGains=vector_gains,
            )
            self._set_gripper_control()
            self.client.step()
            world_states.append(self.world.save_world())

        if not self.warnings_suppressed:
            logger.warning(f'{self.body_name}: Moving timeout ({timeout}s).')
        return False

    def move_pose_smooth(self, pos: np.ndarray, quat: np.ndarray, speed: float = 0.01, max_qpos_distance: float = 1.0) -> bool:
        if self.ik_fast:
            # In ik_fast mode, we need to specify the `max_qpos_distance` to avoid the IK solution to be too far away.
            target_qpos = self.ik(pos, quat, max_distance=max_qpos_distance)
        else:
            target_qpos = self.ik(pos, quat)

        rv = self.move_qpos(target_qpos, speed=speed)
        return rv

    def move_home_cfree(self, speed: float = 0.01, timeout: float = 10.0):
        with self.client.disable_rendering(suppress_warnings=False):
            try:
                success, qpath = self.rrt_collision_free(self.get_home_qpos())
            except ValueError:
                success = False

        if not success:
            if not self.warnings_suppressed:
                logger.warning('Cannot find a collision-free path to home. Doing a direct move.')
            return self.move_qpos(self.get_home_qpos(), speed=speed, timeout=timeout)
        return self.move_qpos_trajectory(qpath, speed=speed, timeout=timeout)

    def internal_set_gripper_state(self, activate: bool, constraint_info: Optional[ConstraintInfo] = None) -> None:
        if not activate:
            if self.use_magic_gripper:
                if self.gripper_constraint is not None:
                    self.client.p.removeConstraint(self.gripper_constraint)
                    self.gripper_constraint = None
        else:
            if self.use_magic_gripper:
                if constraint_info is not None:
                    self.create_gripper_constraint(constraint_info.child_body)

        self.gripper_activated = activate

    def open_gripper_free(self, timeout: float = 2, force: bool = False) -> bool:
        return self._change_gripper_state_free(False, timeout=timeout, force=force)

    def close_gripper_free(self, timeout: float = 2, force: bool = False) -> bool:
        return self._change_gripper_state_free(True, timeout=timeout, force=force)

    @contextlib.contextmanager
    def set_next_pand_open(self, open_target: float):
        # TODO(Jiayuan Mao @ 2023/04/21): find a better way to do this.
        type(self).PANDA_GRIPPER_OPEN = open_target
        yield
        type(self).PANDA_GRIPPER_OPEN = 0.04

    def _change_gripper_state_free(self, activate: bool, timeout: float = 2.0, force: bool = False, verbose: bool = False) -> bool:
        """A helper function that change the gripper state assuming no contact with other objects.

        Args:
            activate (bool): True to activate the gripper, False to release it.

        Returns:
            bool: True if the gripper state changed, False otherwise.
        """

        if self.gripper_activated is not None and activate == self.gripper_activated and not force:
            return True

        if verbose:
            logger.info(f'{self.body_name}: Change gripper state to {activate}, use_magic_gripper={self.use_magic_gripper}, gripper_constraint={self.gripper_constraint}')

        if not activate:
            if self.use_magic_gripper:
                if self.gripper_constraint is not None:
                    self.client.p.removeConstraint(self.gripper_constraint)
                    self.gripper_constraint = None

        self.gripper_activated = activate
        target = type(self).PANDA_GRIPPER_CLOSE if activate else type(self).PANDA_GRIPPER_OPEN
        target_qpos = np.array([target, target])

        for _ in self.client.timeout(timeout):
            current_qpos = self.client.w.get_batched_qpos_by_id(self.panda, self.gripper_joints)
            diff = target_qpos - current_qpos

            if all(np.abs(diff) < 1e-3):
                return False

            self._set_gripper_control()
            self.client.step()

        if not self.warnings_suppressed:
            logger.warning(f'{self.body_name}: Moving gripper timeout ({timeout}s).')
        return True

    def _set_gripper_control(self, target: Optional[float] = None):
        if target is None:
            target = type(self).PANDA_GRIPPER_CLOSE if self.gripper_activated else type(self).PANDA_GRIPPER_OPEN

        for joint_index in self.gripper_joints:
            self.client.p.setJointMotorControl2(
                self.panda, joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target,
                force=40
            )

    def create_gripper_constraint(self, object_id, verbose: bool = False):
        if self.gripper_constraint is not None:
            self.client.p.removeConstraint(self.gripper_constraint)
            self.gripper_constraint = None

        if verbose:
            logger.info(f'{self.body_name}: Creating constraint between {self.panda} and {object_id}.')

        panda_leftfinger = self.world.link_names[f'{self.body_name}/panda_leftfinger'].link_id

        body_pose = self.p.getLinkState(self.panda, panda_leftfinger)  # the base link.
        obj_pose = self.p.getBasePositionAndOrientation(object_id)
        world_to_body = p.invertTransform(body_pose[0], body_pose[1])
        obj_to_body = p.multiplyTransforms(world_to_body[0], world_to_body[1], obj_pose[0], obj_pose[1])
        self.gripper_constraint = self.p.createConstraint(
            parentBodyUniqueId=self.panda,
            parentLinkIndex=panda_leftfinger,
            childBodyUniqueId=object_id,
            childLinkIndex=-1,  # should be the link index of the contact link.
            jointType=p.JOINT_FIXED,
            jointAxis=(0, 0, 0),
            parentFramePosition=obj_to_body[0],
            parentFrameOrientation=obj_to_body[1],
            childFramePosition=(0, 0, 0),
            childFrameOrientation=(0, 0, 0)
        )


class PandaGrasp(RobotActionPrimitive):
    robot: PandaRobot

    def __init__(self, robot: PandaRobot):
        super().__init__(robot)
        self.panda_leftfinger = self.w.link_names[f'{self.robot.body_name}/panda_leftfinger'].link_id
        self.panda_rightfinger = self.w.link_names[f'{self.robot.body_name}/panda_rightfinger'].link_id

    def __call__(self, timeout=2, target_object: Optional[int] = None, force_constraint: bool = False, verbose: bool = False) -> bool:
        target_qpos = np.array([type(self.robot).PANDA_GRIPPER_CLOSE] * 2)
        self.robot.gripper_activated = True

        for _ in self.client.timeout(timeout):
            current_qpos = self.w.get_batched_qpos_by_id(self.robot.panda, self.robot.gripper_joints)
            diff = target_qpos - current_qpos

            if self.detect_contact(verbose=verbose):
                return True

            if all(np.abs(diff) < 1e-3):
                return False

            for joint_index, target in zip(self.robot.gripper_joints, target_qpos):
                self.p.setJointMotorControl2(
                    self.robot.panda, joint_index,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=target,
                    force=40
                )
            # self.robot._set_gripper_control()
            self.client.step()

        if force_constraint:
            if target_object is not None and self.robot.use_magic_gripper and self.robot.gripper_constraint is None:
                self.robot.create_gripper_constraint(target_object)

        if not self.warnings_suppressed:
            logger.warning(f'{self.robot.body_name}: Grasping timeout ({timeout}s).')
        return False

    def detect_contact(self, verbose: bool = False):
        points1 = self.robot.client.p.getContactPoints(bodyA=self.robot.panda, linkIndexA=self.panda_leftfinger)
        points2 = self.robot.client.p.getContactPoints(bodyA=self.robot.panda, linkIndexA=self.panda_rightfinger)

        objects1 = {point[2] for point in points1 if point[2] != self.robot.panda}
        objects2 = {point[2] for point in points2 if point[2] != self.robot.panda}
        objects_inter = list(set.intersection(objects1, objects2))

        if len(objects_inter) > 0:
            if verbose:
                for obj_id in objects_inter:
                    logger.info(f'{self.robot.body_name}: Contact detected: {obj_id}, name = {self.w.body_names[obj_id]}.')

            if self.robot.use_magic_gripper:
                obj_id = objects_inter[0]
                self.robot.create_gripper_constraint(obj_id)

            return True
        return False


class PandaReachTwoStage(RobotActionPrimitive):
    robot: PandaRobot

    def __call__(self, pos: np.ndarray, quat: np.ndarray, height: float = 0.2, speed: float = 0.01) -> bool:
        logger.info(f'{self.robot.body_name}: Moving to pose (two-stage): {pos}, {quat}.')

        self.robot.move_pose(pos + rotate_vector([0, 0, -height], quat), quat, speed=speed)
        for i in reversed(range(10)):
            self.robot.move_pose(pos + rotate_vector([0, 0, -height / 10 * i], quat), quat, speed=speed)


class PandaPushTwoStage(RobotActionPrimitive):
    robot: PandaRobot

    def __call__(self, start_pos: np.ndarray, normal: np.ndarray, distance: float, timeout: float = 10, prepush_height: float = 0.1, speed: float = 0.01) -> bool:
        self.robot.close_gripper_free()

        pos, quat = self.robot.get_ee_pose()
        self.robot.move_pose(start_pos - prepush_height * normal / np.linalg.norm(normal), quat, speed=speed)

        pos, quat = self.robot.get_ee_pose()
        end_pos = start_pos + normal * distance

        for _ in self.client.timeout(timeout):
            diff = end_pos - pos
            if np.linalg.norm(diff) < 0.01:
                return True
            self.robot.move_pose_smooth(pos + diff / np.linalg.norm(diff) * speed, quat, speed=speed)
            pos, _ = self.robot.get_ee_pose()

        if not self.warnings_suppressed:
            logger.info(f'{self.robot.body_name}: Pushing timeout ({timeout}s).')
        return False


class PandaPlanarPush(RobotActionPrimitive):
    robot: PandaRobot

    def __call__(self, target_pos: np.ndarray, timeout: float = 10, speed: float = 0.01, tol: float = 0.01) -> bool:
        target_pos = np.array(target_pos)
        pos, quat = self.robot.get_ee_pose()
        for _ in self.client.timeout(timeout):
            diff = target_pos - pos

            if np.linalg.norm(diff) < tol:
                return True

            self.robot.move_pose_smooth(pos + diff / np.linalg.norm(diff) * speed, quat, speed=speed)
            pos, _ = self.robot.get_ee_pose()

        if not self.warnings_suppressed:
                logger.warning(f'{self.robot.body_name}: Planar push timeout ({timeout}s).')
        return False


class PandaPickAndPlace(RobotActionPrimitive):
    """A helpful pick-and-place primitive."""

    robot: PandaRobot

    def __call__(
        self,
        pick_pos: np.ndarray, pick_quat: np.ndarray,
        place_pos: np.ndarray, place_quat: np.ndarray,
        reach_kwargs: Optional[Dict[str, Any]] = None, grasp_kwargs: Optional[Dict[str, Any]] = None
    ) -> bool:
        pick_pos = np.array(pick_pos)
        pick_quat = np.array(pick_quat)
        pick_after_pos = pick_pos + np.array([0, 0, 0.2])
        place_pos = np.array(place_pos)
        place_quat = np.array(place_quat)
        place_after_pos = place_pos + np.array([0, 0, 0.2])

        if reach_kwargs is None:
            reach_kwargs = {}
        if grasp_kwargs is None:
            grasp_kwargs = {}

        self.robot.do('open_gripper_free')
        self.robot.do('reach_two_stage', pick_pos, pick_quat, **reach_kwargs)
        self.robot.do('grasp', **grasp_kwargs)
        self.robot.do('move_pose', pick_after_pos, pick_quat)
        self.robot.do('move_pose', place_pos, place_quat)
        self.robot.do('open_gripper_free')
        self.robot.do('move_pose', place_after_pos, place_quat)
