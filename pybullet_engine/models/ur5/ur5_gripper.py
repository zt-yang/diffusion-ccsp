#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : ur5_gripper.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 08/08/2022
#
# This file is part of HACL-PyTorch.
# Distributed under terms of the MIT license.

"""
Classes to handle gripper dynamics.
This file is directly copied from https://github.com/cliport/cliport/blob/master/cliport/tasks/grippers.py
"""

import os.path as osp
import numpy as np
import pybullet as p

from jacinle.utils.enum import JacEnum
from pybullet_engine.client import BulletClient
from pybullet_engine.models.robot import Gripper, GripperObjectIndices
from pybullet_engine.rotation_utils import get_ee_to_tool, solve_tool_from_ee


SPATULA_BASE_URDF = 'ur5/spatula/spatula-base.urdf'
SUCTION_BASE_URDF = 'ur5/suction/suction-base.urdf'
SUCTION_HEAD_URDF = 'ur5/suction/suction-head.urdf'


class UR5GripperType(JacEnum):
    NONE = 'none'
    SPATULA = 'spatula'
    SUCTION = 'suction'

    def get_class(self):
        if self is UR5GripperType.NONE:
            return None
        elif self is UR5GripperType.SPATULA:
            return Spatula
        elif self is UR5GripperType.SUCTION:
            return Suction
        else:
            raise ValueError(f'Unknown gripper type: {self}.')


class Spatula(Gripper):
    """Simulate simple spatula for pushing."""

    def __init__(self, client: BulletClient, robot: int, ee: int, obj_ids: GripperObjectIndices):  # pylint: disable=unused-argument
        """Creates spatula and 'attaches' it to the robot."""
        super().__init__(client)

        # Load spatula model.
        pose = ((0.487, 0.109, 0.438), p.getQuaternionFromEuler((np.pi, 0, 0)))
        base = self.client.load_urdf(osp.join(client.assets_root, SPATULA_BASE_URDF), pose[0], pose[1])
        self.p.createConstraint(
            parentBodyUniqueId=robot,
            parentLinkIndex=ee,
            childBodyUniqueId=base,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=(0, 0, 0),
            parentFramePosition=(0, 0, 0),
            childFramePosition=(0, 0, 0.01)
        )


class Suction(Gripper):
    """Simulate simple suction dynamics."""

    def __init__(self, client, robot, ee, obj_ids):
        """Creates suction and 'attaches' it to the robot.

        Has special cases when dealing with rigid vs deformables. For rigid,
        only need to check contact_constraint for any constraint. For soft
        bodies (i.e., cloth or bags), use cloth_threshold to check distances
        from gripper body (self.head_id) to any vertex in the cloth mesh. We
        need correct code logic to handle gripping potentially a rigid or a
        deformable (and similarly for releasing).

        To be clear on terminology: 'deformable' here should be interpreted
        as a PyBullet 'softBody', which includes cloths and bags. There's
        also cables, but those are formed by connecting rigid body beads, so
        they can use standard 'rigid body' grasping code.

        To get the suction gripper pose, use p.getLinkState(self.head_id, 0),
        and not p.getBasePositionAndOrientation(self.head_id) as the latter is
        about z=0.03m higher and empirically seems worse.

        Args:
          client: the BulletClient instance.
          robot: int representing PyBullet ID of robot.
          ee: int representing PyBullet ID of end effector link.
          obj_ids: list of PyBullet IDs of all suctionable objects in the env.
        """
        super().__init__(client)

        self.robot_id = robot
        self.ee_id = ee

        # Load suction gripper base model (visual only).
        pose = ((0.487, 0.109, 0.438), p.getQuaternionFromEuler((np.pi, 0, 0)))
        self.base_id = self.client.load_urdf(osp.join(client.assets_root, SUCTION_BASE_URDF), pose[0], pose[1])
        self.suction_base_constraint = self.p.createConstraint(
            parentBodyUniqueId=robot,
            parentLinkIndex=ee,
            childBodyUniqueId=self.base_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=(0, 0, 0),
            parentFramePosition=(0, 0, 0),
            childFramePosition=(0, 0, 0.01)
        )

        # Load suction tip model (visual and collision) with compliance.
        # urdf = 'assets/ur5/suction/suction-head.urdf'
        pose = ((0.487, 0.109, 0.347), p.getQuaternionFromEuler((np.pi, 0, 0)))
        self.head_id = self.client.load_urdf(osp.join(client.assets_root, SUCTION_HEAD_URDF), pose[0], pose[1])
        self.suction_head_constraint = self.p.createConstraint(
            parentBodyUniqueId=robot,
            parentLinkIndex=ee,
            childBodyUniqueId=self.head_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=(0, 0, 0),
            parentFramePosition=(0, 0, 0),
            childFramePosition=(0, 0, -0.08)
        )
        self.p.changeConstraint(self.suction_head_constraint, maxForce=50)

        # Reference to object IDs in environment for simulating suction.
        self.obj_ids = obj_ids

        # Indicates whether gripper is gripping anything (rigid or def).
        self.activated = False

        # For gripping and releasing rigid objects.
        self.contact_constraint = None
        self.contact_ee_to_obj = None

        # Defaults for deformable parameters, and can override in tasks.
        self.def_ignore = 0.035  # TODO(daniel) check if this is needed
        self.def_threshold = 0.030
        self.def_nb_anchors = 1

        # Track which deformable is being gripped (if any), and anchors.
        self.def_grip_item = None
        self.def_grip_anchors = []

        # Determines release when gripped deformable touches a rigid/def.
        # TODO(daniel) should check if the code uses this -- not sure?
        self.def_min_vetex = None
        self.def_min_distance = None

        # Determines release when a gripped rigid touches defs (e.g. cloth-cover).
        self.init_grip_distance = None
        self.init_grip_item = None

    def activate(self):
        """Simulate suction using a rigid fixed constraint to contacted object."""
        # TODO(andyzeng): check deformables logic.
        # del def_ids

        if not self.activated:
            points = self.p.getContactPoints(bodyA=self.head_id, linkIndexA=0)
            if points:
                # Handle contact between suction with a rigid object.
                for point in points:
                    obj_id, contact_link = point[2], point[4]
                if obj_id in self.obj_ids['rigid']:
                    body_pose = self.p.getLinkState(self.head_id, 0)  # the base link.
                    obj_pose = self.p.getBasePositionAndOrientation(obj_id)
                    world_to_body = p.invertTransform(body_pose[0], body_pose[1])
                    obj_to_body = p.multiplyTransforms(world_to_body[0], world_to_body[1], obj_pose[0], obj_pose[1])
                    self.contact_constraint = self.p.createConstraint(
                        parentBodyUniqueId=self.head_id,
                        parentLinkIndex=0,
                        childBodyUniqueId=obj_id,
                        childLinkIndex=contact_link,
                        jointType=p.JOINT_FIXED,
                        jointAxis=(0, 0, 0),
                        parentFramePosition=obj_to_body[0],
                        parentFrameOrientation=obj_to_body[1],
                        childFramePosition=(0, 0, 0),
                        childFrameOrientation=(0, 0, 0)
                    )
                    self.contact_ee_to_object = get_ee_to_tool(self.client, self.head_id, 0, obj_id)

                self.activated = True

    def sync_gripper_qpos(self):
        """Sync gripper qpos with PyBullet simulation."""

        # recompute the pose for the suction body
        link_state = self.client.world.get_link_state_by_id(self.robot_id, self.ee_id)
        pos, quat = link_state.pos, link_state.orientation

        # update the suction base
        self.world.set_body_state2_by_id(self.base_id, pos + np.array((0, 0, 0.01)), quat)

        # update the suction head
        self.world.set_body_state2_by_id(self.head_id, pos + np.array((0, 0, -0.08)), quat)

        # update the object being gripped
        if self.contact_constraint is not None:
            constraint = self.world.get_constraint(self.contact_constraint)
            pos, quat = self.world.get_link_state_by_id(self.head_id, 0).get_transformation()
            new_pos, new_quat = solve_tool_from_ee(pos, quat, self.contact_ee_to_object)
            self.world.set_body_state2_by_id(constraint.child_body, new_pos, new_quat)
        self.world.update_contact()

    def release(self):
        """Release gripper object, only applied if gripper is 'activated'.

        If suction off, detect contact between gripper and objects.
        If suction on, detect contact between picked object and other objects.

        To handle deformables, simply remove constraints (i.e., anchors).
        Also reset any relevant variables, e.g., if releasing a rigid, we
        should reset init_grip values back to None, which will be re-assigned
        in any subsequent grasps.
        """
        if self.activated:
            self.activated = False
            print('Releasing gripper.')

            # Release gripped rigid object (if any).
            if self.contact_constraint is not None:
                print('Releasing gripped rigid object.')
                try:
                    self.p.removeConstraint(self.contact_constraint)
                    self.contact_constraint = None
                except:  # noqa: E772
                    print('Failed to release gripped rigid object.')
                    pass
                else:
                    print('Released gripped rigid object.')
                self.init_grip_distance = None
                self.init_grip_item = None
            self.contact_ee_to_object = None

            # Release gripped deformable object (if any).
            if self.def_grip_anchors:
                for anchor_id in self.def_grip_anchors:
                    self.p.removeConstraint(anchor_id)
                self.def_grip_anchors = []
                self.def_grip_item = None
                self.def_min_vetex = None
                self.def_min_distance = None

    def detect_contact(self):
        """Detects a contact with a rigid object."""
        body, link = self.head_id, 0
        if self.activated and self.contact_constraint is not None:
            try:
                info = self.p.getConstraintInfo(self.contact_constraint)
                body, link = info[2], info[3]
            except:  # noqa: E772
                self.contact_constraint = None
                pass

        # Get all contact points between the suction and a rigid body.
        points = self.p.getContactPoints(bodyA=body, linkIndexA=link)
        if self.activated:
            points = [point for point in points if point[2] != self.head_id]
        # print(body, link, [self.client.w.body_names[point[2]] for point in points])

        # We know if len(points) > 0, contact is made with SOME rigid item.
        if points:
            return True

        return False

    def check_grasp(self):
        """Check a grasp (object in contact?) for picking success."""

        suctioned_object = None
        if self.contact_constraint is not None:
            suctioned_object = self.p.getConstraintInfo(self.contact_constraint)[2]
        return suctioned_object is not None

