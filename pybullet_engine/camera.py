#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : camera.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 08/25/2022
#
# This file is part of HACL-PyTorch.
# Distributed under terms of the MIT license.

import math
import collections
import numpy as np
import pybullet as p

from typing import Optional, Tuple
from .rotation_utils import rpy


class CameraConfig(collections.namedtuple('_CameraConfig', ['image_size', 'intrinsics', 'position', 'rotation', 'zrange', 'shadow'])):
    """
    Mostly based on https://github.com/cliport/cliport/blob/e9cde74754606448d8a0495c1efea36c29b201f1/cliport/tasks/cameras.py
    """

    def get_view_and_projection_matricies(self, image_size: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        if image_size is None:
            image_size = self.image_size

        view_matrix = self.get_view_matrix()
        projection_matrix = self.get_projection_matrix(image_size)
        return view_matrix, projection_matrix

    def get_view_matrix(self) -> np.ndarray:
        lookdir = np.float32([0, 0, 1]).reshape(3, 1)
        updir = np.float32([0, -1, 0]).reshape(3, 1)
        rotation = p.getMatrixFromQuaternion(rpy(*self.rotation))
        rotm = np.float32(rotation).reshape(3, 3)
        lookdir = (rotm @ lookdir).reshape(-1)
        updir = (rotm @ updir).reshape(-1)
        lookat = self.position + lookdir
        return p.computeViewMatrix(self.position, lookat, updir)

    def get_projection_matrix(self, image_size: Tuple[int, int]) -> np.ndarray:
        focal_len = self.intrinsics[0]
        znear, zfar = self.zrange
        fovh = (image_size[0] / 2) / focal_len
        fovh = 180 * np.arctan(fovh) * 2 / np.pi        # Notes: 1) FOV is vertical FOV 2) aspect must be float
        aspect_ratio = float(image_size[1]) / image_size[0]
        return p.computeProjectionMatrixFOV(fovh, aspect_ratio, znear, zfar)


class RealSenseD415(object):
    """Default configuration with 3 RealSense RGB-D cameras."""

    # Mimic RealSense D415 RGB-D camera parameters.
    image_size = (480, 640)
    intrinsics = (450., 0, 320., 0, 450., 240., 0, 0, 1)

    # Set default camera poses.
    front_position = (1., 0, 0.75)
    front_rotation = np.rad2deg((np.pi / 4, np.pi, -np.pi / 2))
    left_position = (0, 0.5, 0.75)
    left_rotation = np.rad2deg((np.pi / 4.5, np.pi, np.pi / 4))
    right_position = (0, -0.5, 0.75)
    right_rotation = np.rad2deg((np.pi / 4.5, np.pi, 3 * np.pi / 4))

    @classmethod
    def get_configs(cls):
        return [
            CameraConfig(cls.image_size, cls.intrinsics, cls.front_position, cls.front_rotation, (0.1, 10.), True),
            CameraConfig(cls.image_size, cls.intrinsics, cls.left_position, cls.left_rotation, (0.1, 10.), True),
            CameraConfig(cls.image_size, cls.intrinsics, cls.right_position, cls.right_rotation, (0.1, 10.), True)
        ]


class TopDownOracle(object):
    """Top-down noiseless image used only by the oracle demonstrator."""

    # Near-orthographic projection.
    image_size = (480, 640)
    intrinsics = (63e4, 0, 320., 0, 63e4, 240., 0, 0, 1)
    position = (0.5, 0, 1000.)
    rotation = np.rad2deg((0, np.pi, -np.pi / 2))

    @classmethod
    def get_configs(cls):
        return [CameraConfig(cls.image_size, cls.intrinsics, cls.position, cls.rotation, (999.7, 1001.), False)]


class RS200Gazebo(object):
    """Gazebo Camera"""

    # Near-orthographic projection.
    image_size = (480, 640)
    intrinsics = (554.3826904296875, 0.0, 320.0, 0.0, 554.3826904296875, 240.0, 0.0, 0.0, 1.0)
    position = (0.5, 0, 1.0)
    rotation = np.rad2deg((0, np.pi, np.pi / 2))

    @classmethod
    def get_configs(cls):
        return [CameraConfig(cls.image_size, cls.intrinsics, cls.position, cls.rotation, (0.1, 10.), False)]


class KinectFranka(object):
    """Kinect Franka Camera"""

    # Near-orthographic projection.
    image_size = (424, 512)
    intrinsics = (365.57489013671875, 0.0, 257.5205078125, 0.0, 365.57489013671875, 205.26710510253906, 0.0, 0.0, 1.0)
    position = (1.082, -0.041, 1.027)
    rotation = np.rad2deg((-2.611, 0.010, 1.553))

    @classmethod
    def get_configs(cls):
        return [CameraConfig(cls.image_size, cls.intrinsics, cls.position, cls.rotation, (0.1, 10.), True)]


class Cliport(object):
    """Cliport camera configuration."""

    image_size = (480, 640)
    intrinsics = (450., 0, 320., 0, 450., 240., 0, 0, 1)
    position1 = (1.5, 0, 0.8)
    rotation1 = (0, -115, 90)
    position2 = (0, -1, 0.8)
    rotation2 = (-120, 20, 10)
    position3 = (0, 1, 0.8)
    rotation3 = (120, 20, 170)

    @classmethod
    def get_configs(cls):
        return [
            CameraConfig(cls.image_size, cls.intrinsics, cls.position1, cls.rotation1, (0.1, 10.), False),
            CameraConfig(cls.image_size, cls.intrinsics, cls.position2, cls.rotation2, (0.1, 10.), False),
            CameraConfig(cls.image_size, cls.intrinsics, cls.position3, cls.rotation3, (0.1, 10.), False)
        ]


def get_point_cloud(config, color, depth, segmentation=None):
    """Reconstruct point clouds from the color and depth images."""
    # based on https://stackoverflow.com/questions/59128880/getting-world-coordinates-from-opengl-depth-buffer
    height, width = depth.shape[0:2]
    view_matrix, proj_matrix = config.get_view_and_projection_matricies()

    # create a 4x4 transform matrix that goes from pixel coordinates (and depth values) to world coordinates
    proj_matrix = np.asarray(proj_matrix).reshape([4, 4], order="F")
    view_matrix = np.asarray(view_matrix).reshape([4, 4], order="F")
    tran_pix_world = np.linalg.inv(np.matmul(proj_matrix, view_matrix))

    # create a grid with pixel coordinates and depth values
    y, x = np.mgrid[-1:1:2 / height, -1:1:2 / width]
    y *= -1.
    x, y, z = x.reshape(-1), y.reshape(-1), depth.reshape(-1)
    h = np.ones_like(z)

    pixels = np.stack([x, y, z, h], axis=1)
    mask = z < 0.99
    # mask = ...
    # filter out "infinite" depths
    pixels = pixels[mask]
    pixels[:, 2] = 2 * pixels[:, 2] - 1

    # turn pixels to world coordinates
    points = np.matmul(tran_pix_world, pixels.T).T
    points /= points[:, 3: 4]
    points = points[:, :3]

    if segmentation is not None:
        return points, color.reshape(-1, color.shape[-1])[mask], segmentation.reshape(-1)[mask]
    return points, color.reshape(-1, color.shape[-1])[mask]


def lookat_rpy(camera_pos: np.ndarray, target_pos: np.ndarray, roll: float = 0) -> np.ndarray:
    """Construct the roll, pitch, yaw angles of a camera looking at a target.
    This function assumes that the camera is pointing to the z-axis ([0, 0, 1]),
    in the camera frame.
    Args:
        camera_pos: the position of the camera.
        target_pos: the target position.
        roll: the roll angle of the camera.

    Returns:
        a numpy array of the roll, pitch, yaw angles in degrees.
    """
    camera_pos = np.asarray(camera_pos)
    target_pos = np.asarray(target_pos)

    delta = target_pos - camera_pos
    pitch = math.atan2(-np.linalg.norm(delta[:2]), delta[2])
    yaw = math.atan2(-delta[1], -delta[0])
    rv = np.array([np.deg2rad(roll), pitch, yaw], dtype="float32")

    return np.rad2deg(rv)

