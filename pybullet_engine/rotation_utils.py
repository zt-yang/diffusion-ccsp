#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : rotation_utils.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/18/2020
#
# This file is part of HACL-PyTorch.
# Distributed under terms of the MIT license.

import numpy as np
from typing import Tuple, TYPE_CHECKING
from pybullet_engine.rotationlib import euler2quat

if TYPE_CHECKING:
    from pybullet_engine.client import BulletClient


# For testing whether a number is close to zero
_FLOAT_EPS = np.finfo(np.float64).eps
_EPS4 = _FLOAT_EPS * 4.0


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Normalize a vector."""
    return v / np.linalg.norm(v)


def find_orthogonal_vector(v: np.ndarray) -> np.ndarray:
    """Find an orthogonal vector to the given vector.

    The returned vector is guaranteed to be normalized.
    """
    v = v / np.linalg.norm(v, axis=-1, keepdims=True)

    mask = np.abs(v[..., 0]) < 0.5
    return np.cross(v, np.array([1, 0, 0])) * mask + np.cross(v, np.array([0, 1, 0])) * (1 - mask)


def getQuaternionFromMatrix(mat):
    """ Convert Rotation Matrix to Quaternion.  See rotation.py for notes """
    mat = np.asarray(mat, dtype=np.float64)
    assert mat.shape[-2:] == (3, 3), "Invalid shape matrix {}".format(mat)

    Qxx, Qyx, Qzx = mat[..., 0, 0], mat[..., 0, 1], mat[..., 0, 2]
    Qxy, Qyy, Qzy = mat[..., 1, 0], mat[..., 1, 1], mat[..., 1, 2]
    Qxz, Qyz, Qzz = mat[..., 2, 0], mat[..., 2, 1], mat[..., 2, 2]
    # Fill only lower half of symmetric matrix
    K = np.zeros(mat.shape[:-2] + (4, 4), dtype=np.float64)
    K[..., 0, 0] = Qxx - Qyy - Qzz
    K[..., 1, 0] = Qyx + Qxy
    K[..., 1, 1] = Qyy - Qxx - Qzz
    K[..., 2, 0] = Qzx + Qxz
    K[..., 2, 1] = Qzy + Qyz
    K[..., 2, 2] = Qzz - Qxx - Qyy
    K[..., 3, 0] = Qyz - Qzy
    K[..., 3, 1] = Qzx - Qxz
    K[..., 3, 2] = Qxy - Qyx
    K[..., 3, 3] = Qxx + Qyy + Qzz
    K /= 3.0
    # TODO: vectorize this -- probably could be made faster
    q = np.empty(K.shape[:-2] + (4,))
    it = np.nditer(q[..., 0], flags=['multi_index'])
    while not it.finished:
        # Use Hermitian eigenvectors, values for speed
        vals, vecs = np.linalg.eigh(K[it.multi_index])
        # Select largest eigenvector, reorder to w,x,y,z quaternion
        q[it.multi_index] = vecs[:, np.argmax(vals)]
        # Prefer quaternion with positive w
        # (q * -1 corresponds to same rotation as q)
        if q[it.multi_index][3] < 0:
            q[it.multi_index] *= -1
        it.iternext()
    return q


def get_quaternion_from_matrix(mat):
    """Convert a rotation matrix to a quaternion.

    The quaternion is represented as a 4-element array in the order of (x, y, z, w),
    which follows the convention of pybullet.
    """
    mat = np.asarray(mat, dtype=np.float64)
    assert mat.shape[-2:] == (3, 3), "Invalid shape matrix {}".format(mat)
    return getQuaternionFromMatrix(mat)


def get_quaternion_from_axes(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Converts a rotation matrix to a quaternion."""
    m = np.stack([x, y, z], axis=1)
    return get_quaternion_from_matrix(m)


def compose_transformation(pos1, quat1, pos2, quat2):
    """Compsoe two transformations.

    The transformations are represented as a tuple of position and quaternion.
    The quaternion is represented as a 4-element array in the order of (x, y, z, w), which follows the convention of pybullet.
    """

    pos1 = np.asarray(pos1, dtype=np.float32)
    quat1 = np.asarray(quat1, dtype=np.float32)
    pos2 = np.asarray(pos2, dtype=np.float32)
    quat2 = np.asarray(quat2, dtype=np.float32)

    return pos1 + rotate_vector(pos2, quat1), quat_mul(quat1, quat2)


def inverse_transformation(pos, quat):
    """Inverse a transformation.

    The transformation is represented as a tuple of position and quaternion.
    The quaternion is represented as a 4-element array in the order of (x, y, z, w), which follows the convention of pybullet.
    """
    pos = np.asarray(pos, dtype=np.float32)
    quat = np.asarray(quat, dtype=np.float32)

    inv_pos = -pos
    inv_quat = quat_conjugate(quat)
    return inv_pos, inv_quat


def get_transform_a_to_b(pos1, quat1, pos2, quat2):
    """Get the transformation from frame A to frame B.

    The transformations are represented as a tuple of position and quaternion.
    The quaternion is represented as a 4-element array in the order of (x, y, z, w), which follows the convention of pybullet.
    """
    pos1 = np.asarray(pos1, dtype=np.float32)
    quat1 = np.asarray(quat1, dtype=np.float32)
    pos2 = np.asarray(pos2, dtype=np.float32)
    quat2 = np.asarray(quat2, dtype=np.float32)

    inv_quat1 = quat_conjugate(quat1)
    a_to_b_pos = rotate_vector(pos2 - pos1, inv_quat1)
    a_to_b_quat = quat_mul(inv_quat1, quat2)
    return a_to_b_pos, a_to_b_quat


def frame_mul(pos_a, quat_a, a_to_b):
    """Multiply a frame with a transformation.

    The frame is represented as a tuple of position and quaternion.
    The transformation is represented as a tuple of position and quaternion.
    The quaternion is represented as a 4-element array in the order of (x, y, z, w), which follows the convention of pybullet.
    """
    pos_a = np.asarray(pos_a, dtype=np.float32)
    quat_a = np.asarray(quat_a, dtype=np.float32)
    transform_pos = np.asarray(a_to_b[0], dtype=np.float32)
    transform_quat = np.asarray(a_to_b[1], dtype=np.float32)

    pos_b = pos_a + rotate_vector(transform_pos, quat_a)
    quat_b = quat_mul(quat_a, transform_quat)
    return pos_b, quat_b


def frame_inv(pos_b, quat_b, a_to_b):
    """Inverse a frame with a transformation.

    The frame is represented as a tuple of position and quaternion.
    The transformation is represented as a tuple of position and quaternion.
    The quaternion is represented as a 4-element array in the order of (x, y, z, w), which follows the convention of pybullet.
    """
    pos_b = np.asarray(pos_b, dtype=np.float32)
    quat_b = np.asarray(quat_b, dtype=np.float32)
    transform_pos = np.asarray(a_to_b[0], dtype=np.float32)
    transform_quat = np.asarray(a_to_b[1], dtype=np.float32)

    inv_transform_quat = quat_conjugate(transform_quat)
    quat_a = quat_mul(quat_b, inv_transform_quat)
    pos_a = pos_b - rotate_vector(transform_pos, quat_a)
    return pos_a, quat_a


def quat_mul(q0, q1, *args):
    if len(args) > 0:
        return quat_mul(q0, quat_mul(q1, *args))

    q0 = np.asarray(q0)
    q1 = np.asarray(q1)
    assert q0.shape == q1.shape
    assert q0.shape[-1] == 4
    assert q1.shape[-1] == 4

    w0 = q0[..., 3]
    x0 = q0[..., 0]
    y0 = q0[..., 1]
    z0 = q0[..., 2]

    w1 = q1[..., 3]
    x1 = q1[..., 0]
    y1 = q1[..., 1]
    z1 = q1[..., 2]

    w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
    x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    y = w0 * y1 + y0 * w1 + z0 * x1 - x0 * z1
    z = w0 * z1 + z0 * w1 + x0 * y1 - y0 * x1
    q = np.array([x, y, z, w])
    if q.ndim == 2:
        q = q.swapaxes(0, 1)
    assert q.shape == q0.shape

    if w0 > 0:
        return q
    else:
        return -q


def quat_conjugate(q):
    q = np.asarray(q)
    assert q.shape[-1] == 4
    q = np.array([-q[..., 0], -q[..., 1], -q[..., 2], q[..., 3]])
    return q


def rpy(r, p, y, degree=True):
    """Create a quaternion from euler angles."""
    if degree:
        r = np.deg2rad(r)
        p = np.deg2rad(p)
        y = np.deg2rad(y)
    quat_wxyz = euler2quat((r, p, y))
    return quat_wxyz[..., [1, 2, 3, 0]]


def rotate_vector(vec, quat):
    """Rotate a vector by a quaternion."""
    vec = np.asarray(vec)
    quat = np.asarray(quat)
    u = quat[:3]
    s = quat[3]
    return 2.0 * np.dot(u, vec) * u + (s * s - np.dot(u, u)) * vec + 2.0 * s * np.cross(u, vec)


def rotate_vector_batch(batch_vector, quat):
    """Rotate a batch of vectors by a quaternion."""
    vec = np.asarray(batch_vector)
    quat = np.asarray(quat)

    u = quat[:3]
    s = quat[3]

    return 2.0 * np.dot(vec, u)[..., np.newaxis] * u + (s * s - np.dot(u, u)) * vec + 2.0 * s * np.cross(u, vec)


def quaternion_from_vectors(vec1, vec2):
    """Create a rotation quaternion q such that q * vec1 = vec2."""
    vec1 = np.asarray(vec1)
    vec2 = np.asarray(vec2)
    assert vec1.shape == vec2.shape, f'Vector shapes do not match: {vec1.shape} != {vec2.shape}'
    assert vec1.shape[-1] == 3, f'Vector shape must be 3, got {vec1.shape[-1]}'
    assert vec2.shape[-1] == 3, f'Vector shape must be 3, got {vec2.shape[-1]}'

    vec1 = vec1 / np.linalg.norm(vec1, axis=-1, keepdims=True)
    vec2 = vec2 / np.linalg.norm(vec2, axis=-1, keepdims=True)

    u = np.cross(vec1, vec2)
    s = np.dot(vec1, vec2)

    if np.linalg.norm(u) < 1e-6:
        return np.array([0, 0, 0, 1])

    opposite_pairs = (s < -1 + 1e-6)
    u = u * (1 - opposite_pairs) + opposite_pairs * find_orthogonal_vector(u)

    if len(u.shape) == 1:
        s = np.array([s], dtype=u.dtype)
    q = np.concatenate([u, s + 1], axis=-1)

    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    return q


def sample_quaternion_from_vectors(input_normal, target_normal, nr_samples: int = 4):
    base_quat = quaternion_from_vectors(input_normal, target_normal)

    yaw_quat = quaternion_from_vectors(target_normal, np.array([0, 0, 1]))
    for yaw in np.arange(0, 2 * np.pi, 2 * np.pi / nr_samples):
        quat = quat_mul(
            quat_conjugate(yaw_quat),
            rpy(0, 0, yaw, degree=False),
            yaw_quat,
            base_quat
        )
        yield quat


def patch_pybullet():
    import pybullet as p

    p.getQuaternionFromMatrix = getQuaternionFromMatrix


def get_ee_to_tool(client: 'BulletClient', robot_id, ee_id, tool_id: int) -> Tuple[np.ndarray, np.ndarray]:
    """Get the transformation from the end-effector to the tool that the robot is currently holding."""

    # if robot.gripper_constraint is not None:
    #     constraint = robot.world.get_constraint(robot.gripper_constraint)
    #     assert constraint.child_body == tool_id
    #     from pybullet_engine.rotation_utils import quat_conjugate
    #     return -np.array(constraint.parent_frame_pos), quat_conjugate(np.array(constraint.parent_frame_orn))

    robot_pos, robot_quat = client.world.get_link_state_by_id(robot_id, ee_id, fk=True).get_transformation()
    tool_pos, tool_quat = client.world.get_body_state_by_id(tool_id).get_transformation()
    return get_transform_a_to_b(robot_pos, robot_quat, tool_pos, tool_quat)


def solve_tool_from_ee(ee_pos, ee_quat, ee_to_tool):
    return frame_mul(ee_pos, ee_quat, ee_to_tool)


def solve_ee_from_tool(target_tool_pos, target_tool_quat, ee_to_tool):
    """Solve for the end-effector position and orientation given the tool position and orientation."""
    return frame_inv(target_tool_pos, target_tool_quat, ee_to_tool)

