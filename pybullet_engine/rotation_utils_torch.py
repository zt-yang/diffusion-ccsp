#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : rotation_utils_torch.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 09/22/2022
#
# This file is part of HACL-PyTorch.
# Distributed under terms of the MIT license.

import torch

__all__ = ['th_quat_mul', 'th_quat_conjugate', 'th_compose_transformation', 'th_inverse_transformation']


def th_quat_mul(quat1: torch.Tensor, quat2: torch.Tensor, *args: torch.Tensor) -> torch.Tensor:
    """Multiply two quaternions. The quaternions are represented as 4D vectors,
    in the form of [x, y, z, w].

    Args:
        quat1: a tensor of shape (..., 4).
        quat2: a tensor of shape (..., 4).
        *args: additional tensors of shape (..., 4).

    Returns:
        A tensor of shape (..., 4).
    """

    if len(args) > 0:
        return th_quat_mul(quat1, th_quat_mul(quat2, *args))

    assert quat1.shape == quat2.shape and quat1.shape[-1] == 4 and quat2.shape[-1] == 4

    x1, y1, z1, w1 = quat1[..., 0], quat1[..., 1], quat1[..., 2], quat1[..., 3]
    x2, y2, z2, w2 = quat2[..., 0], quat2[..., 1], quat2[..., 2], quat2[..., 3]

    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

    q = torch.stack([x, y, z, w], dim=-1)
    sign = -1 + 2 * (w > 0)

    return q * sign


def th_quat_conjugate(quat: torch.Tensor) -> torch.Tensor:
    """Compute the conjugate of a quaternion. The quaternion is represented as a 4D vector,
    in the form of [x, y, z, w].

    Args:
        quat: the quaternion, as a 4D vector.

    Returns:
        the conjugate, as a 4D vector.
    """

    return torch.stack([-quat[..., 0], -quat[..., 1], -quat[..., 2], quat[..., 3]], dim=-1)


def th_compose_transformation(pose1: torch.Tensor, pose2: torch.Tensor, *args: torch.Tensor) -> torch.Tensor:
    """Compose two transformations.

    Args:
        pose1: the first transformation, as a 7D vector.
        pose2: the second transformation, as a 7D vector.
        *args: additional transformations, as 7D vectors.

    Returns:
        the composed transformation, as a 7D vector.
    """

    if len(args) > 0:
        return th_compose_transformation(pose1, th_compose_transformation(pose2, *args))

    pos1, quat1 = pose1[..., :3], pose1[..., 3:]
    pos2, quat2 = pose2[..., :3], pose2[..., 3:]

    pos = pos1 + th_rotate_vector(pos2, quat1)
    quat = th_quat_mul(quat1, quat2)
    return torch.cat([pos, quat], dim=-1)


def th_inverse_transformation(pose1: torch.Tensor) -> torch.Tensor:
    """Compute the inverse of a transformation.

    Args:
        pose1 the transformation, as a 7D vector.

    Returns:
        the inverse transformation, as a 7D vector.
    """

    pos1, quat1 = pose1[..., :3], pose1[..., 3:]
    quat = th_quat_conjugate(quat1)
    return torch.cat([-pos1, quat], dim=-1)


def th_rotate_vector(vec: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    """Rotate a vector by a quaternion."""
    u = quat[..., :3]
    s = quat[..., 3]
    return 2.0 * torch.einsum('...i,...i->...', u, vec) * u + (s * s - torch.einsum('...i,...i->...', u, u)) * vec + 2.0 * s * torch.cross(u, vec)
