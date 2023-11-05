#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : cspace.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/17/2020
#
# This file is part of HACL-PyTorch.
# Distributed under terms of the MIT license.

import numpy as np
import contextlib
from typing import List
from .client import BulletClient

__all__ = ['BulletConfigurationSpace']


class BulletConfigurationSpace(object):
    def __init__(self, client: BulletClient, cspace_names: List[str]):
        self.client = client
        self.cspace_names = cspace_names

    @property
    def world(self):
        return self.client.world

    def get_qpos(self):
        return np.array([self.world.get_qpos(name) for name in self.cspace_names])

    def set_qpos(self, qpos, forward=False):
        for name, v in zip(self.cspace_names, qpos):
            self.world.set_qpos(name, v)
        if forward:
            self.client.p.stepSimulation()

    @contextlib.contextmanager
    def qpos_context(self, forward=False):
        backup = self.get_qpos()
        yield
        self.set_qpos(backup, forward=forward)

    def get_xpos(self, name, type=None):
        return self.world.get_xpos(name, type=type)

    def get_xquat(self, name, type=None):
        return self.world.get_xquat(name, type=type)

    def get_xmat(self, name, type=None):
        return self.world.get_xmat(name, type=type)

    def fk(self, end_effector_name, qpos):
        with self.qpos_context():
            self.set_qpos(qpos)
            return self.get_xpos(end_effector_name), self.get_xmat(end_effector_name)

    def ik(self, end_effector_name, goal_xpos, goal_xquat=None):
        raise NotImplementedError()

    def get_collisions(self, qpos):
        raise NotImplementedError()

