#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : base.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 08/24/2022
#
# This file is part of HACL-PyTorch.
# Distributed under terms of the MIT license.

from pybullet_engine.client import BulletClient


class BulletModel(object):
    def __init__(self, client: BulletClient):
        self.client = client

    @property
    def p(self):
        return self.client.p

    @property
    def world(self):
        return self.client.w

    @property
    def w(self):
        return self.client.w

