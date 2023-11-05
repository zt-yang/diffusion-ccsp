#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : range.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 04/23/2021
#
# This file is part of HACL-PyTorch.
# Distributed under terms of the MIT license.

# Below are classes for defining the ranges of values of the joints.
# These classes are generic, they don't depend on the particular
# robot.

import random

__all__ = ['Range']


class Range(object):
    """A range of values, handles wraparound."""

    def __init__(self, low, high, wrap_around=False):
        self.low = low
        self.high = high
        self.wrap_around = wrap_around

    def difference(self, config1, config2):  # difference (with wraparound)
        if self.wrap_around:
            if config1 < config2:
                if abs(config2 - config1) < abs((self.low - config1) + (config2 - self.high)):
                    return config2 - config1
                else:
                    return (self.low - config1) + (config2 - self.high)
            else:
                if abs(config2 - config1) < abs((self.high - config1) + (config2 - self.low)):
                    return config2 - config1
                else:
                    return (self.high - config1) + (config2 - self.low)
        else:
            return config2 - config1

    def in_range(self, value):  # test if in range
        if self.wrap_around:
            altered = value
            while not (self.low <= altered <= self.high):
                if altered > self.high:
                    altered -= self.high - self.low
                else:
                    altered += self.high - self.low
            return altered
        else:
            if self.contains(value):
                return value
            else:
                return None

    def sample(self):  # sample random value
        return (self.high - self.low) * random.random() + self.low

    def contains(self, x):
        return self.wrap_around or (self.low <= x <= self.high)
