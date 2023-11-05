#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : interpolation_utils.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 05/26/2023
#
# This file is part of HACL-PyTorch.
# Distributed under terms of the MIT license.

from typing import Optional

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from scipy.interpolate import interp1d


class LinearSpline(object):
    """Linear spline interpolation.

    This class is a wrapper of scipy.interpolate.interp1d that mimics the CubicSpline interface.
    """
    def __init__(self, xs: np.ndarray, ys: np.ndarray):
        assert len(xs) == len(ys)
        self.xs = np.asarray(xs, dtype=np.float32)
        self.ys = np.asarray(ys, dtype=np.float32)
        assert len(self.xs.shape) == 1, 'xs should be a 1D array.'
        self.interpolator = interp1d(xs, ys, axis=0, kind='linear', fill_value='extrapolate')

    def __call__(self, x: float):
        return self.interpolator(x)


def gen_cubic_spline(ys: np.ndarray):
    """Generate a cubic spline interpolation from an array of points."""
    xs = list(range(len(ys)))
    xs = np.asarray(xs, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.float32)
    return CubicSpline(xs, ys)


def gen_linear_spline(ys):
    """Generate a linear spline interpolation from an array of points."""
    xs = list(range(len(ys)))
    return LinearSpline(xs, ys)


def project_to_cubic_spline(spl: CubicSpline, y: np.ndarray, ys: np.ndarray):
    y = np.asarray(y, dtype=np.float32)
    dspl = spl.derivative()

    def f(x):
        return ((spl(float(x)) - y) ** 2).sum()

    def df(x):
        return 2 * ((spl(float(x)) - y) * dspl(float(x))).sum()

    x0 = np.argmin(np.abs(ys - y).sum(axis=-1))
    # res = minimize(f, x0, jac=df, method='L-BFGS-B', bounds=[(0, len(ys) - 1)])
    res = minimize(f, x0, jac=df, method='L-BFGS-B', bounds=[(max(0, x0 - 1), min(len(ys) - 1, x0 + 1))])

    # print('estimated time x0', x0)
    # print('bound', [(max(0, x0 - 1), min(len(ys) - 1, x0 + 1))])
    # print(res)
    return float(res.x[0])


def get_next_target_cubic_spline(spl: CubicSpline, y: np.ndarray, step_size: float, ys: np.ndarray):
    x = project_to_cubic_spline(spl, y, ys)
    # print('current time x', x, '/', len(ys) - 1)
    x = min(max(x + step_size, 0), len(ys) - 1)
    # print('next time x', x, '/', len(ys) - 1)
    return spl(x)


def project_to_linear_spline(spl: LinearSpline, y: np.ndarray, minimum_x=None):
    y = np.asarray(y, dtype=np.float32)

    def f(x):
        y_prime = spl(x)
        return ((y - y_prime) ** 2).sum()

    x0 = np.argmin(np.abs(spl.ys - y).sum(axis=-1))
    min_x = max(0, x0 - 1)
    if minimum_x is not None:
        min_x = max(min_x, minimum_x)
    res = minimize(f, x0, bounds=[(min_x, max(min_x + 0.1, min(len(spl.ys) - 1, x0 + 1)))])
    return float(res.x[0])


def get_next_target_linear_spline(spl: LinearSpline, y: np.ndarray, step_size: float, minimum_x: Optional[float] = None):
    x = project_to_linear_spline(spl, y, minimum_x=minimum_x)
    x = min(max(x + step_size, 0), len(spl.ys) - 1)
    return x, spl(x)
