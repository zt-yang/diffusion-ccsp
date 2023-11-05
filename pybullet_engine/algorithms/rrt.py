#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : rrt.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/01/2019
#
# This file is part of HACL-PyTorch.
# Distributed under terms of the MIT license.

import numpy as np
import numpy.random as npr

from typing import Optional, Tuple, List
from pybullet_engine.algorithms.space import ProblemSpace

__all__ = ['RRTNode', 'RRT', 'smooth_path', 'get_smooth_path', 'optimize_path', 'rrt', 'birrt']

"""
Basic RRT algorithm.

The following algorithms and data structures have a generic interface. The are not designed specifically for a
specific robot.
"""


class RRTNode(object):
    def __init__(self, config, parent=None):
        self.config = config
        self.parent = parent
        self.children = list()

    def add_to_children(self, other):
        self.children.append(other)
        return self

    def attach_to(self, other):
        self.parent = other
        self.parent.add_to_children(self)
        return self

    def backtrace(self, config=True):
        path = list()

        def dfs(x):
            if x.parent is not None:
                dfs(x.parent)
            path.append(x.config if config else x)

        try:
            dfs(self)
            return path
        finally:
            del dfs

    @classmethod
    def from_states(cls, states):
        if isinstance(states, list):
            return [cls(s) for s in states]
        else:
            return cls(states)

    def __repr__(self):
        return f'RRTNode(config={self.config}, parent={self.parent})'


def traverse_rrt_bfs(nodes):
    queue = nodes.copy()
    results = list()

    while len(queue) > 0:
        x = queue[0]
        queue = queue[1:]
        results.append(x)
        for y in x.children:
            queue.append(y)

    return results


class RRT(object):
    """The RRT tree."""

    def __init__(self, pspace, roots):
        if isinstance(roots, RRTNode):
            roots = [roots]
        else:
            assert isinstance(roots, list)

        self.pspace = pspace
        self.roots = roots
        self.size = len(roots)

    def extend(self, parent, child_config):
        child = RRTNode(child_config).attach_to(parent)
        self.size += 1
        return child

    def nearest(self, config, pspace=None):
        if pspace is None:
            pspace = self.pspace

        best_node, best_value = None, np.inf
        for node in traverse_rrt_bfs(self.roots):
            distance = pspace.distance(node.config, config)
            if distance < best_value:
                best_value = distance
                best_node = node

        return best_node


def smooth_path(pspace, path, nr_attemps=100, use_fine_path: bool = False):
    if nr_attemps is None:
        nr_attemps = 100

    if use_fine_path:
        path = get_smooth_path(pspace, path)

    for i in range(nr_attemps):
        if len(path) <= 2:
            break

        # Use the uniform pair sampling method.
        a, b = npr.randint(0, len(path) - 1), npr.randint(0, len(path) - 1)
        if a > b:
            a, b = b, a + 1
        else:
            b = b + 1
        # The original version:
        # a = npr.randint(0, len(path) - 1)
        # b = npr.randint(a + 1, len(path))
        if use_fine_path:
            success, _, subpath = pspace.try_extend_path(path[a], path[b])
            if success:
                path = path[:a + 1] + subpath[1:] + path[b:]
        else:
            if pspace.validate_path(path[a], path[b]):
                path = path[:a + 1] + path[b:]

    return path


def get_smooth_path(pspace, path):
    cpath = [path[0]]
    for config in path[1:]:
        cpath.extend(pspace.cspace.gen_path(cpath[-1], config)[1][1:])
    path = cpath
    return path


def optimize_path_forward(pspace, path):
    spath = [path[0]]
    ptr = 1
    while ptr + 1 < len(path):
        if pspace.validate_path(spath[-1], path[ptr + 1]):
            ptr += 1
        else:
            _, sub_path = pspace.cspace.gen_path(spath[-1], path[ptr])
            spath.append(sub_path[1])
            if len(sub_path) == 2:
                ptr += 1
    spath.extend(pspace.cspace.gen_path(spath[-1], path[ptr])[1][1:])
    return spath


def optimize_path(pspace, path, *args, **kwargs):
    path = get_smooth_path(path)
    while True:
        length = len(path)
        path = optimize_path_forward(pspace, path)
        path.reverse()
        path = optimize_path_forward(pspace, path)
        path.reverse()
        if len(path) >= length:
            break
    return path


def rrt(pspace, start_state, goal_state, nr_iterations=1000, p_sample_goal=0.05, nr_smooth_iterations=None):
    rrt = RRT(pspace, RRTNode.from_states(start_state))

    for i in range(nr_iterations):
        sample_goal = False
        if npr.uniform() < p_sample_goal:
            sample_goal = True
            next_config = goal_state
        else:
            next_config = pspace.sample()

        node = rrt.nearest(next_config)
        success, next_config, _ = pspace.try_extend_path(node.config, next_config)
        if next_config is not None:
            new_node = rrt.extend(node, next_config)
            if sample_goal and success:
                return smooth_path(pspace, new_node.backtrace(), nr_smooth_iterations), rrt

    return None, rrt


def birrt(pspace: ProblemSpace, start_state, goal_state, nr_iterations=1000, nr_smooth_iterations=None, smooth_fine_path=False) -> Tuple[Optional[List[np.ndarray]], Tuple[RRT, RRT]]:
    rrt_start = RRT(pspace, RRTNode.from_states(start_state))
    rrt_goal = RRT(pspace, RRTNode.from_states(goal_state))
    swapped = False

    if True:
        # Try to directly connecting the starting state and the goal state.
        success, next_config, _ = pspace.try_extend_path(start_state, goal_state)
        if success:
            rrt_start.extend(rrt_start.nearest(goal_state), goal_state)
            return [start_state, goal_state], (rrt_start, rrt_goal)

    for i in range(nr_iterations):
        next_config = pspace.sample()

        node_start = rrt_start.nearest(next_config)
        success, next_config, _ = pspace.try_extend_path(node_start.config, next_config)
        if next_config is not None:
            new_node_start = rrt_start.extend(node_start, next_config)

            node_goal = rrt_goal.nearest(next_config)
            success, next_config, _ = pspace.try_extend_path(node_goal.config, next_config)
            if success:
                path1 = new_node_start.backtrace()
                path2 = node_goal.backtrace()
                path = list(path1) + list(reversed(path2))

                # still add the new node to the rrt_goal for better visualization.
                _ = rrt_goal.extend(node_goal, next_config)

                if swapped:
                    path.reverse()
                    rrt_start, rrt_goal = rrt_goal, rrt_start
                return smooth_path(pspace, path, nr_smooth_iterations, use_fine_path=smooth_fine_path), (rrt_start, rrt_goal)
            elif next_config is not None:
                _ = rrt_goal.extend(node_goal, next_config)

        rrt_start, rrt_goal, swapped = rrt_goal, rrt_start, not swapped

    return None, (rrt_start, rrt_goal)

