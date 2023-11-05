from numpy.random import rand
from typing import Iterable, Tuple
import numpy as np
from math import sqrt
import random

from data_utils import compute_qualitative_constraints, summarize_constraints, r as rd


def get_tray_splitting_gen(num_samples=40, min_num_regions=2, max_num_regions=6, max_depth=3, default_min_size=0.4):

    Region = Tuple[float, float, float, float]  # l, t, w, h

    def partition(box: Region, depth: int = 3) -> Iterable[Region]:
        if rand() < 0.3 or depth == 0:
            yield box

        else:
            if rand() < 0.5:
                axis = 0
            else:
                axis = 1

            split_point = rand() * box[axis + 2]
            if axis == 0:
                yield from partition((box[0], box[1], split_point, box[3]), depth - 1)
                yield from partition((box[0] + split_point, box[1], box[2] - split_point, box[3]), depth - 1)
            else:
                yield from partition((box[0], box[1], box[2], split_point), depth - 1)
                yield from partition((box[0], box[1] + split_point, box[2], box[3] - split_point), depth - 1)

    def filter_regions(regions: Iterable[Region], min_size: float) -> Iterable[Region]:
        return [r for r in regions if r[2] > min_size and r[3] > min_size]

    def gen(w, l):
        min_size = min([w, l]) / 2 * default_min_size
        def get_regions():
            regions = []
            for region in partition((0, 0, w, l), max_depth):
                regions.append(region)
            return regions
        count = num_samples
        while True:
            regions = get_regions()
            regions = filter_regions(regions, min_size)
            if min_num_regions <= len(regions) <= max_num_regions:
                count -= 1
                yield regions
            if count == 0:
                break
        yield None
    return gen


def test_tray_splitting():
    gen = get_tray_splitting_gen(num_samples=2)
    for boxes in gen(4, 3):
        print(boxes)


##########################################################################


def construct_objects(regions, w, l, h, z):
    objects = {
        'bottom': {
            'extents': (w, l, 0.1),
            'center': (0, 0, -0.05)
        }
    }
    for i, region in enumerate(regions):
        objects[f"tile_{i}"] = {
            'extents': (region[2], region[3], h),
            'center': (-w/2+region[0]+region[2]/2, -l/2+region[1]+region[3]/2, z + h / 2)
        }
    return objects


def get_3d_box_splitting_gen(num_samples=40, min_num_regions=6, max_num_regions=10, **kwargs):

    bottom_gen = get_tray_splitting_gen(num_samples=num_samples, min_num_regions=min_num_regions-3,
                                        max_num_regions=max_num_regions-2, **kwargs)
    top_gen = get_tray_splitting_gen(num_samples=num_samples, min_num_regions=1,
                                     max_num_regions=2, **kwargs)

    def point_outside_of_box(point, box):
        if box[0] < point[0] < box[0] + box[2] and box[1] < point[1] < box[1] + box[3]:
            return False
        return True

    def point_in_boxes(point, boxes):
        for box in boxes:
            if box[0] <= point[0] <= box[0] + box[2] and box[1] <= point[1] <= box[1] + box[3]:
                return True
        return False

    def get_sides(boxes):
        lefts = sorted([box[0] for box in boxes])
        tops = sorted([box[1] for box in boxes], reverse=True)
        rights = sorted([box[0] + box[2] for box in boxes], reverse=True)
        bottoms = sorted([box[1] + box[3] for box in boxes])
        return lefts, tops, rights, bottoms

    def compute_secondary_support_region(boxes, region):
        """ given a set of boxes, compute the largest region that is supported
            by one of them but not the primary region """
        areas = []
        for box in boxes:
            left = max(region[0], box[0])
            top = max(region[1], box[1])
            right = min(region[0] + region[2], box[0] + box[2])
            bottom = min(region[1] + region[3], box[1] + box[3])
            w = right - left
            h = bottom - top
            areas.append(box[2] * box[3] - w * h)
        box = boxes[areas.index(max(areas))]

        ## find the region that's inside box but outside of region
        boxes = [box, region]
        lefts, tops, rights, bottoms = get_sides(boxes)
        xx = sorted(lefts + rights)
        yy = sorted(bottoms + tops)
        areas = {}
        for x1 in xx:
            for x2 in reversed(xx):
                if x1 >= x2:
                    continue
                for y1 in yy:
                    for y2 in reversed(yy):
                        if y1 >= y2:
                            continue
                        points = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
                        failed = False
                        for point in points:
                            if not point_outside_of_box(point, region) or not point_in_boxes(point, [box]):
                                failed = True
                                break
                        if failed:
                            continue
                        w = x2 - x1
                        h = y2 - y1
                        areas[(x1, y1, w, h)] = w * h
        if len(areas) == 0:
            return None
        return max(areas, key=areas.get)

    def compute_support_region(boxes):
        """ given a set of boxes, compute the largest region that is supported by all of them """
        lefts, tops, rights, bottoms = get_sides(boxes)
        areas = {}
        for l in range(2):
            for t in range(2):
                for r in range(2):
                    for b in range(2):
                        left = lefts[l]
                        top = tops[t]
                        right = rights[r]
                        bottom = bottoms[b]
                        points = [(left, top), (right, top), (left, bottom), (right, bottom)]
                        if not all([point_in_boxes(point, boxes) for point in points]):
                            continue
                        w = right - left
                        h = bottom - top
                        areas[(left, top, w, h)] = w * h
        return max(areas, key=areas.get)

    def sample_support_boxes(constraints):
        """ select two or three boxes that are adjacent to each other to be the support of the next split """
        finding = []
        found = []
        used_pairs = []
        repetitions = []
        pool = [c[1:] for c in constraints if c[0] == 'close-to']
        random.shuffle(pool)
        for pair in pool:
            if (pair[1], pair[0]) in repetitions or pair in used_pairs:
                continue
            repetitions.append(pair)

            for k in finding:
                for X in k:
                    if X in pair:
                        Y = [p for p in k if p != X][0]
                        Z = [p for p in pair if p != X][0]
                        if (Y, Z) in pool or (Z, Y) in pool:
                            found.append((X, Y, Z))
                            used_pairs.extend([(X, Y), (Y, Z), (Z, X), (Y, X), (Z, Y), (X, Z)])
                            finding.remove(k)
                            continue

            if pair not in used_pairs:
                finding.append(pair)

        if len(found) > 0:
            random.shuffle(found)
            return [f-1 for f in found[0]]
        return None

    def add_top_regions(r, h2, h4):
        x, y, w, l = r
        new_regions = []
        cc = num_samples
        while True:
            for small_regions in top_gen(w, l):
                cc -= 1
                if cc == 0:
                    return None
                if small_regions is None or len(small_regions) < 2:
                    continue
                for sr in small_regions:
                    xx, yy, ww, ll = sr
                    new_regions.append((x+xx, y+yy, h2, ww, ll, h4))
                return new_regions

    def add_3d_regions(regions, w, l, h):
        """ let's call the 2d split algorithm -> 2D-SPLIT
            A. use 2D-SPLIT to generate 3d regions in the bottom layer
            B. select three of the regions (h1) to support a top layer (h3) generated by 2D-SPLIT,
                the largest remaining area supports another box (h5)
            C. for each region not selected (h2), it will support a top layer (h4) generated by 2D-SPLIT
        """
        h1 = np.random.uniform(0, h * 0.66)
        h2 = np.random.uniform(h1, h * 0.8)
        h3 = np.random.uniform(0.2*(h-h1), h-h1)
        h4 = np.random.uniform(0.2*(h-h2), h-h2)
        h5 = np.random.uniform(0.2*(h-h1), h-h1)

        def dh():
            return np.random.uniform(0, h * 0.1)

        ## generate middle layer
        objects = construct_objects(regions, w, l, h1, 0)
        constraints = compute_qualitative_constraints(objects)
        boxes = sample_support_boxes(constraints)
        if boxes is None:
            return None
        selected_regions = [regions[b] for b in boxes]
        region = compute_support_region(selected_regions)
        region_secondary = compute_secondary_support_region(selected_regions, region)

        ## add heights to all regions
        new_regions = []
        for i, r in enumerate(regions):
            x, y, w, l = r
            if i not in boxes:
                new_regions.append((x, y, 0, w, l, h2))
                top_regions = add_top_regions(r, h2, h4)
                if top_regions is None:
                    return None
                new_regions.extend(top_regions)
            else:
                new_regions.append((x, y, 0, w, l, h1))
        x, y, w, l = region
        new_regions.append((x, y, h1, w, l, h3))
        if region_secondary is not None:
            x, y, w, l = region_secondary
            new_regions.append((x, y, h1, w, l, h5))

        ## minus for stability concerns
        new_regions = [tuple(list(r[:5]) + [r[5]-dh()]) for r in new_regions]

        return new_regions

    def fn(w, l, h):
        count = num_samples
        for regions in bottom_gen(w, l):
            if regions is None:
                continue
            if min_num_regions-3 > len(regions) or len(regions) > max_num_regions-2:
                continue
            regions = add_3d_regions(regions, w, l, h)
            if regions is None:
                continue
            if min_num_regions <= len(regions) <= max_num_regions:
                count -= 1
                yield regions
            if count == 0:
                break
    return fn


def test_3d_box_splitting():
    gen = get_3d_box_splitting_gen()
    for triangle in gen(3, 2, 1):
        print(triangle)


###########################################################################

class Delaunay2D(object):
    """
    Class to compute a Delaunay triangulation in 2D
    ref: https://github.com/jmespadero/pyDelaunay2D/blob/master/delaunay2D.py
    """

    def __init__(self, center=(0, 0), radius=0.5):
        """ Init and create a new frame to contain the triangulation
        center -- Optional position for the center of the frame. Default (0,0)
        radius -- Optional distance from corners to the center. """
        center = np.asarray(center)
        # Create coordinates for the corners of the frame
        self.coords = [center + radius * np.array((-1, -1)),
                       center + radius * np.array((+1, -1)),
                       center + radius * np.array((+1, +1)),
                       center + radius * np.array((-1, +1))]

        # Create two dicts to store triangle neighbours and circumcircles.
        self.triangles = {}
        self.circles = {}

        # Create two CCW triangles for the frame
        T1 = (0, 1, 3)
        T2 = (2, 3, 1)
        self.triangles[T1] = [T2, None, None]
        self.triangles[T2] = [T1, None, None]

        # Compute circumcenters and circumradius for each triangle
        for t in self.triangles:
            self.circles[t] = self.circumcenter(t)

    def circumcenter(self, tri):
        """ Compute circumcenter and circumradius of a triangle in 2D.
        Uses an extension of the method described here:
        http://www.ics.uci.edu/~eppstein/junkyard/circumcenter.html """
        pts = np.asarray([self.coords[v] for v in tri])
        pts2 = np.dot(pts, pts.T)
        A = np.bmat([[2 * pts2, [[1],
                                 [1],
                                 [1]]],
                     [[[1, 1, 1, 0]]]])

        b = np.hstack((np.sum(pts * pts, axis=1), [1]))
        x = np.linalg.solve(A, b)
        bary_coords = x[:-1]
        center = np.dot(bary_coords, pts)

        # radius = np.linalg.norm(pts[0] - center) # euclidean distance
        radius = np.sum(np.square(pts[0] - center))  # squared distance
        return (center, radius)

    def in_circle_fast(self, tri, p):
        """ Check if point p is inside of precomputed circumcircle of tri. """
        center, radius = self.circles[tri]
        return np.sum(np.square(center - p)) <= radius

    def in_circle_robust(self, tri, p):
        """ Check if point p is inside of circumcircle around the triangle tri.
        This is a robust predicate, slower than compare distance to centers
        ref: http://www.cs.cmu.edu/~quake/robust.html """
        m1 = np.asarray([self.coords[v] - p for v in tri])
        m2 = np.sum(np.square(m1), axis=1).reshape((3, 1))
        m = np.hstack((m1, m2))  # The 3x3 matrix to check
        return np.linalg.det(m) <= 0

    def add_point(self, p):
        """ Add a point to the current DT, and refine it using Bowyer-Watson. """
        p = np.asarray(p)
        idx = len(self.coords)
        # print("coords[", idx,"] ->",p)
        self.coords.append(p)

        # Search the triangle(s) whose circumcircle contains p
        bad_triangles = []
        for T in self.triangles:
            # Choose one method: inCircleRobust(T, p) or inCircleFast(T, p)
            if self.in_circle_fast(T, p):
                bad_triangles.append(T)

        # Find the CCW boundary (star shape) of the bad triangles,
        # expressed as a list of edges (point pairs) and the opposite
        # triangle to each edge.
        boundary = []
        # Choose a "random" triangle and edge
        T = bad_triangles[0]
        edge = 0
        # get the opposite triangle of this edge
        while True:
            # Check if edge of triangle T is on the boundary...
            # if opposite triangle of this edge is external to the list
            tri_op = self.triangles[T][edge]
            if tri_op not in bad_triangles:
                # Insert edge and external triangle into boundary list
                boundary.append((T[(edge + 1) % 3], T[(edge - 1) % 3], tri_op))

                # Move to next CCW edge in this triangle
                edge = (edge + 1) % 3

                # Check if boundary is a closed loop
                if boundary[0][0] == boundary[-1][1]:
                    break
            else:
                # Move to next CCW edge in opposite triangle
                edge = (self.triangles[tri_op].index(T) + 1) % 3
                T = tri_op

        # Remove triangles too near of point p of our solution
        for T in bad_triangles:
            del self.triangles[T]
            del self.circles[T]

        # Retriangle the hole left by bad_triangles
        new_triangles = []
        for (e0, e1, tri_op) in boundary:
            # Create a new triangle using point p and edge extremes
            T = (idx, e0, e1)

            # Store circumcenter and circumradius of the triangle
            self.circles[T] = self.circumcenter(T)

            # Set opposite triangle of the edge as neighbour of T
            self.triangles[T] = [tri_op, None, None]

            # Try to set T as neighbour of the opposite triangle
            if tri_op:
                # search the neighbour of tri_op that use edge (e1, e0)
                for i, neigh in enumerate(self.triangles[tri_op]):
                    if neigh:
                        if e1 in neigh and e0 in neigh:
                            # change link to use our new triangle
                            self.triangles[tri_op][i] = T

            # Add triangle to a temporal list
            new_triangles.append(T)

        # Link the new triangles each another
        N = len(new_triangles)
        for i, T in enumerate(new_triangles):
            self.triangles[T][1] = new_triangles[(i + 1) % N]  # next
            self.triangles[T][2] = new_triangles[(i - 1) % N]  # previous

    def export_triangles(self):
        """ Export the current list of Delaunay triangles """
        # Filter out triangles with any vertex in the extended BBox
        return self.triangles
        # return [(a - 4, b - 4, c - 4)
        #         for (a, b, c) in self.triangles if a > 3 and b > 3 and c > 3]

    def export_circles(self):
        """ Export the circumcircles as a list of (center, radius) """
        # Remember to compute circumcircles if not done before
        # for t in self.triangles:
        #     self.circles[t] = self.circumcenter(t)

        # Filter out triangles with any vertex in the extended BBox
        # Do sqrt of radius before of return
        return [(self.circles[(a, b, c)][0], sqrt(self.circles[(a, b, c)][1]))
                for (a, b, c) in self.triangles if a > 3 and b > 3 and c > 3]


def get_triangles_splitting_gen():
    """ randomly sample some 2D points in a rectangle and connects them to form triangles """

    def get_sides_area(points):
        # x, y = zip(*points)
        # area = 0.5 * (x[0] * (y[1] - y[2]) + x[1] * (y[2] - y[0]) + x[2] * (y[0] - y[1]))
        x1, y1 = points[0]
        x2, y2 = points[1]
        x3, y3 = points[2]
        l3 = sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        l1 = sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2)
        l2 = sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)
        p = (l1 + l2 + l3) / 2
        area = sqrt(p * (p - l1) * (p - l2) * (p - l3))
        return (l1, l2, l3), area

    def move_points_closer(point, points):
        middle = np.mean([p for p in points if tuple(p) != tuple(point)], axis=0)
        return middle + (point-middle) * (1 - 0.3 * np.random.random())

    def gen(w, h, num_points=4):
        # Create a random set of points
        bbox = np.asarray([w, h])
        seeds = np.random.random((num_points, 2))

        # Create Delaunay Triangulation and insert points one by one
        dt = Delaunay2D(radius=0.5)
        for s in seeds:
            dt.add_point(s - 0.5)

        triangles = dt.export_triangles()
        tri_points = []
        for triangle in triangles:
            points = [dt.coords[t] * bbox for t in triangle]
            modified_points = [move_points_closer(p, points) for p in points]
            lengths, area = get_sides_area(modified_points)
            if area < 0.01 * w * h or min([area / l for l in lengths]) < 0.1:
                continue
            tri_points.append([[p for p in modified_points], lengths])
        yield tri_points
    return gen


def test_triangle_splitting():
    gen = get_triangles_splitting_gen()
    for triangle in gen(4, 3):
        print(triangle)


###########################################################################


if __name__ == "__main__":
    # test_tray_splitting()
    # test_triangle_splitting()
    test_3d_box_splitting()
