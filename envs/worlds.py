import math
import os
import time
import copy
import json
import random
from os.path import join, dirname, abspath, isdir, isfile, basename
from os import listdir
from trimesh.transformations import translation_matrix as T
from trimesh.creation import axis, box
import trimesh
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np

from mesh_utils import CLOUD, create_tray, create_grid_meshes, Rotation2D, \
    get_color, fit_shape_in_bounds, transform_by_constraints, RENDER_PATH, \
    add_shape, regions_to_meshes, BLACK, R, get_color_name, triangles_to_meshes, \
    reorganize_points, CLOUD, RAINBOW_COLORS, get_area, is_inside, save_mesh, \
    reorganize_points_2, RAINBOW_COLOR_NAMES
from data_utils import get_grid_index, get_grid_offset, save_graph_data, get_grids_offsets, \
    print_line, compute_pairwise_collisions, apply_grid_mask, print_tensor, grid_offset_to_pose, r, \
    compute_world_constraints, expand_unordered_constraints
from render_utils import export_gif
from builders import get_tray_splitting_gen, get_triangles_splitting_gen, get_3d_box_splitting_gen


def get_world_class(world_name):
    from inspect import getmembers, isclass
    import sys
    import robot_worlds
    current_module = sys.modules[__name__]
    results = [a[1] for a in getmembers(current_module) + getmembers(robot_worlds)
               if isclass(a[1]) and a[0] == world_name]
    if len(results) > 0:
        return results[0]
    else:
        assert False, f"Unknown world name: {world_name}"


class CSPWorld(object):
    def __init__(self, w=3, l=2, h=0.5, orthographic=True, grid_size=0.5):
        self.name = self.__class__.__name__
        self.w = w
        self.l = l
        self.h = 0.01 if orthographic else h
        self.orthographic = orthographic

        self.grid_size = grid_size
        self.axis_markers = None
        self.grid_markers = None
        self.images = []
        self.labels = {}  ## semantic names given to meshes
        self.cfree = []  ## collision free pairs of objects
        self.color_assignments = {}  ## color assignments for meshes
        self.ignore_nodes = ['north', 'south', 'east', 'west']
        self.rotations = None  ## for qualitative task
        self.img_name = None

    @property
    def axis(self):
        """ create a 3D axis """
        if self.axis_markers is None:
            self.axis_markers = axis(origin_size=0.05)
            self.axis_markers.metadata['label'] = 'axis'
        return self.axis_markers

    @property
    def grids(self):
        if self.grid_markers is None:
            grids, c = create_grid_meshes(self.w, self.l, self.h, grid_size=self.grid_size)
            self.grid_markers = grids
            self.color_assignments = c
        return self.grid_markers

    def get_meshes(self):
        assert NotImplementedError

    def get_scene(self, show_axis=False, show_grid=False):
        meshes = self.get_meshes()
        if show_axis:
            meshes += [self.axis]
        if show_grid:
            meshes += self.grids
        lights = [trimesh.scene.lighting.Light(color=np.ones(3), intensity=10)]
        lights = None
        return trimesh.Scene(meshes, lights=lights)

    def sample_scene(self, **kwargs):
        """ sample based on world builders """
        assert NotImplementedError

    def load_scene(self):
        """ recreate scene from lisdf file """
        pass

    def randomize_scene(self):
        """ change the objects' dimensions, colors, texture, poses, etc. """
        pass

    def render(self, img_name=None, topdown=True, show_grid=False, show_axis=False, **kwargs):
        from render_utils import show_and_save
        if img_name is None:
            img_name = f'{self.name}.png'
        self.img_name = img_name
        scene = self.get_scene(show_grid=show_grid, show_axis=show_axis)
        img = show_and_save(scene, img_name, topdown=topdown, **kwargs)
        if isinstance(img, np.ndarray):
            self.images.append(img)
            return img

    def export_gif(self, gif_file=None, pause=1, save_pngs=False):
        if gif_file is None:
            gif_file = join(RENDER_PATH, f'{self.name}.gif')
        export_gif(self.images, gif_file, pause=pause, save_pngs=save_pngs)

    def render_object_crops(self, orthographic=True):
        """ render a cropped image of each movable in the scene """
        pass

    def generate_lisdf(self):
        """ record shapes and poses of all objects in the scene
                in a lisdf file for later planning in Pybullet
        """
        pass

    def generate_constraints(self, objects, sequential_sampling=False, same_order=True):
        """ generate constraints for the scene """
        objects = [mesh['label'] for mesh in objects.values() if mesh['label'] not in self.ignore_nodes]
        if sequential_sampling:
            constraints = [['in', 1, 0]]
            for i in range(1, len(objects)):
                if same_order or np.random.rand() < 0.5:
                    constraints.append(['cfree', i+1, i])
                else:
                    constraints.append(['cfree', i, i+1])
        else:
            constraints = [['in', i, 0] for i in range(1, len(objects))]
            for i in range(1, len(objects) - 1):
                for j in range(i + 1, len(objects)):
                    if same_order or np.random.rand() < 0.5:
                        constraints.append(['cfree', i, j])
                    else:
                        constraints.append(['cfree', j, i])
        return constraints

    def generate_json(self, input_mode='collisions', json_name=None,
                      constraints={}, world={}, same_order=True):
        """ record in a json file for collision checking """
        world.update({
            'name': self.name,
            'objects': {},
            'constraints': constraints,
        })
        scene = self.get_scene()
        for name, mesh in scene.geometry.items():
            if name not in self.labels:
                continue
            label = self.labels[name]
            if 'shadow' in label:
                continue

            center = mesh.centroid

            if 'shape' not in mesh.metadata:
                extents = mesh.extents
                if mesh.vertices.shape[0] == 6:
                    shape = 'triangle'
                    if extents[0] != extents[1]:
                        shape = 'arbitrary_triangle'
                        extents, center = self.get_triangle_representation(label)
                else:
                    shape = 'pointcloud'
            else:
                shape = mesh.metadata['shape']
                if 'extents' not in mesh.metadata:
                    if shape == 'cylinder':
                        extents = [mesh.metadata['radius'], mesh.metadata['height']]
                    else:
                        assert False, "unknown extents"
                else:
                    extents = mesh.metadata['extents']

                ## the box
                if isinstance(self, TriangularRandomSplitWorld):
                    center = list(center) + [0]

            rgba = mesh.visual.face_colors.tolist()[0] if hasattr(mesh.visual, 'face_colors') else [255] * 4
            color = get_color_name(rgba)
            world['objects'][name] = {
                'label': label,
                'shape': shape,
                'extents': tuple(extents),
                'center': tuple(center),
                'centroid': tuple(mesh.centroid),
                'rgba': rgba,
                'color': color,
            }
            if 'triangle' in shape:
                world['objects'][name].update({
                    'vertices': [tuple(vertex) for vertex in mesh.vertices],
                    'vertices_centered': [tuple(vertex - mesh.centroid) for vertex in mesh.vertices],
                    'faces': [tuple([int(p) for p in face]) for face in mesh.faces],
                })
            if shape == 'pointcloud':
                world['objects'][name].update({
                    'vertices': [tuple(vertex) for vertex in mesh.vertices],
                    'vertices_centered': [tuple(vertex - mesh.centroid) for vertex in mesh.vertices],
                })
            if label == 'east':
                world['objects']['geometry_0']['extents'] = tuple(list(world['objects']['geometry_0']['extents'][:2])
                                                                  + extents[-1:].tolist())
            if 'fitted_theta' in world and label in world['fitted_theta']:
                world['objects'][name]['theta'] = world['fitted_theta'][label]

            """ compute grid_offset """
            if 'grid_offset' in input_mode:
                grid_label = None
                if self.labels[name].startswith('tile_') or self.labels[name] == 'bottom':
                    if input_mode == 'grid_offset_mp4':
                        grid_label = get_grids_offsets(mesh.centroid, self.w, self.l, self.grid_size)
                    else:
                        grid_label = get_grid_offset(mesh.centroid, self.w, self.l, self.grid_size)
                world['objects'][name]['grid_label'] = grid_label

        """ compute constraints """
        if len(world['constraints']) == 0:
            world['constraints'] = self.generate_constraints(world['objects'], same_order=same_order)
        if 'qualitative' in input_mode:
            scale = min([self.w / 3, self.l / 2])
            world = compute_world_constraints(world, rotations=self.rotations, same_order=same_order, scale=scale)

        """ compute pairwise collisions """
        if 'collisions' in input_mode or 'diffuse_pairwise' in input_mode:
            world['collisions'] = self.check_collisions_in_scene(world['objects'], verbose=False)
            world = compute_pairwise_collisions(world)

        """ save to json file for inspections """
        if json_name is not None:
            with open(json_name, 'w') as f:
                for k, data in world['objects'].items():
                    world['objects'][k]['extents'] = list(data['extents'])
                    world['objects'][k]['center'] = list(data['center'])
                json.dump(world, fp=f, indent=3)
        return world

    def generate_pt(self, data=None, data_path=None, verbose=False, input_mode='diffuse_pairwise',
                    return_nodes=False, **kwargs):
        """ record each object as [type, w, l, x, y]
            output the count of labels (e.g. grid_index, is_collided) """

        if data is None:
            data = self.generate_json(input_mode=input_mode, **kwargs)

        nodes = []
        edge_index = []
        labels = []
        objects = [mesh['label'] for mesh in data['objects'].values() if mesh['label'] not in self.ignore_nodes]
        class_counts = {}

        if 'grid_offset' in input_mode:
            num_grids = int(self.w / self.grid_size) * int(self.l / self.grid_size)
            class_counts = {k: 0 for k in range(num_grids)}
        elif 'collisions' in input_mode:
            class_counts = {k: 0 for k in [0, 1]}

        if verbose:
            print()
        for name, mesh in data['objects'].items():
            name = mesh['label']
            if name in self.ignore_nodes:
                continue

            ## basic type, shape, and pose encoding
            obj_type = 1 if name.startswith('tile_') else 0
            shape_features = list(mesh['extents'])
            pose_feature = list(mesh['center'])

            if isinstance(self, RandomSplitQualitativeWorld):
                w, l = shape_features[:2]
                yaw = np.pi / 2
                if l > w:
                    l, w = shape_features[:2]
                    yaw = np.pi
                shape_features = [w, l]
                pose_feature = pose_feature[:2] + [np.sin(yaw), np.cos(yaw)]
            elif not isinstance(self, TriangularRandomSplitWorld):
                shape_features = shape_features[:2]
                pose_feature = pose_feature[:2]
            elif len(shape_features) == 3 and self.encoding == 'P2':
                shape_features = shape_features[:2] + [0, 0, 0, 0]
                pose_feature = pose_feature[:2] + [0, 0]

            node = [obj_type] + shape_features + pose_feature
            label = None

            if 'grid_offset' in input_mode:
                label = mesh['grid_label']
                if 'bottom' not in name:
                    if isinstance(label[0], int):
                        grids = [label[0]]
                    else:
                        grids = [l[0] for l in label]
                    for g in grids:
                        class_counts[g] += 1

            elif 'collisions' in input_mode:
                label = np.asarray([0]*len(objects))
                for xx in mesh['collisions']:
                    if xx not in objects:
                        xx = 'bottom'
                        labels[0][objects.index(name)] = 1
                    label[objects.index(xx)] = 1
                    class_counts[1] += 1

            elif 'diffuse' in input_mode:
                label = obj_type
                if 'image' in input_mode and name in self.images:
                    image = np.sum(self.images[name], axis=-1)
                    image = image / np.max(image)
                    image = image.reshape(-1).tolist()
                    node = [obj_type] + shape_features + image + pose_feature

            if verbose:
                print(f"{name}: {[round(n, 3) for n in node]}\t -> {label}")
            nodes.append(node)
            labels.append(label)

        if return_nodes:
            return np.array(nodes)

        ## add edge_index
        if 'diffuse_pairwise' in input_mode or 'qualitative' in input_mode:
            edge_index = data['constraints']
            class_counts[len(nodes)-1] = 1
        else:
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    if i != j:
                        edge_index.append([i, j])

            if 'collisions' in input_mode and len(data['collisions']) > 0:
                class_counts[0] = len(nodes) * (len(nodes)-1) / 2 - class_counts[1] / 2
                if verbose:
                    for i in range(len(objects)):
                        print(f"{i}:\t {objects[i]},\t {[round(nn, 2) for nn in nodes[i]]},\t {labels[i]}")
                    print(f"edge_index: {edge_index}")

        ## modify labels
        if input_mode in ['grid_offset_mp4', 'grid_offset_oh4']:
            if isinstance(labels[0], list):
                labels = [label + [[-1] * 3] * (4 - len(label)) for label in labels]

        ## save .pt files
        if data_path is not None:
            save_graph_data(nodes, edge_index, labels, data_path)

        return class_counts

    def summarize_objects(self, tile_pose_only=False, input_mode='grid_offset_mp4'):
        data = self.generate_json(input_mode=input_mode)
        if tile_pose_only:
            print_data = {}
            for k, v in data['objects'].items():
                if not v['label'].startswith('tile_'):
                    continue
                grid_label = v['grid_label']
                if grid_label is not None:
                    grid_label = [grid_label[0], round(grid_label[1], 3), round(grid_label[2], 3)]
                center = tuple([round(c, 3) for c in v['center'][:2]])
                print_data[v['label']] = [v['shape'], v['color'], center, grid_label, v['collisions']]
        else:
            print_data = data
        pprint(print_data, width=200)
        return data

    def check_constraints_satisfied(self, **kwargs):
        return self.check_collisions_in_scene(**kwargs)

    def check_collisions_in_scene(self, objects=None, verbose=True):
        from collisions import check_collisions_in_scene
        if objects is None:
            objects = self.generate_json()['objects']
        collisions = check_collisions_in_scene(objects, rotations=self.rotations, verbose=verbose)
        collisions = [c for c in collisions if c not in self.cfree and \
                      c[0] != 'bottom' and c[1] != 'bottom']
        if verbose: print('collisions', collisions)
        return collisions


class TrayWorld(CSPWorld):
    """ objects are in a tray """
    def __init__(self, t=0.1, color=CLOUD, **kwargs):
        super(TrayWorld, self).__init__(**kwargs)
        self.t = t
        self.tray, names = create_tray(self.w, self.l, self.h, t, color=color)
        self.tiles = []
        self.cfree = [('north', 'east'), ('south', 'east'), ('north', 'west'), ('south', 'west')]

    def get_meshes(self):
        return self.tray + self.tiles

    def get_scene(self, **kwargs):
        scene = super(TrayWorld, self).get_scene( **kwargs)
        self.labels = {name: m.metadata['label'] for name, m in scene.geometry.items() if 'label' in m.metadata}
        # unlabeled = [m for m in scene.geometry.values() if 'label' not in m.metadata]
        if 'show_grid' in kwargs and kwargs['show_grid']:
            size = 0.04 * self.grid_size
            indicators = []
            for name, mesh in scene.geometry.items():
                if 'label' not in mesh.metadata:
                    print(f"unlabeled mesh\t|  vertices = {mesh.vertices.shape}, metadata = {mesh.metadata}")
                if not mesh.metadata['label'].startswith('tile_'):
                    continue
                name = name + '_indicator'
                t = copy.deepcopy(mesh.centroid)
                t[2] += 0.05
                index = get_grid_index(t, self.w, self.l, grid_size=self.grid_size)
                mesh = box(extents=[size, size, 0.01], transform=T(t))
                if index not in self.color_assignments:
                    continue
                mesh.visual.vertex_colors = self.color_assignments[index] ## BLACK
                mesh.metadata['label'] = name
                indicators.append(mesh)
                self.labels[name] = name
            scene = trimesh.Scene(list(scene.geometry.values()) + indicators)
        return scene

    @property
    def base(self):
        return self.tray[0]

    # def get_node_features(self, mode='discretize', **kwargs):


class ShapeSettingWorld(TrayWorld):
    """ objects are simples shapes, e.g. box, circle, triangle, etc.
            collision-free is not enforced
    """
    def __init__(self, **kwargs):
        super(ShapeSettingWorld, self).__init__(**kwargs)
        self.used_colors = []
        self.constraints = {}
        self.randomization_count = 0

    def get_shape_name(self, shape, label=None):
        if label is None:
            label = f"{shape}_{len([n for n in self.labels.values() if f'{shape}_' in n])}"
        return f'tile_{label}'

    def remove_shape(self, mesh):
        label = mesh.metadata['label']
        index = self.tiles.index(mesh)
        self.tiles.remove(mesh)
        if label in self.constraints:
            del self.constraints[label]
        return label, index

    def add_shape(self, shape, height=None, size=0.2, x=0, y=0, z=None, R=None, random_color=True,
                  color=None, alpha=1.0, constraints=[], label=None):
        if color is None:
            color = get_color(alpha, used=self.used_colors, random_color=random_color)
            self.used_colors.append(tuple(color))
        elif isinstance(color, str) and color in RAINBOW_COLOR_NAMES:
            color = RAINBOW_COLORS[RAINBOW_COLOR_NAMES.index(color)]
        if height is None:
            height = self.h
        mesh = add_shape(shape, size, height, color)

        if len(constraints) > 0:
            x, y = transform_by_constraints(mesh, self.base, constraints)
        if z is None:
            z = height / 2
        mesh.apply_transform(T([x, y, z]))
        if R is not None:
            mesh.apply_transform(R)
        self.tiles.append(mesh)

        label = self.get_shape_name(shape, label)
        mesh.metadata['label'] = label
        self.constraints[label] = constraints
        self.labels[len(self.labels)] = label
        return mesh

    def set_shape_pose(self, mesh_data, mesh, x, y):
        """ somehow tile.apply_transform(T(diff)) doesn't work """
        label, index = self.remove_shape(mesh)

        shape = mesh_data['shape']
        if shape == 'arbitrary_triangle':
            size = mesh_data['vertices']
            shape = 'triangle'
            dz = None
        else:
            dx, dy, dz = mesh_data['extents']
            size = [dx, dy]

        mesh = add_shape(shape, size, dz, mesh_data['rgba'])
        mesh.apply_transform(T([x, y, self.h / 2 + 0.01]))
        mesh.metadata['label'] = label
        self.tiles.insert(index, mesh)

    def randomize_scene(self):
        """ TODO: Doesn't work yet """
        print(np.random.randint(0, 100000))
        new_tiles = []
        for mesh in self.tiles:
            constraints = self.constraints.pop(mesh)
            x, y = transform_by_constraints(mesh, self.base, constraints)
            print(x, y)
            mesh.apply_transform(T([x, y, self.h / 2]))
            new_tiles.append(mesh)
            self.constraints[mesh] = constraints
        self.tiles = new_tiles
        self.render(img_name=f'{self.name}_{self.randomization_count}.png',
                    show=False, array=True)
        self.randomization_count += 1

    def sample_scene(self, case=None, cases=[]):
        if len(cases) == 0:
            cases = list(range(3))
        if case is None:
            case = random.choice(cases)

        ## for debugging learner
        if case == 111:
            self.add_shape('square', size=0.5)
        elif case == 222:
            m = self.add_shape('square', size=0.2)
            m.apply_transform(T([-0.5, -0.5, self.h / 2]))
        elif case == 333:
            m = self.add_shape('square', size=0.3)
            m.apply_transform(T([0.5, 0.5, self.h / 2]))

        elif case == 0:
            self.add_shape('square', size=0.5, constraints=[('LeftIn',)])
            self.add_shape('square', size=0.5, constraints=[('RightIn',)])
        elif case == 1:
            # self.add_shape('triangle', size=0.5, constraints=[('TopIn',)])
            self.add_shape('circle', size=0.5, constraints=[('TopIn',)])
            self.add_shape('circle', size=0.5, constraints=[('BottomIn',)])
        elif case == 2:
            self.add_shape('circle', size=0.5)
            self.add_shape('circle', size=0.5)
            self.add_shape('square', size=0.5)
            self.add_shape('square', size=1.2)

    def shake_scenes_gen(self, num=10, is_generator=False, img_name_template="", verbose=False, visualize=False):
        """ shake the scene by moving the movable objects """
        def perturb_pose(pose, delta=0.2):
            return np.array(pose)[:2] + np.random.uniform(-delta, delta, size=2)

        if verbose:
            print_line(f"original:")
            data = self.summarize_objects(tile_pose_only=True, input_mode='collisions')
        else:
            data = self.generate_json(input_mode='collisions')

        centroids = {tile.metadata['label']: tile.centroid for tile in self.tiles}
        for i in range(num):
            for tile in self.tiles:
                # tile.apply_transform(perturb_pose(tile.centroid))
                label = tile.metadata['label']
                mesh_data = [m for m in data['objects'].values() if m['label'] == label][0]
                x, y = perturb_pose(centroids[label])
                self.set_shape_pose(mesh_data, tile, x, y)

            if verbose:
                print_line(f"shake {i}:")
                self.summarize_objects(tile_pose_only=True, input_mode='collisions')

            if visualize:
                image_name = img_name_template.format(i)
                self.render(show=False, show_grid=True, save=True, array=True)  ## , img_name=image_name

            if is_generator:
                yield self
        if visualize:
            self.export_gif(gif_file='RS_shake_scenes.gif', save_pngs=True)


class QiQiaoWorld(ShapeSettingWorld):
    """ objects are simples shapes, e.g. box, circle, triangle, etc.
            collision-free is not enforced
    """
    def __init__(self, w=10, **kwargs):
        kwargs['w'] = kwargs['l'] = w
        kwargs['t'] = w / 20
        super(QiQiaoWorld, self).__init__(**kwargs)

    def sample_scene(self, case='fit'):
        s = self.w/2  ## size of medium-sized triangle
        h = self.h/2 + 0.1  ## height of the tray
        a = 1
        if case == 'logo':
            s *= 0.35

        ## load all shapes
        tl1 = self.add_shape('triangle', size=s*np.sqrt(2), label='triangle_l1', alpha=a)
        tl2 = self.add_shape('triangle', size=s*np.sqrt(2), label='triangle_l2', alpha=a)
        tm1 = self.add_shape('triangle', size=s, label='triangle_m1', alpha=a)
        ts1 = self.add_shape('triangle', size=s/np.sqrt(2), label='triangle_s1', alpha=a)
        ts2 = self.add_shape('triangle', size=s/np.sqrt(2), label='triangle_s2', alpha=a)
        pg = self.add_shape('parallelogram', size=s/np.sqrt(2), label='parallelogram', alpha=a)
        sq = self.add_shape('square', size=s/np.sqrt(2), label='square', alpha=a)

        def transform(mesh, x, y, theta, dh=0.0, resize=None):
            # if resize is not None:
            #     x = x * s / resize
            #     y = y * s / resize
            mesh.apply_transform(Rotation2D(theta))
            mesh.apply_transform(T([x, y, h+dh]))

        if case == 'fit':
            transform(tl1, 0, 0, 45)
            transform(tl2, 0, 0, 135)
            transform(tm1, s, -s, 90)
            transform(ts1, 0, 0, -45)
            transform(ts2, -s/2, -s/2, -135)
            transform(pg, s, 0, 45)
            transform(sq, 0, -s/2, 45)

        elif case == 'logo':  ## designed for 10 by 10
            transform(tl1, -0.3, -2.3, 15, resize=5)
            transform(tl2, -0.33, -2.34, 15+90, resize=5)
            transform(tm1, 0, 0.5, 30, resize=5)
            transform(ts1, -4.8, -3.4, -90+15, resize=5)
            transform(ts2, 4, -2.3, 90-30, resize=5)
            pg.apply_transform(R(angle=np.deg2rad(180), direction=[0, 1, 0], point=[0, 0, 0]))
            transform(pg, -1.5, 0.85, -(90-30-45), dh=self.h, resize=5)
            transform(sq, 0, 3.25, 45, resize=5)


class RandomSplitWorld(ShapeSettingWorld):
    """ boxes are arranged by random splitting the tray into sections
            and putting shapes in each section,
            so collision-free is guaranteed
    """
    def __init__(self, **kwargs):
        super(RandomSplitWorld, self).__init__(**kwargs)

    def sample_scene(self, min_num_objects=2, max_num_objects=6, min_offset_perc=0.1, **kwargs):
        """ first get region boxes from `get_tray_spliting_gen` """
        max_depth = math.ceil(math.log2(max_num_objects)) + 1
        gen = get_tray_splitting_gen(num_samples=2, min_num_regions=min_num_objects,
                                     max_num_regions=max_num_objects, max_depth=max_depth)
        regions = next(gen(self.w, self.l))
        meshes = regions_to_meshes(regions, self.w, self.l, self.h, min_offset_perc=min_offset_perc)
        self.tiles.extend(meshes)

    def construct_scene_from_objects(self, objects, rotations):
        for i, (name, obj) in enumerate(objects.items()):
            if name == 'bottom':
                continue
            size = obj['extents'][:2]
            # size = [n * 2 for n in size]
            x, y, _ = obj['center']
            self.add_shape('box', size=size, x=x, y=y, color=obj['color'])
            if isinstance(self, RandomSplitQualitativeWorld):
                self.rotations[f"tile_box_{i}"] = rotations[i]

    def construct_scene_from_graph_data(self, nodes, labels=None, predictions=None, verbose=False, phase='truth'):
        """ check collisions during model evaluation """
        w, l = nodes[0, 1:3]
        # if w == 1 and l == 1:
        #     nodes[:, 1] *= 3
        #     nodes[:, 2] *= 2
        #     nodes[:, -2] *= 0.5
        #     nodes[:, -1] *= 0.5
        if verbose:
            print_tensor('nodes', nodes)
            print_tensor('predictions', predictions)

        for i in range(1, nodes.shape[0]):

            yaw = None
            g = None
            if nodes[i].shape[0] == 5:
                t, bw, bl, x, y = nodes[i]
            # elif nodes[i].shape[0] == 6:
            #     t, bw, bl, g, dx, dy = nodes[i]
            elif nodes[i].shape[0] == 6:
                t, bw, bl, x, y, yaw = nodes[i]
                if isinstance(self, RandomSplitQualitativeWorld):
                    self.rotations[f"tile_box_{i-1}"] = yaw
            elif nodes[i].shape[0] == 8:
                t, bw, bl, x, y, g, dx, dy = nodes[i]

            color = RAINBOW_COLORS[i - 1]
            prediction = None
            if phase == 'prediction' and nodes[i, 0] == 2:
                g = predictions[0]
                result = apply_grid_mask(predictions[0], predictions[1])
                (dx, dy) = result
                prediction = g, (dx, dy)
            elif phase == 'prediction' and nodes[i, 0] == 1:
                color = CLOUD

            if g is not None:
                x, y = grid_offset_to_pose(g, (dx, dy), w, l, grid_size=0.5)
            if verbose:
                print(f"{i}\t nodes: {r(nodes[i])}\t -> (predictions = {r(prediction)})\t | labels: {r(labels[i])}")

            if yaw is not None:
                import transformations
                direction = [0, 0, 1]
                center = [x, y, self.h/2]
                rot_matrix = transformations.rotation_matrix(-yaw, direction, center)
            else:
                rot_matrix = None

            self.add_shape('box', size=(bw, bl), x=x, y=y, color=color, R=rot_matrix)


class RandomSplitQualitativeWorld(RandomSplitWorld):
    """ tiles with extra labels such as
            'in', 'center-in', 'left-in', 'right-in', 'top-in', 'bottom-in',
            'cfree', 'left-of', 'right-of', 'top-of', 'bottom-of',
            'touching', 'close-to', 'away-from', 'h-aligned', 'v-aligned'
    """
    def __init__(self, **kwargs):
        super(RandomSplitQualitativeWorld, self).__init__(**kwargs)
        self.qualitative_constraints = None
        self.rotations = {}

    def sample_scene(self, min_offset_perc=0, **kwargs):
        super().sample_scene(min_offset_perc=min_offset_perc, **kwargs)

    def get_current_constraints(self):
        from denoise_fn import ignored_constraints
        data = self.generate_json(input_mode='qualitative')
        return [tuple(d) for d in data['constraints'] if d[0] not in ignored_constraints]

    def check_constraints_satisfied(self, same_order=False, **kwargs):
        from denoise_fn import ignored_constraints
        collisions = self.check_collisions_in_scene(**kwargs)
        ## check other constraints
        if len(collisions) > 0:
            if self.img_name is not None:
                json_name = self.img_name.replace('.png', '.json')
                world = {
                    'check_constraints_satisfied': collisions,
                }
                self.generate_json(input_mode='qualitative', json_name=json_name, world=world)
            return collisions
        current_constraints = self.get_current_constraints()
        given_constraints = [tuple(d) for d in self.qualitative_constraints if d[0] not in ignored_constraints]
        if not same_order:
            current_constraints = expand_unordered_constraints(current_constraints)
            given_constraints = expand_unordered_constraints(given_constraints)
        missing = [ct for ct in given_constraints if ct not in current_constraints]

        if self.img_name is not None:
            json_name = self.img_name.replace('.png', '.json')
            world = {
                'current_constraints': current_constraints,
                'given_constraints': given_constraints,
                'missing': missing,
            }
            # print('\n', self.img_name, missing)
            # print('\t', current_constraints)
            # print('\t', given_constraints)
            self.generate_json(input_mode='qualitative', json_name=json_name, world=world)
        return missing

    def construct_scene_from_graph_data(self, nodes, constraints, **kwargs):
        """ give ground truth constraints to check if they are satisfied """
        self.qualitative_constraints = constraints
        super().construct_scene_from_graph_data(nodes, **kwargs)


class RandomSplitWorld3D(ShapeSettingWorld):
    """ boxes are arranged by random splitting the tray into sections
            and putting shapes in each section,
            so collision-free is guaranteed
    """
    def __init__(self, h=1, **kwargs):
        color = CLOUD
        color[-1] *= 0.3
        super(RandomSplitWorld3D, self).__init__(h=h, orthographic=False, color=color, **kwargs)

    def sample_scene(self, min_num_objects=6, max_num_objects=10, **kwargs):
        """ first get region boxes from `get_tray_splitting_gen` """
        max_depth = math.ceil(math.log2(max_num_objects)) + 1
        gen = get_3d_box_splitting_gen(num_samples=40, min_num_regions=min_num_objects,
                                       max_num_regions=max_num_objects, max_depth=max_depth)
        regions = next(gen(self.w, self.l, self.h))
        meshes = regions_to_meshes(regions, self.w, self.l, self.h, max_offset=0.1)
        self.tiles.extend(meshes)

    def construct_scene_from_graph_data(self, nodes, labels=None, predictions=None, verbose=False, phase='truth'):
        """ reconstruct to check collisions during model evaluation """
        w, l = nodes[0, 1:3]
        if w == 1 and l == 1:
            nodes[:, 1] *= 3
            nodes[:, 2] *= 2
            nodes[:, -2] *= 0.5
            nodes[:, -1] *= 0.5
        if verbose:
            print_tensor('nodes', nodes)
            print_tensor('predictions', predictions)

        for i in range(1, nodes.shape[0]):

            if nodes[i].shape[0] == 5:
                t, bw, bl, x, y = nodes[i]
                g = None
            elif nodes[i].shape[0] == 6:
                t, bw, bl, g, dx, dy = nodes[i]
            elif nodes[i].shape[0] == 8:
                t, bw, bl, x, y, g, dx, dy = nodes[i]

            color = RAINBOW_COLORS[i - 1]
            prediction = None
            if phase == 'prediction' and nodes[i, 0] == 2:
                g = predictions[0]
                result = apply_grid_mask(predictions[0], predictions[1])
                (dx, dy) = result
                prediction = g, (dx, dy)
            elif phase == 'prediction' and nodes[i, 0] == 1:
                color = CLOUD

            if g is not None:
                x, y = grid_offset_to_pose(g, (dx, dy), w, l, grid_size=0.5)
            if verbose:
                print(f"{i}\t nodes: {r(nodes[i])}\t -> (predictions = {r(prediction)})\t | labels: {r(labels[i])}")
            self.add_shape('box', size=(bw, bl), x=x, y=y, color=color)


class TriangularRandomSplitWorld(ShapeSettingWorld):
    """ triangles are arranged by random splitting the tray into sections
        and putting shapes in each section, so collision-free is guaranteed

        with corresponding
            data_transform_cn_diffuse_batch() in data_transforms.py
            render_world_from_graph() in data_utils.py

    """
    def __init__(self, image_dim=64, **kwargs):
        super(TriangularRandomSplitWorld, self).__init__(**kwargs)
        self.triangle_gt_extents = []
        self.triangle_gt_centers = []
        self.images = {}
        self.image_dim = image_dim
        self.name += f'[{image_dim}]_'
        self.encoding = 'P1'  ## 'P2

    def sample_scene(self, min_num_objects=6, max_num_objects=8, show_orientation=False,
                     input_mode='diffuse_pairwise_image'):
        """ first get region boxes from `get_tray_splitting_gen` """
        gen = get_triangles_splitting_gen()
        k = -4
        results = next(gen(self.w, self.l, num_points=max([max_num_objects+k, 1])))
        while len(results) > max_num_objects or len(results) < min_num_objects:
            if len(results) > max_num_objects:
                if np.random.rand() > 0.5:
                    chosen = range(len(results))
                    chosen = random.sample(chosen, min_num_objects)
                    results = [results[i] for i in chosen]
                    break
                k -= 1
            elif len(results) < min_num_objects:
                k += 1
            results = next(gen(self.w, self.l, num_points=max([max_num_objects+k, 1])))
            # print('len(regions)', len(results), [min_num_objects, max_num_objects], 'k', k)
        triangles, triangles_recentered = self.process_triangles(results, show_orientation=show_orientation)
        meshes = triangles_to_meshes(results, self.h, triangles_recentered=triangles_recentered,
                                     show_orientation=show_orientation)
        self.tiles.extend(meshes)
        if show_orientation:
            self.render(show=True, save=False, show_grid=True)
        if input_mode == 'diffuse_pairwise_image':
            self.generate_images_from_meshes(visualize=False)

    def process_triangles(self, results, show_orientation=False, verbose=False):
        triangles = []
        triangles_recentered = []
        if show_orientation and verbose:
            print()
        for i, (tri, lengths) in enumerate(results):
            [p1, p2, p3], extent, center = self.get_rotation(tri, lengths)
            triangles.append(tri)
            triangles_recentered.append([p1, p2, p3])
            self.triangle_gt_extents.append(extent)
            self.triangle_gt_centers.append(center)
            if show_orientation:
                if verbose:
                    print(f"triangle {i}:\t points: {tri}\t extent: {r(extent)}\t center: {r(center)}")
                self.triangle_gt_extents.append(extent)
                self.triangle_gt_centers.append([0, 0, 1, 0])
        return triangles, triangles_recentered

    def generate_images_from_meshes(self, visualize=False):
        ## one box
        resolution = (self.image_dim, self.image_dim, 3)
        mesh = self.tray[0]
        color = mesh.visual.face_colors[0][:3]
        img = np.ones(resolution, dtype=np.uint8) * color
        self.images['bottom'] = img

        ## all triangles
        for i, mesh in enumerate(self.tiles):
            color = mesh.visual.face_colors[0][:3]
            extent = self.triangle_gt_extents[i]
            img = self.get_triangle_image(extent, color=color, visualize=visualize, png_name=f'triangle_{i}')
            self.images[mesh.metadata['label']] = img

    def get_triangle_image(self, extent, resolution=(64, 64, 3), color=(255, 255, 255), visualize=True, png_name=None):
        """ put the triangle inside a square, with P1 at origin """

        if self.encoding == 'P1':
            x1, y1 = resolution[0] / 2, resolution[1] / 2
            x2 = x1
            y2 = y1 + extent[0] / (self.w*2) * resolution[1]
            x3 = x1 - extent[2] / (self.l*2) * resolution[0]
            y3 = y1 + extent[1] / (self.w*2) * resolution[1]
            result = x1, x2, x3, y1, y2, y3
            x1, x2, x3, y1, y2, y3 = [int(x) for x in result]

        else:
            scale = 0.6
            x, y = resolution[0] / 2, resolution[1] / 2
            x1 = int(x - extent[1] / self.l * resolution[0] * scale)
            x2 = int(x - extent[3] / self.l * resolution[0] * scale)
            x3 = int(x - extent[5] / self.l * resolution[0] * scale)
            y1 = int(y + extent[0] / self.w * resolution[1] * scale)
            y2 = int(y + extent[2] / self.w * resolution[1] * scale)
            y3 = int(y + extent[4] / self.w * resolution[1] * scale)

        ## --------------------------------------------------------

        area = get_area(x1, y1, x2, y2, x3, y3)
        canvas = np.zeros(resolution, dtype=np.uint8)
        for x in range(resolution[0]):
            for y in range(resolution[1]):
                if is_inside(x1, y1, x2, y2, x3, y3, x, y, A_ABC=area):
                    canvas[x, y] = color
        if visualize:
            render_dir = join(RENDER_PATH, 'get_triangle_image_2')
            if not isdir(render_dir):
                os.makedirs(render_dir)
            plt.imshow(canvas, interpolation='nearest')
            plt.axis('off')
            plt.savefig(join(render_dir, png_name), bbox_inches='tight')
            plt.close()
        return canvas

    def get_triangle_representation(self, label):
        idx = eval(label.split('_')[-1])
        return self.triangle_gt_extents[idx], self.triangle_gt_centers[idx]

    def get_rotation(self, triangle, lengths, debug=False):
        if self.encoding == 'P1':
            """ convert triangles to 3D meshes
            left: actual orientation
            right: reconstructed recentered triangle, represented by [v1, v4]
    
              P2_0
             / \
            /   \
        P3_0 .   \ v2          P3
               .  \      v4     ..
            v3   . \       . '    `.
                P1 .\ _'____________`_ P2
                              v1
            """
            ## visualize orientation
            order = reorganize_points(lengths)
            p1 = triangle[order[0]]
            p2_0 = triangle[order[1]]
            p3_0 = triangle[order[2]]

            l_12 = np.sqrt((p1[0] - p2_0[0]) ** 2 + (p1[1] - p2_0[1]) ** 2)
            p2 = np.array([p1[0] + l_12, p1[1]])
            v1 = np.array(p2) - np.array(p1)  ## after recentering, p2 is aligned with x axis
            v2 = np.array(p2_0) - np.array(p1)

            cs = np.dot(v2, v1) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            sn = np.cross(v2, v1) / (np.linalg.norm(v1) * np.linalg.norm(v2))

            R = np.array([[cs, -np.abs(sn)], [np.abs(sn), cs]])
            if v2[1] > 0:
                R = np.array([[cs, np.abs(sn)], [-np.abs(sn), cs]])

            v3 = np.array(p3_0) - np.array(p1)

            if debug:
                p2_0_recon = R @ v1 + np.array(p1)
                p2_recon = np.linalg.inv(R) @ (p2_0_recon - np.array(p1)) + np.array(p1)

                ## identify if the rotation direction is CW or CCW
                ## return rotation radian in CW direction in range [-pi, pi], starting from +x axis
                theta = np.arccos(cs)
                if np.cross(v1, v2) > 0:
                    theta = 2 * np.pi - theta
                if theta > np.pi:
                    theta = theta - 2 * np.pi

                cs_ = np.cos(theta)
                sn_ = np.sin(theta)

                printout = f"\np2_0_recon: {p2_0_recon}, p2_0: {p2_0}\tp2_recon: {p2_recon}, p2: {r(p2)}\t " \
                           f"cos: {[r(cs, 3), r(cs_, 3)]}, sin: {[r(sn, 3), r(sn_, 3)]}\t v1: {v1}, v2: {v2}"
                result = [np.linalg.norm(p2_0_recon - p2_0) < 0.001, np.linalg.norm(p2_recon - p2) < 0.001]
                result += [np.linalg.norm(cs - cs_) < 0.001, np.linalg.norm(sn - sn_) < 0.001]
                printout += f"\t result: {result}"
                print(printout)

                p3 = np.array([v3[0] * cs_ - v3[1] * sn_, v3[0] * sn_ + v3[1] * cs_]) + np.array(p1)

            p3 = R @ v3 + np.array(p1)
            v4 = np.array(p3) - np.array(p1)
            # beta = np.arccos(np.dot(v3, v4) / (np.linalg.norm(v3) * np.linalg.norm(v4)))

            extent = [l_12] + list(v4)
            center = list(p1) + [cs, sn]  ## [theta]

            return [p1, p2, p3], extent, center

        else:

            """ convert triangles to 3D meshes
            left: actual orientation
            right: reconstructed recentered triangle, represented by [v1, v4]

              P3_0
             / \
            /   \
        P1_0 . . \   v              P1
               .  \             u    ..
             v   . \            . '  . `.
                   .\   P2 _'____________`_ P3
                  P2_0             u
            """
            ## visualize orientation
            centroid = np.mean(triangle, axis=0)
            order = reorganize_points_2(lengths)

            while True:
                p1_0 = triangle[order[0]]
                p2_0 = triangle[order[1]]
                p3_0 = triangle[order[2]]

                v23 = np.array(p2_0) - np.array(p3_0)
                u23 = np.array([1, 0])

                cs = np.dot(v23, u23) / (np.linalg.norm(v23) * np.linalg.norm(u23))
                sn = np.cross(v23, u23) / (np.linalg.norm(v23) * np.linalg.norm(u23))

                R = np.array([[cs, -sn], [sn, cs]])

                centroids = np.repeat(centroid[np.newaxis, :], 3, axis=0)
                cv = np.stack([p1_0, p2_0, p3_0], axis=0) - centroids
                cu = (R @ cv.T).T

                if cu[0, 1] < 0:
                    order = order[0], order[2], order[1]
                    continue

                u = cu + centroids
                p1 = u[0].tolist()
                p2 = u[1].tolist()
                p3 = u[2].tolist()
                break

            extent = cu.reshape(-1)
            center = list(centroid) + [cs, sn]

            return [p1, p2, p3], extent, center

    def construct_scene_from_graph_data(self, nodes, labels=None, predictions=None, verbose=False, **kwargs):

        if self.encoding == 'P1':
            """ convert triangles to 3D meshes
            left: reconstructed triangle
            right: given triangle, cos, sin
    
              P2_0
             / \
            /   \
        P3_0 .   \ v2          P3
               .  \      v4     ..
            v3   . \       . '    `.
                P1 .\ _'____________`_ P2
                              v1
            """
            def rotate_vector(p, theta):
                x, y = p
                if isinstance(theta, list):
                    cs, sn = theta
                    R = np.array([[cs, -np.abs(sn)], [np.abs(sn), cs]])
                    if sn > 0:
                        R = np.array([[cs, np.abs(sn)], [-np.abs(sn), cs]])
                else:
                    cs = np.cos(theta)
                    sn = np.sin(theta)
                    R = np.array([[cs, -sn], [sn, cs]])
                return R @ np.array([x, y])
                # return np.linalg.inv(R) @ np.array([x, y])

            for i in range(1, nodes.shape[0]):

                ## ----------------- version 1: using theta -----------------
                # _, l, x3, y3, x1, y1, r1 = nodes[i]
                #
                # if r1 < 0:
                #     r1 = 2 * np.pi + r1
                # r1 = 2 * np.pi - r1
                # v2 = rotate_vector([l, 0], r1)
                # v3 = rotate_vector([x3, y3], r1)

                ## ----------------- version 2: using theta -----------------
                _, l, x3, y3, x1, y1, cs, sn = nodes[i]

                norm = np.sqrt(cs ** 2 + sn ** 2)
                cs = cs / norm
                sn = sn / norm

                v2 = rotate_vector([l, 0], [cs, sn])
                v3 = rotate_vector([x3, y3], [cs, sn])

                ## ----------------------------------------------------------
                p1_0 = x1, y1
                p2_0 = x1 + v2[0], y1 + v2[1]
                p3_0 = x1 + v3[0], y1 + v3[1]

                points = [p1_0, p2_0, p3_0]
                color = RAINBOW_COLORS[i - 1]
                self.add_shape('triangle', size=points, color=color)
                self.triangle_gt_extents.append([l, x3, y3])
                self.triangle_gt_centers.append([x1, y1, cs, sn])

        else:

            """ convert triangles to 3D meshes
            left: reconstructed triangle
            right: given triangle, cos, sin
    
              P2_0
             / \
            /   \
        P3_0 . . \ v                P3
               .  \             u    ..
            v    . \            . '  . `.
                   .\   P1 _'____________`_ P2
                  P1_0             u
            """

            for i in range(1, nodes.shape[0]):

                _, x1, y1, x2, y2, x3, y3, x, y, cs, sn = nodes[i]

                norm = np.sqrt(cs ** 2 + sn ** 2)
                cs = cs / norm
                sn = sn / norm
                R = np.array([[cs, -sn], [sn, cs]])

                cu = np.array([[x1, y1], [x2, y2], [x3, y3]])
                cv = (R.T @ cu.T).T
                v = cv + np.array([x, y])

                points = [v[0].tolist(), v[1].tolist(), v[2].tolist()]

                color = RAINBOW_COLORS[i - 1]
                self.add_shape('triangle', size=points, color=color, **kwargs)
                self.triangle_gt_extents.append(v.reshape(-1).tolist())
                self.triangle_gt_centers.append([x, y, cs, sn])

    def generate_meshes(self, nodes=None, mesh_dir_name='test'):
        """ generate urdf files for all triangles in the scene
            if error "ValueError: convex compositions require testVHACD installed!",
            install v-hacd at https://github.com/kmammou/v-hacd """
        mesh_dir = abspath(join(__file__, '..', '..', 'models', 'TriangularRandomSplitWorld', mesh_dir_name))
        if not isdir(mesh_dir):
            os.makedirs(mesh_dir, exist_ok=True)

        ## recreate the meshes in base pose
        original_meshes = copy.deepcopy(self.tiles)
        if nodes is None:
            nodes = self.generate_pt(input_mode='diffuse_pairwise', verbose=False, return_nodes=True)
        nodes[:, -4:] = [0, 0, 1, 0]
        self.construct_scene_from_graph_data(nodes, height=0.5)
        for i, mesh in enumerate(self.tiles[len(original_meshes):]):
            save_mesh(mesh, join(mesh_dir, f'tile_{i}.obj'))
