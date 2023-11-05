import trimesh
import numpy as np
import random
import time
from collections import namedtuple

from trimesh.visual.color import hex_to_rgba
from trimesh.viewer import windowed
from trimesh.creation import box, cylinder, axis
from trimesh.transformations import translation_matrix as T, rotation_matrix as R

from config import *

""" project configurations """

RED = hex_to_rgba('#e74c3c')
ORANGE = hex_to_rgba('#e67e22')
BLUE = hex_to_rgba('#3498db')
GREEN = hex_to_rgba('#2ecc71')
YELLOW = hex_to_rgba('#f1c40f')
PURPLE = hex_to_rgba('#9b59b6')
GREY = hex_to_rgba('#95a5a6')
CLOUD = hex_to_rgba('#ecf0f1')
MIDNIGHT = hex_to_rgba('#34495e')
WHITE = hex_to_rgba('#ffffff')
BLACK = hex_to_rgba('#000000')
## Russian pallate
PINK = hex_to_rgba('#f8a5c2')
AQUA = hex_to_rgba('#63cdda')
PLUM = hex_to_rgba('#B33771')
NAVY = hex_to_rgba('#273c75')

DARKER_RED = hex_to_rgba('#c0392b')
DARKER_ORANGE = hex_to_rgba('#d35400')
DARKER_BLUE = hex_to_rgba('#2980b9')
DARKER_GREEN = hex_to_rgba('#27ae60')
DARKER_YELLOW = hex_to_rgba('#f39c12')
DARKER_PURPLE = hex_to_rgba('#8e44ad')
DARKER_GREY = hex_to_rgba('#7f8c8d')
DARKER_CLOUD = hex_to_rgba('#bdc3c7')
DARKER_MIDNIGHT = hex_to_rgba('#2c3e50')
DARKER_PINK = hex_to_rgba('#f78fb3')
DARKER_AQUA = hex_to_rgba('#3dc1d3')
DARKER_PLUM = hex_to_rgba('#6D214F')
DARKER_NAVY = hex_to_rgba('#192a56')

RAINBOW_COLORS = [RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE, MIDNIGHT, GREY, PINK, AQUA, PLUM, NAVY]
DARKER_COLORS = [DARKER_RED, DARKER_ORANGE, DARKER_YELLOW, DARKER_GREEN, DARKER_BLUE, DARKER_PURPLE,
                 DARKER_MIDNIGHT, DARKER_GREY, DARKER_PINK, DARKER_AQUA, DARKER_PLUM, DARKER_NAVY]
CLASSIC_COLORS = [BLACK, WHITE, CLOUD, DARKER_CLOUD]

RAINBOW_COLOR_NAMES = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'midnight',
                       'grey', 'cloud', 'pink', 'aqua', 'plum', 'navy']
DARKER_COLOR_NAMES = [f'darker_{n}' for n in RAINBOW_COLOR_NAMES]
CLASSIC_COLOR_NAMES = ['black', 'white', 'cloud', 'darker_cloud']

COLOR_NAMES = {tuple(RAINBOW_COLORS[i][:3]): RAINBOW_COLOR_NAMES[i] for i in range(len(RAINBOW_COLORS))}
COLOR_NAMES.update({tuple(DARKER_COLORS[i][:3]): DARKER_COLOR_NAMES[i] for i in range(len(DARKER_COLORS))})
COLOR_NAMES.update({tuple(CLASSIC_COLORS[i][:3]): CLASSIC_COLOR_NAMES[i] for i in range(len(CLASSIC_COLORS))})

AABB = namedtuple('AABB', ['lower', 'upper'])
PANDA_GRIPPER_MESH_PATH = abspath(join(dirname(dirname(__file__)), 'packing_models',
                                       'models', 'franka_description', 'hand.ply'))


def Rotation2D(theta, point=[0, 0, 0]):
    return R(angle=np.deg2rad(theta), direction=[0, 0, 1], point=point)


def get_color(alpha=1.0, used=[], random_color=True, colors=RAINBOW_COLORS):
    colors = [c for c in colors if tuple(c) not in used]
    if len(colors) == 0:
        return None
    for c in colors:
        c[3] *= alpha
    if random_color:
        return random.choice(colors)
    return colors[0]


def get_color_name(color):
    if tuple(color[:3]) not in COLOR_NAMES:
        return 'unknown'
    return COLOR_NAMES[tuple(color[:3])]


def random_point_in_bounds(bounds2d):
    x = np.random.uniform(bounds2d[0][0], bounds2d[1][0])
    y = np.random.uniform(bounds2d[0][1], bounds2d[1][1])
    return [x, y]


def fit_shape_in_bounds(bounds2d, shape):
    bb = shape.bounding_box.bounds[:, :2]
    extents = (bb[1] - bb[0])
    new_bounds = np.array([bounds2d[0]+extents/2, bounds2d[1]-extents/2])
    return random_point_in_bounds(new_bounds)


def transform_by_constraints(mesh, base, constraints):
    np.random.seed(np.random.randint(0, 1000))
    base_bounds = np.copy(base.bounding_box.bounds)
    base_center = base_bounds.mean(axis=0)[:2]
    for c in constraints:
        if c[0] == 'LeftIn':
            base_bounds[1][0] = base_center[0]
        elif c[0] == 'RightIn':
            base_bounds[0][0] = base_center[0]
        elif c[0] == 'TopIn':
            base_bounds[0][1] = base_center[1]
        elif c[0] == 'BottomIn':
            base_bounds[1][1] = base_center[1]
    return fit_shape_in_bounds(base_bounds[:, :2], mesh)


def triangle(size, height, transform=None, z=0.0):
    """ example of making a sandwich shaped mesh """

    if isinstance(size, list):
        if len(size) == 3:
            vertices = [[size[i][0], size[i][1], z] for i in range(len(size))]
            vertices += [[size[i][0], size[i][1], z+height] for i in range(len(size))]
        else:
            vertices = size
        faces = [[2, 1, 0], [1, 3, 0], [3, 2, 0], [2, 5, 1], [4, 3, 1], [5, 4, 1], [3, 5, 2], [4, 5, 3]]
        faces += [[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 5, 2], [1, 3, 4], [1, 4, 5], [2, 5, 3], [3, 5, 4]]
    else:
        ## define vertices and faces of each piece
        vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1]]
        vertices = np.multiply(np.array(vertices), np.array([size, size, height])) - np.array([0, 0, height/2])
        vertices = vertices.tolist()

        ## Note the counter-clockwise winding order.
        faces = [[2, 1, 0], [1, 3, 0], [3, 2, 0], [2, 5, 1], [4, 3, 1], [5, 4, 1], [3, 5, 2], [4, 5, 3]]
        ## Reversing the triangle order to clockwise, results in all invisible faces:
        # faces = [[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 5, 2], [1, 3, 4], [1, 4, 5], [2, 5, 3], [3, 5, 4]]
        ## Reversing some triangle order, e,g, to [2, 1, 0], results in some invisible faces:
        # faces = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 5], [1, 3, 4], [1, 4, 5], [2, 3, 5], [3, 4, 5]]

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    if transform is not None:
        mesh.apply_transform(transform)
    return mesh


def parallelogram(size, height, transform=None):
    """ example of making parallelogram out of two triangles """

    ## define vertices and faces of each piece
    vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1]]
    vertices.extend([[-1, 1, 0], [-1, 1, 1]])
    vertices = np.multiply(np.array(vertices), np.array([size, size, height])) - np.array([0, 0, height/2])
    vertices = vertices.tolist()

    ## Note the counter-clockwise winding order.
    faces = [[2, 1, 0], [1, 3, 0], [3, 2, 0], [2, 5, 1], [4, 3, 1], [5, 4, 1], [3, 5, 2], [4, 5, 3]]
    faces.remove([3, 5, 2])
    faces.remove([3, 2, 0])
    faces.extend([[6, 2, 0], [6, 0, 3], [6, 5, 2], [7, 6, 3], [6, 7, 5], [7, 3, 5]])

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.apply_transform(R(angle=np.deg2rad(180), direction=[0, 1, 0], point=[0, 0, 0]))
    if transform is not None:
        mesh.apply_transform(transform)
    return mesh


def view_mesh(mesh_file=join(MESH_PATH, 'triangle_000.obj')):
    """ view the mesh """
    mesh = trimesh.load_mesh(mesh_file)
    mesh.show()


def create_tray(w, l, h, t=0.1, color=CLOUD):
    """ create a tray with given width, length, height, and thickness, placed at origin """

    lighter_color = np.copy(color)
    lighter_color[-1] = 0.3 * 255
    meshes = [
        box(extents=[w, l, t], transform=T([0, 0, -t/2])),  # bottom
        box(extents=[w, t, h], transform=T([0, (l+t)/2, h/2])),  # north
        box(extents=[w, t, h], transform=T([0, -(l+t)/2, h/2])),  # south
        box(extents=[t, l+2*t, h], transform=T([-(w+t)/2, 0, h/2])),  # west
        box(extents=[t, l+2*t, h], transform=T([(w+t)/2, 0, h/2])),  # east
    ]
    names = ['bottom', 'north', 'south', 'west', 'east']
    for i, m in enumerate(meshes):
        m.visual.vertex_colors = color if i != 0 else lighter_color
        m.metadata['label'] = names[i]

    return meshes, names


def test_tray_scene(show=False, save=True, orthographic=True):
    """ create a tray and take a top-down picture of it """
    w, l, h, t = 3, 2, 0.5, 0.1
    if orthographic:
        h = 0.001
    meshes = create_tray(w, l, h, t)
    scene = trimesh.Scene(meshes)
    if show:
        scene.show()
    if save:
        from render_utils import show_and_save
        show_and_save(scene, img_name='tray_scene.png')
    return scene


def add_shape(shape, size, height, color=CLOUD, **kwargs):
    if shape == 'square':
        mesh = box(extents=[size, size, height], **kwargs)
    elif shape == 'box':
        mesh = box(extents=[size[0], size[1], height], **kwargs)
    elif shape == 'circle':
        mesh = cylinder(radius=size / 2, height=height, **kwargs)
    elif shape == 'triangle':
        mesh = triangle(size=size, height=height, **kwargs)
    elif shape == 'parallelogram':
        mesh = parallelogram(size=size, height=height, **kwargs)
    else:
        assert "what's this shape?"
        return None
    mesh.visual.vertex_colors = color
    return mesh


def regions_to_meshes(regions, width, length, height,
                      max_offset=0.2, min_offset_perc=0.1):
    """ convert 2D regions [top, left, width, height] to 3D meshes centered at origin """

    meshes = []
    used_colors = []
    # print("regions:", len(regions), len(RAINBOW_COLORS))
    for region in regions:
        if len(region) == 4:
            x, y, w, l = region
            z = 0
            h = height
        elif len(region) == 6:
            x, y, z, w, l, h = region
        else:
            assert "what's this region?"
            continue
        ps = np.random.uniform(max_offset*min_offset_perc, max_offset, 4)  ## padding [top, left, bottom, right]
        if w <= ps[1]+ps[3] or l <= ps[0]+ps[2]:
            continue
        w -= (ps[1]+ps[3])
        x += ps[1]
        l -= (ps[0]+ps[2])
        y += ps[0]
        mesh = box(extents=[w, l, h], transform=T([-width/2+x+w/2, -length/2+y+l/2, z+h/2]))

        color = get_color(used=used_colors, random_color=False)
        used_colors.append(tuple(color))
        mesh.visual.vertex_colors = color
        mesh.metadata['label'] = f"tile_{len(meshes)}"
        meshes.append(mesh)
    return meshes


def load_panda_meshes(pose):
    import open3d as o3d
    import transformations as tf
    pcd = o3d.io.read_point_cloud(PANDA_GRIPPER_MESH_PATH)
    mesh = trimesh.points.PointCloud(pcd.points)
    T = tf.translation_matrix(pose[0])
    R = tf.quaternion_matrix(pose[1])
    M = tf.concatenate_matrices(T, R)
    mesh.apply_transform(M)
    mesh.visual.vertex_colors = GREY
    mesh.metadata['label'] = f"floating_gripper"
    return mesh


def assets_to_meshes(models):
    import open3d as o3d
    import transformations as tf
    from assets import get_pointcloud_path
    meshes = []
    used_colors = []
    for asset in models:
        (cat, model_id), scale, extent, pose, theta = asset

        pcd = o3d.io.read_point_cloud(get_pointcloud_path(cat, model_id))
        mesh = trimesh.points.PointCloud(pcd.points)

        # print('\n', mesh.bounds, pose[0][2], scale)
        T = tf.translation_matrix(pose[0])
        R = tf.quaternion_matrix(pose[1])
        S = tf.scale_matrix(scale)  ## scale
        M = tf.concatenate_matrices(T, R, S)
        mesh.apply_transform(M)
        # mesh.apply_transform(tf.translation_matrix((0, 0, -mesh.bounds[0][2])))

        color = get_color(used=used_colors, random_color=False, colors=DARKER_COLORS)
        used_colors.append(tuple(color))
        mesh.visual.vertex_colors = color
        mesh.metadata['label'] = f"tile_{len(meshes)}[{cat}_{model_id}]"
        meshes.append(mesh)
    return meshes


def reorganize_points(lengths):
    index_shortest = np.argmin(lengths)
    index_longest = np.argmax(lengths)
    index_middle = [i for i in range(3) if i != index_shortest and i != index_longest][0]
    return [index_shortest, index_middle, index_longest]


def reorganize_points_2(lengths):
    index_shortest = np.argmin(lengths)
    index_longest = np.argmax(lengths)
    index_middle = [i for i in range(3) if i != index_shortest and i != index_longest][0]
    return [index_longest, index_middle, index_shortest]


def triangles_to_meshes(triangles, height, triangles_recentered=None, show_orientation=False):
    """ convert triangles to 3D meshes
    """

    meshes = []
    used_colors = []
    for i, (tri, lengths) in enumerate(triangles):
        z = 0 ## 0.5 * i if show_orientation else 0
        mesh = triangle(tri, height, transform=None, z=z)
        color = get_color(used=used_colors, random_color=False)
        used_colors.append(tuple(color))
        mesh.visual.vertex_colors = color
        mesh.metadata['label'] = f"tile_{len(meshes)}"
        meshes.append(mesh)

        if show_orientation and triangles_recentered is not None:
            p1, p2, p3 = triangles_recentered[i]
            mesh = triangle([p1, p2, p3], height, transform=None, z=z)
            mesh.visual.vertex_colors = CLOUD
            mesh.metadata['label'] = f"tile_shadow_{len(meshes)}"
            meshes.append(mesh)

    return meshes


def get_color_gradient(n=10, end=2/3):
    from colorsys import hls_to_rgb
    return [hls_to_rgb(end * i/(n-1), 0.5, 1) for i in range(n)]


def create_grid_meshes(width, length, height, grid_size=1, verbose=False):
    """ create a grid of boxes """
    meshes = []
    g = grid_size
    dot_size = 0.04 * g
    cols = int(width / g)
    rows = int(length / g)
    n = cols * rows
    colors = get_color_gradient(n=n)
    color_assignments = {}
    for j in range(rows):
        for i in range(cols):
            t = [-width/2+i*g+g/2, length/2-j*g-g/2, height/2 + 0.1]
            mesh = cylinder(dot_size/2, height, transform=T(t))
            index = len(meshes)
            if verbose:
                print('create_grid_meshes\t', index, '\t', t[:2])
            color_assignments[(j, i)] = colors[index]
            mesh.visual.vertex_colors = colors[index]
            mesh.metadata['label'] = 'grid_{}'.format(index)
            meshes.append(mesh)
    return meshes, color_assignments


def test_create_mesh(shape='triangle', size=1, height=0.5, mesh_name='triangle_000.obj',
                     show=False, save=False, show_axis=True, **kwargs):
    mesh = add_shape(shape, size, height, **kwargs)
    # mesh = triangle(size=1, height=0.5, transform=T([0, 0, 0]))
    if show:
        if show_axis:
            axis_marker = axis(origin_size=0.05)
            trimesh.Scene([axis_marker, mesh]).show()
        else:
            mesh.show()

    ## save the mesh as obj file
    if save:
        return save_mesh(mesh, mesh_name)


def save_mesh(mesh, mesh_name, view=False):
    e = mesh.export(file_type='obj')
    if '/' in mesh_name:
        mesh_file = mesh_name
    else:
        mesh_file = join(MESH_PATH, mesh_name)
    with open(mesh_file, 'w') as f:
        f.write(e)
        # print('Saved mesh to {}'.format(mesh_file))
    if view:
        view_mesh(mesh_file)
    return mesh_file


def get_area(x1, y1, x2, y2, x3, y3):
    return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0)


def is_inside(x1, y1, x2, y2, x3, y3, x, y, A_ABC=None):
    """ if point P(x,y) is inside triangle ABC """
    if A_ABC is None:
        A_ABC = get_area(x1, y1, x2, y2, x3, y3)
    A_PBC = get_area(x, y, x2, y2, x3, y3)
    A_PAC = get_area(x1, y1, x, y, x3, y3)
    A_PAB = get_area(x1, y1, x2, y2, x, y)
    return A_ABC == A_PBC + A_PAC + A_PAB


if __name__ == "__main__":
    r = Rotation2D(90)
    mesh_file = test_create_mesh(shape='parallelogram', show=True, transform=r)
    # view_mesh(mesh_file)

    # test_tray_scene(show=True)

    # init_blocks_by_cutting_plane(3, 2, 0.001)
