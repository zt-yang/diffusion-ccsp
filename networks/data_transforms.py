import torch
from torch_geometric.data import Data
import numpy as np
from os.path import join, isfile

from data_utils import print_tensor, get_one_hot, r, get_grasp_side_from_grasp, \
    get_ont_hot_grasp_side
from denoise_fn import robot_constraints, puzzle_constraints, \
    stability_constraints, qualitative_constraints, robot_qualitative_constraints


####################################################################################################


def pre_transform(data, data_idx, input_mode, debug_mode=0, **kwargs):
    if 'diffuse_pairwise' in input_mode or 'robot' in input_mode or 'stability' in input_mode \
            or 'qualitative' in input_mode:
        return data_transform_cn_diffuse_batch(data, data_idx, input_mode, **kwargs)
    if input_mode == 'collisions':
        return data_transform_cn_graph(data, data_idx, **kwargs)
    return data_transform_pose_gen(data, data_idx, input_mode, debug_mode=debug_mode)


####################################################################################################

def data_transform_cn_diffuse_batch(data, data_idx, input_mode, dir_name=None, visualize=False, verbose=False):
    """
        excluding the first feature on type [0 (container) / 1 (tiles)]
        for BoxWorld: each object has 4 features
            var node (1) obj :     n = (w, l)
            var node (2) pose :    n = (x, y)
        for TriangularWorld: each object has 7 features
            var node (1) obj :     n = (l, x3, y3)
            var node (2) pose :    n = (x1, y1, cs, sn)
        for TriangularWorld: each object has 10 features
            var node (1) obj :     n = (x1, y1, x2, y2, x3, y3)
            var node (2) pose :    n = (x, y, cs, sn)
        for TableToBoxWorld: each object has 13 features
            var node (1) obj :     n = (w, l, w0, l0)
            var node (2) pose :    n = (x, y, x0, y0)
            var node (3) grasp :   n = (sx+, sy+, sx-, sy-, sz+)
    """

    features = []
    conditioned_variables = [0] ## don't change the pose of tray
    all_constraints = puzzle_constraints

    ## --------------------- normalization --------------------- ##
    if verbose: print()
    w_tray, l_tray = data.x[0, 1:3].clone().detach()
    w_tray = w_tray.item()
    l_tray = l_tray.item()
    world_dims = (w_tray, l_tray)
    for i in range(len(data.x)):
        dd = data.x[i].tolist()

        if len(dd) == 5:
            typ, w, l, x, y = dd
            w /= w_tray
            l /= l_tray
            x /= (w_tray / 2)
            y /= (l_tray / 2)
            geom = [w, l]
            pose = [x, y]

        elif len(dd) == 7:

            ## triangle P1 encoding with theta
            if 'diffuse_pairwise' in input_mode:
                if dd[0] == 0:
                    typ, w, l, _, x, y, _ = dd
                    w /= w_tray
                    l /= l_tray
                    geom = [w, l, 0]
                    pose = [x, y, 0]
                else:
                    typ, l, x3, y3, x1, y1, r1 = dd
                    l /= w_tray
                    x3 /= w_tray
                    y3 /= l_tray
                    x1 /= (w_tray / 2)
                    y1 /= (l_tray / 2)
                    r1 /= np.pi
                    geom = [l, x3, y3]
                    pose = [x1, y1, r1]

            ## box encoding with sin/cos (stability, qualitative)
            else:

                if dd[0] == 0:
                    typ, w, l, x, y, _, _ = dd
                    w /= w_tray
                    l /= l_tray
                    geom = [w, l]
                    pose = [x, y, 0, 0]
                else:
                    if 'stability' in input_mode:
                        all_constraints = stability_constraints
                        geom = dd[1:3]
                        pose = dd[3:]
                    elif 'qualitative' in input_mode:
                        all_constraints = qualitative_constraints
                        _, w, l, x, y, sn, cs = dd
                        w /= w_tray
                        l /= l_tray
                        x /= (w_tray / 2)
                        y /= (l_tray / 2)
                        geom = [w, l]
                        pose = [x, y, cs, sn]

        ## triangle P1 encoding with sin/cos
        elif len(dd) == 8 or len(dd) == 8 + 32 ** 2 or len(dd) == 8 + 64 ** 2:
            if dd[0] == 0:
                typ, w, l, _, x, y, _, _ = dd[:8]
                w /= w_tray
                l /= l_tray
                geom = [w, l, 0]
                pose = [x, y, 0, 0]
            else:
                typ, l, x3, y3, x1, y1, cs, sn = dd[:8]
                l /= w_tray
                x3 /= w_tray
                y3 /= l_tray
                x1 /= (w_tray / 2)
                y1 /= (l_tray / 2)
                geom = [l, x3, y3]
                pose = [x1, y1, cs, sn]

            if len(dd) != 8:
                pose += dd[8:]

        ## triangle centroid encoding with sin/cos
        elif len(dd) == 11 or len(dd) == 11 + 32 ** 2 or len(dd) == 11 + 64 ** 2:
            if dd[0] == 0:
                typ, w, l, _, _, _, _ = dd[:7]
                x, y, _, _ = dd[-4:]
                w /= w_tray
                l /= l_tray
                geom = [w, l, 0, 0, 0, 0]
                pose = [x, y, 0, 0]
            else:
                typ, x1, y1, x2, y2, x3, y3 = dd[:7]
                x, y, cs, sn = dd[-4:]
                x1 /= w_tray
                x2 /= w_tray
                x3 /= w_tray
                x /= w_tray
                y1 /= l_tray
                y2 /= l_tray
                y3 /= l_tray
                y /= l_tray
                geom = [x1, y1, x2, y2, x3, y3]
                pose = [x, y, cs, sn]

            if len(dd) != 11:
                pose = dd[7:-4] + pose

        ## robot, pose, grasp
        elif len(dd) in [22, 22+7, 22+7+7]:
            geom = dd[1:9]
            pose = dd[9:]
            all_constraints = robot_constraints
            world_dims = geom[3:5]

            if 'robot' in input_mode and 'qualitative' in input_mode:
                all_constraints = robot_qualitative_constraints

        feature = geom + pose
        if verbose:
            print(f'feature {i}: {r(feature)}')
        features.append(feature)

    ## for each edge, add one constraint node, and add edge_attr
    edge_attr = [all_constraints.index(elems[0]) for elems in data.edge_index]
    edge_index = [elems[1:] for elems in data.edge_index]

    if visualize:
        draw_constraint_network_df_batch(features, edge_index, edge_attr, conditioned_variables, input_mode,
                                         name=f"idx={data_idx}_cn", dir_name=dir_name, save_png=True)

    features = torch.tensor(np.stack([np.asarray(n) for n in features]), dtype=torch.float)
    mask = torch.zeros(features.shape[0])
    mask[conditioned_variables] = 1

    x = torch.tensor(np.stack([np.asarray(n) for n in features]), dtype=torch.float)
    mask = torch.tensor(np.stack([np.asarray(n) for n in mask]), dtype=torch.int8)
    conditioned_variables = torch.tensor(
        np.stack([np.asarray(n) for n in conditioned_variables]), dtype=torch.int8)
    edge_index = torch.tensor(np.stack([np.asarray(n) for n in edge_index]), dtype=torch.int64).T
    edge_attr = torch.tensor(np.stack([np.asarray(n) for n in edge_attr]), dtype=torch.float)

    ## the dataset is biased with the order of the data, so shuffle it
    shuffled = torch.randperm(x.shape[0])

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, shuffled=shuffled,
                conditioned_variables=conditioned_variables, mask=mask,
                x_extract=torch.ones(x.shape[0])*data_idx,
                edge_extract=torch.ones(edge_index.shape[1])*data_idx,
                world_dims=world_dims, original_x=data.x, original_y=data.y)
    return [data]


def robot_data_json_to_pt(data, data_sub_dir_name, qualitative_constraints=[]):
    """
    for TableToBoxWorld: each object has 21 + 7 + 7 features
        var node obj (10) :     n = (w, l, h, w0, l0, h0, x0, y0, |  mobility_id, scale) where h0 = 0.25
        var node grasp (6) :   n = (kx+, kx-, ky+, ky-, kz+, |  grasp_id) + (7) + (7)
        var node pose (5) :    n = (x, y, z, sin, cos)
    """
    import pybullet_planning as pp

    ## convert json to x features
    w0, l0 = data['container']['tray_dim'][:2]
    x0, y0, z0 = data['container']['tray_pose']
    # if np.linalg.norm(data['container']['rot'] % np.pi) < 1e-3:
    #     w0, l0 = data['container']['tray_dim'][:2]
    #     x0, y0, z0 = data['container']['tray_pose']
    # else:
    #     l0, w0 = data['container']['tray_dim'][:2]
    #     x0, y0, z0 = data['container']['tray_pose']
    h0 = 0.25

    features = [[0] + [1, 1, 0, w0, l0, h0, x0, y0, 0, 0]
                + [0, 0, 0, 0, 0, 0]
                + [0, 0, 0, 0, 0]
                + [0] * 7 # + [0] * 7
                ]
    for i, obj in enumerate(data['placements']):
        grasp_id = obj['grasp_id']
        scale = obj['scale']
        w, l, h = obj['extent']
        if 'place_pose' in obj:  ## for debugging
            x, y, z = obj['place_pose'][0]
            x = (x - x0) / w0 * 2
            y = (y - y0) / l0 * 2
            z = z / h0
            yaw = pp.euler_from_quat(obj['place_pose'][1])[2]
            mobility_id = int(obj['name'].split('_')[1])
        else:
            x, y, z, yaw = 0, 0, 0, 0
            mobility_id = data['stats']['scene_id']
        sn = np.sin(yaw)
        cs = np.cos(yaw)
        if 'grasp_side' in obj:
            grasp_side = obj['grasp_side']
        else:
            grasp_side = get_grasp_side_from_grasp(obj['name'], obj['grasp_pose'][1])
        grasp_side = get_ont_hot_grasp_side(grasp_side)

        ## for debugging trimesh reconstruction, no longer needed, some data don't have hand_pose saved
        # hand_pose = (np.array(obj['hand_pose'][0]) - np.array([x0, y0, 0])).tolist() + obj['hand_pose'][1]
        # grasp_pose = obj['grasp_pose'][0] + obj['grasp_pose'][1]

        pick_pose = obj['pick_pose'][0] + obj['pick_pose'][1]
        features.append([1] + [w/w0, l/l0, h/h0, w0, l0, h0, x0, y0, mobility_id, scale]
                        + grasp_side + [grasp_id]
                        + [x, y, z, sn, cs]
                        + pick_pose # + hand_pose + grasp_pose  ## passing hand_pose and grasp_pose for debugging
                        )

    ## generate constraints
    edge_index = [('gin', i, 0) for i in range(1, len(features))]
    for i in range(1, len(features)):
        for j in range(i+1, len(features)):
            edge_index.append(('gfree', j, i))

    edge_index += qualitative_constraints

    return Data(x=torch.tensor(features, dtype=torch.float), edge_index=edge_index, y=data_sub_dir_name)


def stability_data_json_to_pt(data, data_sub_dir_name, input_mode):
    """
    for RandomSplitWorld: each object has 7 features
        var node obj (2) :     n = (w, l)
        var node pose (5) :    n = (x, y, z, sin, cos)
    """
    w0, l0 = data['container']['shelf_extent'][:2]
    x0, y0 = data['container']['shelf_pose'][:2]
    features = [[0] + [w0, l0] + [0, 0, 0, 0]]
    for i, obj in enumerate(data['placements']):
        w, l = obj['extents'][:2]
        x, y = obj['centroid'][:2]
        x = (x - x0) / w0 * 2
        y = (y - y0) / l0 * 2
        yaw = obj['theta']
        if 'flat' in input_mode and w > l:
            l, w = obj['extents'][:2]
            yaw = yaw + np.pi / 2

        sn = np.sin(yaw)
        cs = np.cos(yaw)
        f = [w/w0, l/l0, x, y, sn, cs]
        features.append([1] + f)

    ## generate constraints
    edge_index = [('within', i, 0) for i in range(1, len(features))]
    edge_index += [('supportedby', i, j) for i, j in data['supports']]
    for i in range(1, len(features)):
        for j in range(i+1, len(features)):
            if ('supportedby', i, j) not in edge_index and ('supportedby', j, i) not in edge_index:
                edge_index.append(('cfree', i, j))
    return Data(x=torch.tensor(features, dtype=torch.float), edge_index=edge_index, y=data_sub_dir_name)


def draw_constraint_network_df_batch(features, edge_index, edge_attr, conditioned_variables, input_mode,
                                     evaluations=None, name=f"cn", dir_name='', save_png=True):
    import graphviz

    def n(inputs):
        if isinstance(inputs, torch.Tensor):
            return inputs.tolist()
        return inputs

    file_name = join('renders', dir_name, name) if dir_name is not None else name
    features = n(features)

    node_names = {}
    colors = ['orange', 'purple', 'green', 'blue', 'yellow', 'gray', 'cyan', 'black']
    if 'robot' in input_mode:
        notes = {'obj': ['w', 'l', 'h', 'w0', 'l0', 'h0', 'x0', 'y0', 'id', 'scale'],
                 'grasp': ['x+', 'x-', 'y+', 'y-', 'z+', 'id'],
                 'pose': ['x', 'y', 'z', 'sin', 'cos']}
        indices = {'obj': [0, 10], 'grasp': [10, 16], 'pose': [16, 21]}
        all_constraints = robot_constraints
    elif 'stability' in input_mode:
        notes = {'obj': ['w', 'l'],
                 'pose': ['x', 'y', 'sin', 'cos']}
        indices = {'obj': [0, 2], 'pose': [2, 6]}
        all_constraints = stability_constraints
    else:
        notes = {'obj': ['w', 'l'], 'pose': ['x', 'y']}
        indices = {'obj': [0, 2], 'pose': [2, 4]}
        all_constraints = puzzle_constraints
    con_indices = {i: 0 for i in range(len(all_constraints))}

    def get_node_text(i, note):
        node = []
        b, e = indices[note]
        for n in features[i][b: e]:
            if n // 1 == n:
                node.append(int(n))
            else:
                node.append(round(n, 2))
        nnote = str(notes[note]).replace("'", '')
        node_name = f"{note}{i}\n{nnote}\n{node}"
        if (i, typ) not in node_names:
            node_names[(i, typ)] = node_name
        return node_name

    f = graphviz.Digraph(name, engine='sfdp', filename=join(f'{name}.gv'))
    for k, (typ, shape) in enumerate([('obj', 'doubleoctagon'), ('grasp', 'ellipse'), ('pose', 'octagon')]):
        if typ not in notes:
            continue
        f.attr('node', shape=shape)
        for i in range(len(features)):
            n = get_node_text(i, typ)
            if n.startswith('grasp0'):
                continue
            if i in conditioned_variables or k in [0, 2]:
                f.node(n, color='red', fillcolor='lightgrey', style='filled')
            else:
                f.node(n, color='red')
    f.attr('node', shape='rectangle')

    ## draw edges (from var to con) and edge labels
    types = list(notes.keys())
    for i in range(len(edge_index)):
        con = edge_attr[i]
        des = f"{all_constraints[con]}{con_indices[con]}"
        con_indices[con] += 1
        color = colors[con]
        f.node(des, color=color)
        label = 0
        for j in range(len(edge_index[i])):
            elem = edge_index[i][j]
            for k in range(len(types)):
                ## only uses grasp in the 1st arg of 'in' and 2nd arg of 'gfree'
                if k == 2 and j == 1:
                    continue
                src = node_names[(elem, types[k])]
                f.edge(src, des, label=str(label), color=color)  ## label=str(2*j+k)
                label += 1

    if save_png:
        f.render(file_name, format='png', view=False)
    else:
        f.view()


def draw_constraint_network_df(features, constraints, conditioned_variables, nodes_info, con_labels,
                               evaluations=None, name=f"cn", dir_name='', save_png=True):
    import graphviz
    from denoise_fn import puzzle_constraints

    def n(inputs):
        if isinstance(inputs, torch.Tensor):
            return inputs.tolist()
        return inputs

    file_name = join('renders', dir_name, name) if dir_name is not None else name
    if isfile(file_name):
        return
    features = n(features)
    node_labels, (obj_indices, pose_indices) = nodes_info
    node_names = {}
    colors = ['orange', 'purple', 'green', 'blue',
              'yellow', 'gray', 'cyan', 'black']
    colors = {puzzle_constraints[i]: colors[i] for i in range(len(puzzle_constraints))}
    notes = {
        'obj': ['w', 'l'],
        'pose': ['x', 'y'],
    }

    def get_node_text(i, note):
        node = []
        for n in features[i]:
            if n // 1 == n:
                node.append(int(n))
            else:
                node.append(round(n, 2))
        nnote = str(notes[note]).replace("'", '')
        node_name = f"{node_labels[i]}\n{nnote}\n{node}"
        if node_labels[i] not in node_names:
            node_names[node_labels[i]] = node_name
        return node_name

    f = graphviz.Digraph(name, engine='sfdp', filename=join(f'{name}.gv'))
    for shape, indices, typ in [('doublecircle', obj_indices, 'obj'), ('circle', pose_indices, 'pose')]:
        f.attr('node', shape=shape)
        for i in range(len(node_labels)):
            if i in indices:
                n = get_node_text(i, typ)
                if i in conditioned_variables:
                    f.node(n, color='red', fillcolor='lightgrey', style='filled')
                else:
                    f.node(n, color='red')
    f.attr('node', shape='rectangle')

    ## draw edges (from var to con) and edge labels
    for i in range(len(constraints)):
        c = constraints[i]
        des = con_labels[i]
        color = colors[c.name]
        f.node(des, color=color)
        for j in range(len(c.args)):
            a = c.args[j]
            src = node_names[node_labels[a]]
            f.edge(src, des, label=str(j), color=color)

    if save_png:
        f.render(file_name, format='png', view=False)
    else:
        f.view()


def data_transform_cn_graph(data, data_idx, dir_name=None, visualize=True):
    """  constraints = ['oftype', 'atpose', 'cfree', 'in']
    var node (1) type :    n = (is_m, is_c, 0, 0, 0, 0)     x MLP[VAR]     -> len(32)
    var node (2) obj :     n = (0, 0, w, l, 0, 0)           x MLP[VAR]     -> len(32)
    var node (3) pose :    n = (0, 0, 0, 0, x, y)           x MLP[VAR]     -> len(32)
    con node [A] oftype :  n = one-hot(3)       x MLP[CON]     -> len(32)
    con node [B] cfree :   n = one-hot(3)       x MLP[CON]     -> len(32)
    con node [C] in :      n = one-hot(3)       x MLP[CON]     -> len(32)
    con-var edge :         e = one-hot(2)       x MLP[{CON}]   -> len(8)
    """
    node_dim = 6
    edge_dim = 6  ## cfree

    nodes = []
    node_labels = []
    edge_index = []
    edge_attr = []
    edge_labels = []
    labels = []

    type_indices = [0, 1]  ## [0, 1] for collision detection | [0, 1, 2] for pose gen
    obj_indices = []
    pose_indices = []
    con_indices = []
    collision_indices = []

    def get_edge_attr(n):
        result = [[0] * edge_dim for i in range(n)]
        for i in range(n):
            result[i][i] = 1
        return result

    def get_edge_labels(n):
        return [str(i) for i in range(n)]

    def get_con_label(typ):
        s = typ + str(con_counts[typ])
        con_counts[typ] += 1
        return s

    constraints = ['oftype', 'atpose', 'cfree', 'in']
    con_features = get_edge_attr(4)
    con_features = {constraints[i]: con_features[i] for i in range(4)}
    con_counts = {constraints[i]: 0 for i in range(4)}

    ## add two types node
    nodes.extend([n + [0] * 4 for n in [[1, 0], [0, 1]]])
    node_labels.extend(['objtype=0', 'objtype=1'])

    ## for each x, add one obj node and one pose node
    w_tray, l_tray = data.x[0, 1:3].clone().detach()
    w_tray = w_tray.item()
    l_tray = l_tray.item()
    for i in range(len(data.x)):
        typ, w, l, x, y = data.x[i].tolist()
        w /= w_tray
        l /= l_tray
        x /= (w_tray / 2)
        y /= (l_tray / 2)

        ## obj var node, + oftype con node, + edge to (obj, type) node
        idx = len(nodes)
        nodes.append([0, 0, w, l, 0, 0])
        obj_indices.append(idx)
        node_labels.extend([f'o{len(obj_indices)}'])

        nodes.append(con_features['oftype'])
        node_labels.append(get_con_label('oftype'))
        con_indices.append(idx + 1)

        edge_index.extend([[type_indices[i != 0], idx + 1], [idx, idx + 1]])
        edge_attr.extend(get_edge_attr(2))
        edge_labels.extend(get_edge_labels(2))

        ## pose var node, + atpose con node, + edge to (obj, pose) node
        nodes.append([0, 0, 0, 0, x, y])
        pose_indices.append(idx + 2)
        node_labels.extend([f'p{len(pose_indices)}'])

        nodes.append(con_features['atpose'])
        node_labels.append(get_con_label('atpose'))
        con_indices.append(idx + 3)

        edge_index.extend([[idx, idx + 3], [idx + 2, idx + 3]])
        edge_attr.extend(get_edge_attr(2))
        edge_labels.extend(get_edge_labels(2))

    ## 0 means excluding from cost computation
    labels = [[0, -1]] * len(nodes)

    ## for each edge, add one constraint node, and add edge_attr
    for i in range(len(data.x)):
        for j in range(i + 1, len(data.x)):
            idx = len(nodes)
            con = 'in' if i == 0 else 'cfree'
            nodes.append(con_features[con])  ## o1, p1, o2, p2
            node_labels.append(get_con_label(con))
            con_indices.append(idx)

            edge_index.extend([
                [obj_indices[i], idx],
                [pose_indices[i], idx],
                [obj_indices[j], idx],
                [pose_indices[j], idx],
            ])
            labels.append([1, int(data.y[i, j] == 1)])
            edge_attr.extend(get_edge_attr(4))
            edge_labels.extend(get_edge_labels(4))

    nodes_info = node_labels, (obj_indices, pose_indices, type_indices, con_indices)
    if visualize:
        draw_constraint_network(nodes_info, nodes, edge_index, edge_labels, predictions=labels,
                                name=f"idx={data_idx}_cn", dir_name=dir_name, save_png=True)

    ## 1 if var nodes, 0 if constraint nodes
    var_indices = obj_indices + pose_indices + type_indices
    typed_nodes = []
    for i in range(len(nodes)):
        typed_nodes.append(nodes[i] + [int(i in var_indices)])

    x = torch.tensor(np.stack([np.asarray(n) for n in typed_nodes]), dtype=torch.float)
    y = torch.tensor(np.stack([np.asarray(n) for n in labels]), dtype=torch.int8)
    edge_index = torch.tensor(np.stack([np.asarray(n) for n in edge_index]), dtype=torch.int64).T
    edge_attr = torch.tensor(np.stack([np.asarray(n) for n in edge_attr]), dtype=torch.float)
    data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr,
                note=np.asarray([data_idx, 0]), original_x=data.x,
                nodes_info=nodes_info, edge_labels=edge_labels)
    return [data]


def draw_constraint_network(nodes_info, nodes, edge_index, edge_labels, predictions=None,
                            save_png=False, name='constraint_network', dir_name=None):
    import graphviz
    file_name = join('renders', dir_name, name) if dir_name is not None else name
    node_labels, (obj_indices, pose_indices, type_indices, con_indices) = nodes_info
    node_names = {}
    colors = {
        'oftype': 'green',
        'atpose': 'orange',
        'cfree': 'purple',
        'in': 'blue',
    }
    notes = {
        'type': ['m', 'c', '_', '_', '_', '_'],
        'obj': ['_', '_', 'w', 'l', '_', '_'],
        'pose': ['_', '_', '_', '_', 'x', 'y'],
        'con': ['is', 'at', 'cf', 'in', '_', '_'],
    }

    def get_pred_name(name):
        return ''.join([c for c in name if not c.isdigit()])

    def get_node_text(i,  note):
        node = []
        for n in nodes[i]:
            if n // 1 == n:
                node.append(int(n))
            else:
                node.append(round(n, 2))
        nnote = str(notes[note]).replace("'", '')
        node_name = f"{node_labels[i]}\n{nnote}\n{node}"
        if node_labels[i] not in node_names:
            node_names[node_labels[i]] = node_name
        return node_name

    f = graphviz.Digraph(name, engine='sfdp', filename=join(f'{name}.gv'))
    f.attr('node', shape='doublecircle')
    [f.node(get_node_text(i, 'obj'), color='red') for i in range(len(node_labels)) if i in obj_indices]
    f.attr('node', shape='circle')
    [f.node(get_node_text(i, 'pose'), color='red') for i in range(len(node_labels)) if i in pose_indices]
    [f.node(get_node_text(i, 'type'), color='black') for i in range(len(node_labels)) if i in type_indices]
    f.attr('node', shape='diamond')

    ## draw edges (from var to con) and edge labels
    for i in range(len(edge_index)):
        # import ipdb; ipdb.set_trace()
        src = node_names[node_labels[edge_index[i][0]]]
        des = node_labels[edge_index[i][1]]
        color = colors[get_pred_name(des)]
        des = get_node_text(node_labels.index(des), 'con')
        f.node(des, color=color)
        label = edge_labels[i]
        f.edge(src, des, label=label, color=color)

    ## color fill for violated constraints
    if predictions is not None:
        for i in range(len(predictions)):
            prediction = predictions[i]
            if prediction == [1, 1]:
                # print('violated constraint: ', node_labels[i])
                node_name = get_node_text(i, 'con')
                color = colors[get_pred_name(node_labels[i])]
                f.node(node_name, color=color, style='filled', fillcolor=color, fontcolor='white')

    if save_png:
        f.render(file_name, format='png', view=False)
    else:
        f.view()


def draw_cn_while_training(dataset, epoch, render_dir, count_limit=100):
    from torch_geometric.data import Data
    from data_transforms import draw_constraint_network
    from tqdm import tqdm
    count = 0
    for data, logits, y_prediction, mask in tqdm(dataset, desc='generating var-con graphs'):
        idx, _ = data.note[0]
        nodes = data.x.detach().cpu()
        predictions = torch.stack([mask, y_prediction])
        print_tensor('predictions', predictions)
        print_tensor('data.y', data.y.T)
        new_data = Data(
            x=nodes,
            edge_index=data.edge_index,
            y=data.y,
            predictions=(logits, predictions.T)
        )
        output_name = f"idx={idx}_v=0_epoch={epoch}"
        output_path = join(render_dir, output_name)
        torch.save(new_data, output_path + '.dt')
        draw_constraint_network(data.nodes_info[0], nodes.numpy(), data.edge_index.T.tolist(), data.edge_labels[0],
                                predictions=predictions.T, name=output_name, dir_name=render_dir, save_png=True)

        count += 1
        if count > count_limit:
            break


#######################################################################################


def data_transform_pose_gen(data, idx, input_mode, debug_mode=0):
    """
    input_mode:
        grid_offset:      x = (type, width, length) -> y = (grid, dx, dy) only one correct
                          label = (grid, dx, dy) only one correct set
        grid_offset_moo:  x = (type, width, length, grid, dx, dy) -> x' Multiple Offsets choose One (MOO)
                          label = (grid [1], offsets [n_grid x 2])
        grif_offset_mp4:  x = (type, width, length, grid, dx, dy) -> x' Multiple Pair choose Four (MP4)
                          label = (grid [1], offsets [n_grid x 2])
        grif_offset_oh4:  x = ([type-one-hot], width, length, [x, y]) -> x' Multiple Pair choose Four (MP4)
                          label = (grid-one-hot [24], offsets [n_grid x 2])
        cn:
            var node (1) type :    n = (is_m, is_c, 0, 0, 0, 0)     x MLP[VAR]     -> len(32)
            var node (2) obj :     n = (0, 0, w, l, 0, 0)           x MLP[VAR]     -> len(32)
            var node (3) pose :    n = (0, 0, 0, 0, x, y)           x MLP[VAR]     -> len(32)
            con node [A] oftype(2) :  n = one-hot(6)       x MLP[CON]     -> len(32)
            con node [B] atpose(2) :  n = one-hot(6)       x MLP[CON]     -> len(32)
            con node [C] cfree(4) :   n = one-hot(6)       x MLP[CON]     -> len(32)
            con node [D] in(4) :      n = one-hot(6)       x MLP[CON]     -> len(32)
            con-var edge :         e = one-hot(4)       x MLP[{CON}]   -> len(8)
    """
    if input_mode == 'grid_offset':
        return [data]

    if data.x.shape[0] <= 2:
        print(f'!!! only one node ')
        return None

    data_list = []
    if input_mode == 'grid_offset_moo':
        y = data.y
        nodes = torch.cat([data.x, y], dim=1)

    elif input_mode == 'grid_offset_mp4':
        y = data.y[:, 0]
        nodes = torch.cat([data.x, y], dim=1)

    elif input_mode == 'grid_offset_oh4':
        """ moved to model.forward() """

        print_tensor('data.x', data.x)

        ## normalize width and length according to tray side
        w_tray, l_tray = torch.clone(data.x[0, 1:3])
        data.x[:, 1] /= w_tray
        data.x[:, 2] /= l_tray
        data.x[:, 3] /= (w_tray/ 2)
        data.x[:, 4] /= (l_tray/ 2)
        print_tensor('normalized data.x', data.x)

        ## one-hot encoding for grid index
        grid_size = 0.5
        n_types = 24 + 1
        original_y = data.y[:, 0, :]
        grids = data.y[:, :, 0]
        n_shape, n_options = grids.shape
        y_inputs = list(grids.reshape(-1))
        vocab = list(range(-1, n_types - 1))
        one_hot = get_one_hot(y_inputs, vocab, n_types).reshape((n_shape, n_options, n_types))  ## (5, 4, 25)
        data.y = torch.cat([torch.tensor(one_hot), data.y[:, :, 1:]/grid_size], dim=2)

        nodes = data.x

    else:
        y = data.y
        nodes = data.x

    if debug_mode == 2:
        print(f'\n-------- {idx} ({len(nodes) - 1}) --------\n', nodes)

    for i in range(1, len(nodes)):
        masked_x = nodes.clone()
        masked_x[i][0] = 2
        masked_x[i][3:] = torch.zeros_like(masked_x[i][3:])
        y = data.y[i]
        y_grid = y[0, 0].view(1)
        kwargs = dict(
            edge_index=data.edge_index,
            note=np.asarray([idx, i])
        )

        ## one-hot encoding for grid type
        if input_mode == 'grid_offset_oh4':
            kwargs.update(dict(
                original_x=masked_x, original_y=original_y,
            ))
            y_grid = grids[i, 0]
            n_types = 3
            x_inputs = masked_x[:, 0]
            n_shape, = x_inputs.shape
            vocab = list(range(n_types))
            one_hot = get_one_hot(x_inputs, vocab, n_types).reshape((n_shape, n_types))  ## (5, 4, 25)
            masked_x = torch.cat([torch.tensor(one_hot), masked_x[:, 1:]], dim=1)

        new_data = Data(x=masked_x, y=y, y_grid=y_grid, **kwargs)
        if debug_mode == 1:
            print_tensor('x', new_data.x)
            print_tensor('y', new_data.y)
            print_tensor('y_grid', new_data.y_grid)
            print_tensor('note', new_data.note)
            print()
        data_list.append(new_data)
    return data_list


def transform_one_hot(x, verbose=False):
    """ inside model.forward() for input mode grid_offset_oh4 """

    ## normalize width, length, x, y according to tray side
    w_tray, l_tray = x[0, 1:3].clone().detach()
    x[:, 1] /= w_tray
    x[:, 2] /= l_tray
    x[:, 3] /= (w_tray / 2)
    x[:, 4] /= (l_tray / 2)
    if verbose: print_tensor('normalized data.x', x)

    ## normalize dx, dy by grid_size
    grid_size = 0.5
    x[:, 7:8] /= grid_size

    n_shape, _ = x.shape

    ## change type encoding to one hot
    n_types = 3
    vocab = list(range(n_types))
    enc_type = get_one_hot(x[:, 0], vocab, n_types).reshape((n_shape, n_types))
    enc_type = torch.tensor(enc_type, device=x.device)

    ## change grid encoding to one hot
    n_types = 24
    vocab = list(range(n_types))
    enc_grid = get_one_hot(x[:, -3], vocab, n_types).reshape((n_shape, n_types))
    enc_grid = torch.tensor(enc_grid, device=x.device)

    # x = torch.cat([torch.tensor(enc_type), x[:, 1:5], torch.tensor(enc_grid), x[:, -2:]], dim=1)
    x = torch.cat([torch.tensor(enc_type), x[:, 1:5]], dim=1)
    if verbose: print_tensor('x after transform', x)
    return x.float()


def transform_fourier(x):
    x1 = torch.cat([torch.sin(x * (2 ** i) * 3.14) for i in range(4)], dim=-1)
    x2 = torch.cat([torch.cos(x * (2 ** i) * 3.14) for i in range(4)], dim=-1)
    x = torch.cat([x1, x2], dim=-1)
    return x
