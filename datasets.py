import json
from os.path import join, abspath, dirname, basename, isdir, isfile
import os
from os import listdir
import shutil
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset, InMemoryDataset, download_url

from data_utils import get_one_hot, print_tensor, render_world_from_graph, constraint_from_edge_attr
from data_transforms import pre_transform, robot_data_json_to_pt, stability_data_json_to_pt
from denoise_fn import qualitative_constraints

device = "cuda" if torch.cuda.is_available() else "cpu"

PROJECT_PATH = dirname(__file__)
DATASET_PATH = abspath(join(PROJECT_PATH, 'data'))
RENDER_PATH = abspath(join(PROJECT_PATH, 'renders'))
VISUALIZATION_PATH = abspath(join(PROJECT_PATH, 'visualizations'))


class GraphDataset(InMemoryDataset): ## Dataset | InMemoryDataset
    def __init__(self, dir_name, input_mode='grid_offset', transform=None, pre_transform=None,
                 pre_filter=None, debug_mode=2, visualize=False):
        self.input_mode = input_mode
        self.dir_name = dir_name
        self.debug_mode = debug_mode
        self.visualize = visualize
        root = join(DATASET_PATH, dir_name)

        processed = join(root, 'processed')
        self.composed_inference = False
        if self.input_mode == 'robot_qualitative' and ('test' in dir_name or 'robot_qualitative' in dir_name):
            """ need to run create_qualitative_data() in 3-panda-box-data.py on the dir_name first """
            processed = join(root, 'processed_robot_qualitative')
            self.composed_inference = True
        # if isdir(processed):
        #     shutil.rmtree(processed)
        self.spass = isfile(join(processed, 'data.pt')) and False
        self.length = eval(dir_name[dir_name.index('(')+1:dir_name.index(')')])
        if 'object_i=' in dir_name:
            self.length = 1
        self.is_json_data = 'robot' in input_mode or 'stability' in input_mode
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        if self.is_json_data:
            dataset_dir = join(DATASET_PATH, self.dir_name, 'raw')
            if not isdir(dataset_dir):
                return []
            data_dirs = listdir(dataset_dir)
            data_dirs = [d for d in data_dirs if isdir(join(dataset_dir, d))]
            data_dirs = sorted(data_dirs)
            return [join(d, f'solution.json') for d in data_dirs]
        else:
            return [f'data_{i}.pt' for i in range(self.length)]

    @property
    def processed_file_names(self):
        return ['data.pt']

    @property
    def processed_dir(self) -> str:
        processed = join(self.root, 'processed')
        if self.input_mode == 'robot_qualitative' and self.composed_inference:
            processed = join(self.root, 'processed_robot_qualitative')
        return processed

    def download(self):
        pass

    def process(self):

        if self.spass:
            return

        data_list = []

        for idx in tqdm(range(self.length)):

            if self.is_json_data:
                if idx > len(self.raw_paths) - 1:
                    print(f'{self.dir_name}\tlen(self.raw_paths) = {len(self.raw_paths)},\tidx = {idx}')
                data = json.load(open(self.raw_paths[idx], 'r'))
                data_sub_dir_name = self.raw_paths[idx].replace('/solution.json', '')
                data_sub_dir_name = basename(data_sub_dir_name)
                if 'robot' in self.input_mode:
                    qualitative_constraints = []
                    if 'qualitative' in self.input_mode and self.composed_inference:
                        con_file = self.raw_paths[idx].replace('/solution.json', '/qualitative_constraints.json')
                        qualitative_constraints = json.load(open(con_file, 'r'))['qualitative_constraints']
                    data = robot_data_json_to_pt(data, data_sub_dir_name, qualitative_constraints)
                elif 'stability' in self.input_mode:
                    data = stability_data_json_to_pt(data, data_sub_dir_name, self.input_mode)
            else:
                data = torch.load(self.raw_paths[idx])

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data, idx, input_mode=self.input_mode,
                                          dir_name=self.dir_name, visualize=self.visualize)
                if data is None:
                    continue

            data_list.extend(data)

        # print('\n'.join([str(data) for data in data_list]))
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    # def len(self):
    #     return len(self.processed_file_names)
    #
    # def get(self, idx):
    #     data = torch.load(join(self.processed_dir, f'data_{idx}.pt'))
    #     return data


def get_world_name(dir_name):
    world_name = dir_name[:dir_name.index('(')]
    if '[' in world_name:
        world_name = world_name[:world_name.index('[')]
    return world_name


def visualize_dataset(dir_name='RandomSplitWorld(1)_test', input_mode='diffuse_pairwise', visualize=True):
    if 'diffuse' in input_mode and input_mode not in dir_name:
        dir_name = dir_name.replace('_test', f'_{input_mode}_test').replace('_train', f'_{input_mode}_train')

    dataset = GraphDataset(dir_name, input_mode=input_mode, pre_transform=pre_transform,
                           debug_mode=2, visualize=visualize)

    png_dir = join(RENDER_PATH, dir_name)
    if not isdir(png_dir):
        os.mkdir(png_dir)

    length = min([10, len(dataset)])
    world_name = get_world_name(dir_name)
    all_features = []
    for i in tqdm(range(length)):
        # if i in [0, 1]:
        #     continue
        png_name = join(dir_name, f'idx={i}.png')
        if isfile(png_name):
            continue
        data = dataset[i]
        world_dims = data.world_dims
        if input_mode in ['grid_offset_oh4', 'collisions']:  ## , 'diffuse_pairwise'
            x, y = data.original_x, dataset[0].y
        elif 'diffuse_pairwise' in input_mode or 'robot' in input_mode or 'stability' in input_mode:
            x, y = data.x, data.original_y
        else:
            x, y = data.x, dataset[0].y
        all_features.append(x.detach().cpu().numpy())

        if 'robot' in input_mode:
            """ can use `render_world_from_graph()` too but won't be able to check collisions """
            from demo_utils import render_robot_world_from_graph
            prediction_json = join(png_dir, f'{y}_prediction.json')
            result = render_robot_world_from_graph(x, prediction_json)

        elif 'stability' in input_mode:
            from demo_utils import render_stability_world_from_graph
            prediction_json = join(png_dir, f'{y}_prediction.json')
            supports = data.edge_index.T[torch.where(data.edge_attr == 1)].T
            result = render_stability_world_from_graph(x, prediction_json, world_dims, supports)
            if result is None:
                data_dir = join(DATASET_PATH, dir_name, 'raw', y)
                # shutil.rmtree(data_dir)
                print('collided', data_dir)
                continue
            # print('rendering result\t', result)

        render_kwargs = dict(world_dims=world_dims, world_name=world_name, png_name=png_name, verbose=False, show=False)
        if 'qualitative' in input_mode:
            constraints = constraint_from_edge_attr(data.edge_attr, data.edge_index)
            evaluations = render_world_from_graph(x, constraints=constraints, **render_kwargs)
        else:
            render_world_from_graph(x, **render_kwargs)
    return np.concatenate(all_features, axis=0)


def analyze_feature_distribution(dir_name='TriangularRandomSplitWorld(10)_test_7_split'):
    features = ['l', 'v3x', 'v3y', 'x1', 'y1', 'cos', 'sin']
    all_features = visualize_dataset(dir_name, visualize=False)
    plt.figure(figsize=(16, 8))
    for k in range(all_features.shape[1]):
        ax = plt.subplot(2, 4, k + 1)
        data = all_features[:, k]
        mean = np.mean(data)
        std = np.std(data)
        print(f'feature {k}: {mean} +- {std}')
        ax.scatter(np.zeros_like(data), data, alpha=0.5)
        plt.errorbar(np.ones_like(data[0])*0.25, mean, std, c='red', linestyle='None', marker='o', capsize=5)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1.1, 1.1])
        ax.set_title(f'feature {features[k]}')
    plt.tight_layout()
    plt.savefig(join(VISUALIZATION_PATH, 'data_distribution', 'analyze_dataset.png'))


###################################################################################################


def load_dataset_for_checking(train_task, input_mode, batch_size=50000):
    dataset_kwargs = dict(input_mode=input_mode, pre_transform=pre_transform)
    train_dataset = GraphDataset(train_task, **dataset_kwargs)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)
    return train_dataloader


def check_data_distribution(train_task="RandomSplitWorld(20000)_train", input_mode='diffuse_pairwise'):
    """ check data balance """
    train_dataloader = load_dataset_for_checking(train_task, input_mode)
    for data in train_dataloader:
        data = data.to('cuda')
        counts = torch.bincount(data.x_extract.to(torch.int))
        counts = torch.bincount(counts).detach().cpu().numpy().tolist()
        counts = {i - 1: counts[i] for i in range(len(counts)) if counts[i] > 0}
        print(counts)
        break


def visualize_qualitative_constraints_two_fold(task_name='RandomSplitQualitativeWorld(30000)_qualitative_train'):
    n = 3
    task_name = f'RandomSplitQualitativeWorld(100)_qualitative_test_{n}_split'
    train_dataloader = load_dataset_for_checking(task_name, 'qualitative', batch_size=1)
    constraint_pairs = defaultdict(list)
    for j, data in enumerate(train_dataloader):
        if data.x.shape[0] != n + 1: continue
        constraint_names = data.edge_attr.numpy().astype(int).tolist()
        constraint_indices = data.edge_index.T.numpy().tolist()
        constraints = defaultdict(list)
        for i in range(len(constraint_names)):
            constraint_name = qualitative_constraints[constraint_names[i]]
            if constraint_name in ['in', 'cfree']:
                continue
            a, b = constraint_indices[i]
            if constraint_name in ['v-aligned', 'h-aligned', 'close-to', 'away-from']:
                constraints[b].append((constraint_name, a))
            constraints[a].append((constraint_name, b))
            # print(constraint_name, a, b)

        for a, v in constraints.items():
            if len(v) != 2:
                continue
            (con1, b1), (con2, b2) = v
            if b1 == b2:
                continue
            key = (con1, con2) if not (con2, con1) in constraint_pairs else (con2, con1)
            constraint_pairs[key].append((j, (con1, a, b1), (con2, a, b2)))

    pprint({k: len(v) for k, v in constraint_pairs.items()})
    constraint_pairs = {f"{k[0]}|{k[1]}": v[:10] for k, v in constraint_pairs.items()}
    with open(join(VISUALIZATION_PATH, 'compose_constraints', f'{task_name}.json'), 'w') as f:
        json.dump(constraint_pairs, f, indent=4)


def visualize_qualitative_distribution(train_task="RandomSplitQualitativeWorld(60000)_qualitative_train"):
    """ check data symmetry
    https://python-graph-gallery.com/80-contour-plot-with-seaborn/
    """

    ## skip if already generated
    dist_dir = join(VISUALIZATION_PATH, 'data_distribution', train_task)
    if len([f for f in listdir(join(dist_dir)) if '.png' in f]) == 13:
        return

    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib
    # use LaTeX, choose nice some looking fonts and tweak some settings
    matplotlib.rc('font', family='serif')
    matplotlib.rc('font', size=16)
    matplotlib.rc('legend', fontsize=16)
    matplotlib.rc('legend', numpoints=1)
    matplotlib.rc('legend', handlelength=1.5)
    matplotlib.rc('legend', frameon=False)
    matplotlib.rc('xtick.major', pad=7)
    matplotlib.rc('xtick.minor', pad=7)
    matplotlib.rc('text', usetex=True)
    matplotlib.rc('text.latex',
                  preamble=r'\usepackage[T1]{fontenc}\usepackage{amsmath}\usepackage{txfonts}\usepackage{textcomp}')

    train_dataloader = load_dataset_for_checking(train_task, 'qualitative', batch_size=6000)
    counts = defaultdict(list)
    poses_data = defaultdict(list)
    total_con = 0
    os.makedirs(dist_dir, exist_ok=True)
    for data in train_dataloader:
        data = data.to('cuda')
        constraint_names = data.edge_attr.to(torch.int)
        constraint_pairs = data.edge_index.to(torch.int).T
        all_poses = data.x.to(torch.float32)[:, 2:4]
        for i in range(len(qualitative_constraints)):
            con_name = qualitative_constraints[i]

            rows = torch.where(constraint_names == i)[0]
            pairs = constraint_pairs[rows]
            new_rows = pairs[:, 0].to(torch.long).T
            pairs = pairs.detach().cpu().numpy().tolist()
            poses = all_poses[new_rows]

            counts[con_name].extend(pairs)
            poses_data[con_name].extend(poses.detach().cpu().numpy().tolist())
            total_con += len(pairs)
        break

    for k, v in counts.items():
        print(k, len(v))
        if len(v) > 0:
            print('\t', round(len([a for a, b in v if a > b]) / len(v), 3))

        xy = np.asarray(poses_data[k])
        sns.jointplot(x=xy[:, 0], y=xy[:, 1], kind='kde', space=0, cmap="Reds", fill=True, bw_adjust=.5)
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.xlabel('x', fontsize=20)
        plt.ylabel('y', fontsize=20)
        plt.subplots_adjust(left=0.15, right=0.98, top=0.98, bottom=0.12)
        plt.annotate(f"{len(v)} / {total_con}", (0.18, 0.8), fontsize=22)
        # plt.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.1)
        # plt.suptitle(f"data distribution of {k} ({len(v)} / {total_con})", fontsize=18)
        # plt.show()
        plt.savefig(join(dist_dir, f'{k}.png'))
        plt.close()



def check_enough_side_grasps(solution_json):
    categories = [d['name'].split('_')[0] for d in json.load(open(solution_json))['placements']]
    side_grasp_objects = [c for c in categories if c in ['BottleOpened', 'Bowl', 'Dispenser']]
    num_side_grasp = len(side_grasp_objects)
    num_object = eval(str(dirname(solution_json))[-1])
    return num_side_grasp, num_side_grasp > min(2, num_object - 1)


def visualize_packing_object_distribution():
    """
    18 230531_143954_i=82_n=5 ['Dispenser', 'Bowl', 'Bowl']
    21 230531_144121_i=88_n=5 ['Bowl', 'Bowl', 'BottleOpened']
    64 230531_225054_i=73_n=5 ['Dispenser', 'BottleOpened', 'BottleOpened']
    66 230531_225114_i=58_n=5 ['Dispenser', 'BottleOpened', 'BottleOpened']
    85 230531_225357_i=74_n=5 ['BottleOpened', 'BottleOpened', 'BottleOpened']
    99 230531_230108_i=98_n=5 ['Dispenser', 'BottleOpened', 'Dispenser']
    """

    data_dir = join(DATASET_PATH, 'TableToBoxWorld(100)_robot_box_test_5_object', 'raw')
    data_dir = join(DATASET_PATH, 'TableToBoxWorld(10000)_robot_box_train', 'raw')
    data_dirs = listdir(data_dir)
    counts = {n: defaultdict(list) for n in [3, 4, 5]}
    for i, sub in enumerate(sorted(data_dirs)):
        solution_json = join(data_dir, sub, 'solution.json')
        num_side_grasp, passed = check_enough_side_grasps(solution_json)
        num_object = eval(sub[-1])
        if passed:
            counts[num_object][num_side_grasp].append(sub)
    pprint({k: {kk: len(vv) for kk, vv in v.items()} for k, v in counts.items()})


if __name__ == "__main__":
    """ 
    given a new World or input mode, change 
        - data_transforms.py    pre_transform()
        - data_utils.py         render_world_from_graph() / robot_data_json_to_pt()
    """
    # visualize_dataset('RandomSplitWorld(50)_test', visualize=True)
    # visualize_dataset('TriangularRandomSplitWorld[64]_(1)_diffuse_pairwise_image_train',
    #                   input_mode='diffuse_pairwise_image', visualize=False)
    # visualize_dataset('TriangularRandomSplitWorld[64]_(1)_diffuse_pairwise_train',
    #                   input_mode='diffuse_pairwise', visualize=True)

    # analyze_feature_distribution('TriangularRandomSplitWorld(100)_test_7_split')

    # visualize_dataset('TableToBoxWorld(10)_train', input_mode='robot_box')
    # visualize_dataset('TableToBoxWorld(3)_test', input_mode='robot_box', visualize=True)
    # visualize_dataset('TableToBoxWorld(1)_robot_real', visualize=True)

    # visualize_dataset('RandomSplitWorld(1)_stability_train', input_mode='stability_flat', visualize=False)
    # visualize_dataset('RandomSplitWorld(20)_stability_train', input_mode='stability_flat', visualize=False)
    # visualize_dataset('RandomSplitWorld(20)_stability_test', input_mode='stability_flat', visualize=False)

    # visualize_dataset('RandomSplitQualitativeWorld(30000)_qualitative_train', input_mode='qualitative', visualize=False)

    ###########################################################

    # check_data_distribution()
    visualize_qualitative_distribution()
    # visualize_packing_object_distribution()
    # visualize_qualitative_constraints_two_fold()
