from typing import Iterable, Tuple
import os
from os import listdir
import shutil
import copy
import json
import sys
from tqdm import tqdm
import math
from config import *
from worlds import TrayWorld, ShapeSettingWorld, RandomSplitWorld, get_world_class
from render_utils import *
from data_utils import world_from_pt
import argparse
from collections import defaultdict


class DataCollector(object):

    def __init__(self, world_class: TrayWorld.__class__,
                 world_args: dict = {},
                 scene_sampler_args: dict = {}):
        self.name = self.__class__.__name__
        self.world_class = world_class
        self.world_args = world_args
        self.scene_sampler_args = scene_sampler_args

    def collect(self, n, dataset_dir=DATASET_PATH, label='',
                del_if_exists: bool = True,
                verbose: bool = True,
                pngs: bool = False,
                jsons: bool = False,
                shake_per_world: int = 1,
                balance_data: bool = True,
                save_meshes: bool = False,
                input_mode: str = 'collisions',
                **kwargs):
        import torch
        world = self.world_class(**self.world_args)
        name = f"{world.name}({n})"
        if len(label) > 0:
            name += f"_{label}"

        ## accounting for data distribution
        if 'diffuse_pairwise' in input_mode or 'qualitative' in input_mode:
            class_counts = defaultdict(int)
        else:
            classes = []
            if input_mode == 'collisions':
                n_classes = 2
                classes = range(n_classes)
            elif 'grid' in input_mode:
                n_classes = int(world.w/world.grid_size) * int(world.l/world.grid_size)
                classes = range(n_classes)
            class_counts = {k: 0 for k in classes}

        ## dataset directory
        dataset_dir = join(dataset_dir, name)
        raw_dir = join(dataset_dir, 'raw')
        json_dir = join(dataset_dir, 'json')
        render_dir = join(RENDER_PATH, name)
        if isdir(dataset_dir):
            if del_if_exists:
                shutil.rmtree(dataset_dir)
                if isdir(render_dir):
                    shutil.rmtree(render_dir)
        if not isdir(dataset_dir):
            os.mkdir(dataset_dir)
        if not isdir(raw_dir):
            os.mkdir(raw_dir)
        if label == 'test' and not isdir(json_dir):
            os.mkdir(json_dir)

        ## images directory
        if pngs or jsons:
            png_dir = join(RENDER_PATH, name)
            if not isdir(png_dir):
                os.mkdir(png_dir)

        if isfile(join(dataset_dir, f'class_counts.pt')):
            print('\nLoading existing class counts...')
            print(torch.load(join(dataset_dir, f'class_counts.pt')))
            print('Loading existing class weights...')
            print(torch.load(join(dataset_dir, f'class_weights.pt')))
            return

        ## sample n data
        newly_generated = 0

        def add_one_pt(world, data_path, png_path, newly_generated):

            json_name = None if not jsons else png_path.replace('.png', '.json')
            data = world.generate_json(json_name=json_name, input_mode=input_mode)
            # if label == 'test':
            #     json_path = data_path.replace('.pt', '.json').replace(raw_dir, json_dir)
            #     with open(json_path, 'w') as f:
            #         json.dump(data, f)
            c = world.generate_pt(data=data, data_path=data_path, verbose=verbose, input_mode=input_mode, **kwargs)
            for k, v in c.items():
                class_counts[k] += v

            if pngs and not isfile(png_path):
                world.render(show=False, save=True, show_grid=True, img_name=png_path)

            """ save object meshes """
            if save_meshes:
                t = png_path.split('/')[-1].split('.')[0].split('=')[-1]
                world.generate_meshes(f"idx={t}")

            newly_generated += 1
            return newly_generated

        n = n // shake_per_world
        counts = defaultdict(int)
        min_n = self.scene_sampler_args['min_num_objects']
        max_n = self.scene_sampler_args['max_num_objects']
        scene_sampler_args = copy.deepcopy(self.scene_sampler_args)
        count_threshold = math.ceil(n / (max_n - min_n + 1))
        current_n = min_n
        scene_sampler_args['max_num_objects'] = current_n
        for t in tqdm(range(n)):
            t = t * shake_per_world
            data_path = join(raw_dir, f'data_{t}.pt')
            png_path = join(RENDER_PATH, name, f'idx={t}.png')

            if isfile(data_path):
                world = world_from_pt(data_path, self.world_class.__name__)
                grids = torch.load(data_path).y.numpy()[1:, 0].tolist()
                for g in grids:
                    class_counts[g] += 1
            else:
                world = self.world_class(**self.world_args)
                world.sample_scene(**scene_sampler_args)
                newly_generated = add_one_pt(world, data_path, png_path, newly_generated)

                ## balancing the dataset for the boxes dataset
                if balance_data:
                    n_objects = len(world.tiles)
                    counts[n_objects] += 1
                    if counts[n_objects] >= count_threshold:
                        scene_sampler_args['min_num_objects'] = n_objects + 1
                        if scene_sampler_args['min_num_objects'] > scene_sampler_args['max_num_objects']:
                            scene_sampler_args['max_num_objects'] = scene_sampler_args['min_num_objects']

                s = t
                shake_scenes_gen = world.shake_scenes_gen(num=shake_per_world-1, is_generator=True)
                for new_world in shake_scenes_gen:
                    s += 1
                    data_path = join(raw_dir, f'data_{s}.pt')
                    png_path = join(RENDER_PATH, name, f'idx={s}.png')
                    newly_generated = add_one_pt(new_world, data_path, png_path, newly_generated)

                # c = world.generate_pt(verbose=verbose, data_path=data_path, **kwargs)
                # for k, v in c.items():
                #     class_counts[k] += v
                # newly_generated += 1
                #
                # if pngs and not isfile(png_name):
                #     world.render(show=False, save=True, show_grid=True, img_name=png_name)

        comment = 'existing' if newly_generated == 0 else 'newly generated'

        print('class_counts', class_counts)
        class_counts = [v for v in class_counts.values()]
        torch.save(class_counts, join(dataset_dir, f'class_counts.pt'))

        class_counts = [v+1 for v in class_counts]
        if len(class_counts) == 2:
            class_weights = torch.tensor(class_counts, dtype=torch.float)
            class_weights /= class_weights.sum()
            class_weights = 1. - class_weights
        else:
            class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
        torch.save(class_weights, join(dataset_dir, f'class_weights.pt'))

        if 'collision' in input_mode or 'grid' in input_mode:
            print(f'\nLoading {comment} class counts...')
            print(class_counts)
            print(f'Loading {comment} class weights...')
            print(class_weights)
        print('saved', dataset_dir)


def get_data_collection_args(world_name='RandomSplitWorld', input_mode='diffuse_pairwise',
                             num_worlds=10, verbose=False, num_shakes=1, data_type='train',
                             min_num_objects=2, max_num_objects=5, pngs=False, jsons=False,
                             del_if_exists=True, world_args=dict(), w=3.0, l=2.0, h=0.5, grid_size=0.5):

    if 'w' in world_args:
        w = world_args['w']
    if 'l' in world_args:
        l = world_args['l']
    if 'h' in world_args:
        h = world_args['h']

    parser = argparse.ArgumentParser()
    parser.add_argument('-world_name', type=str, default=world_name)
    parser.add_argument('-data_type', type=str, default=data_type, choices=['train', 'test'])
    parser.add_argument('-input_mode', type=str, default=input_mode)
    parser.add_argument('-num_worlds', type=int, default=num_worlds)
    parser.add_argument('-num_shakes', type=int, default=num_shakes)
    parser.add_argument('-min_num_objects', type=int, default=min_num_objects)
    parser.add_argument('-max_num_objects', type=int, default=max_num_objects)

    parser.add_argument('-grid_size', type=float, default=grid_size)
    parser.add_argument('-width', type=float, default=w)
    parser.add_argument('-length', type=float, default=l)
    parser.add_argument('-height', type=float, default=h)

    parser.add_argument('-pngs', action='store_true')
    parser.add_argument('-jsons', action='store_true')
    parser.add_argument('-del_if_exists', action='store_true')

    parser.add_argument('-verbose', action='store_true', default=verbose)
    args = parser.parse_args()

    args.pngs = pngs or args.pngs
    args.jsons = jsons or args.jsons
    args.del_if_exists = del_if_exists or args.del_if_exists

    world_args.update(dict(w=args.width, l=args.length, h=args.height, grid_size=args.grid_size))
    args.world_args = world_args

    return args


def main(**kwargs):
    args = get_data_collection_args(**kwargs)
    if args.data_type == 'train':
        generate_train_dataset(args)
    else:
        generate_test_dataset(args)


def generate_train_dataset(args=None, debug=False, save_meshes=False, same_order=False, **kwargs):
    if args is None:
        args = get_data_collection_args(**kwargs)
    scene_sampler_args = dict(min_num_objects=args.min_num_objects, max_num_objects=args.max_num_objects)

    print(args.pngs, args.jsons)

    world_class = get_world_class(args.world_name)
    collector = DataCollector(world_class, world_args=args.world_args, scene_sampler_args=scene_sampler_args)
    collector.collect(args.num_worlds, shake_per_world=args.num_shakes, label=args.input_mode + '_train',
                      verbose=args.verbose, pngs=args.pngs, jsons=args.jsons, debug=debug,
                      save_meshes=save_meshes, same_order=same_order)
    # collector.collect(int(args.num_worlds/10), label='test', **kwargs)


def generate_test_dataset(args=None, pngs=True, jsons=True,
                          save_meshes=False, same_order=False, **kwargs):
    """ Generate a set of test dataset with given number of objects in each scene
    e.g. `num_objects = range(2, 5)` creates three folders, each contains `num_worlds` with 2, 3, 4 objects
    """
    if args is None:
        if 'num_objects' in kwargs:
            kwargs['min_num_objects'], kwargs['max_num_objects'] = kwargs.pop('num_objects')
        args = get_data_collection_args(**kwargs)

    args.pngs = pngs or args.pngs
    args.jsons = jsons or args.jsons
    world_class = get_world_class(args.world_name)
    kwargs.update(dict(input_mode=args.input_mode, shake_per_world=1,
                       pngs=args.pngs, jsons=args.jsons, verbose=False))
    for i in range(args.min_num_objects, args.max_num_objects + 1):
        scene_sampler_args = dict(min_num_objects=i, max_num_objects=i)
        collector = DataCollector(world_class, world_args=args.world_args, scene_sampler_args=scene_sampler_args)
        collector.collect(args.num_worlds, label=f'{args.input_mode}_test_{i}_split', pngs=args.pngs, jsons=args.jsons,
                          save_meshes=save_meshes, same_order=same_order)


######################## tests ############################


def test_random_split_world(n=10):
    world_args = dict(grid_size=0.5)
    collector = DataCollector(RandomSplitWorld, world_args=world_args)
    collector.collect(n, label='train', verbose=False)
    collector.collect(n, label='test')


def test_generate_random_split_data():
    world_class = 'RandomSplitWorld'
    test_random_split_world(n=10)
    generate_train_dataset(world_class=world_class, num_worlds=1, min_num_objects=3, max_num_objects=3)
    # generate_train_dataset(world_class=world_class, num_worlds=20000, min_num_objects=3, max_num_objects=6)
    generate_test_dataset(world_class=world_class, num_objects=range(3, 11), num_worlds=100)


def test_generate_triangular_data():
    world_class = 'TriangularRandomSplitWorld'
    input_mode = 'diffuse_pairwise'  ## 'diffuse_pairwise_image'  ##
    kwargs = dict(world_args=dict(w=3, l=3, image_dim=64), pngs=True, jsons=True)
    # generate_test_dataset(world_class=world_class, num_objects=range(6, 7), num_worlds=5, input_mode=input_mode, save_meshes=True, **kwargs)
    # generate_test_dataset(world_class=world_class, num_objects=range(2, 7), num_worlds=10, input_mode=input_mode, **kwargs)

    kwargs.update(dict(min_num_objects=3, max_num_objects=5, input_mode=input_mode, pngs=False, jsons=False))
    # generate_train_dataset(world_class=world_class, num_worlds=1, **kwargs)
    generate_train_dataset(world_class=world_class, num_worlds=3000, **kwargs)


def test_generate_qualitative_data():
    world_class = 'RandomSplitQualitativeWorld'
    input_mode = 'qualitative'
    kwargs = dict(min_num_objects=2, max_num_objects=4, input_mode=input_mode,
                  jsons=False, pngs=False, same_order=False)
    generate_train_dataset(world_class=world_class, num_worlds=3000, **kwargs)  ## 30000, 60000
    # kwargs.update(jsons=True, pngs=True)
    # generate_test_dataset(world_class=world_class, num_objects=range(2, 7), num_worlds=100, **kwargs)


if __name__ == '__main__':

    if not isdir(DATASET_PATH):
        os.mkdir(DATASET_PATH)
    if not isdir(RENDER_PATH):
        os.mkdir(RENDER_PATH)

    main()
