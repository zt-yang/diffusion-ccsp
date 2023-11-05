import os
import json
import numpy as np
from os.path import join, abspath, isdir, isfile, basename, dirname
from os import listdir
from config import RENDER_PATH, DATASET_PATH
from collections import defaultdict

from mesh_utils import RAINBOW_COLOR_NAMES

#########################################################################


HTML = """
<!DOCTYPE html>
<html>
<head>
<link rel="stylesheet" href="images.css">
</head>
<body>

<div id="header">
  <h1>Diffusion-CSP: {count} runs in {data_name}</h1>
</div>
<div style="padding-top: 90px"></div>
{stats}
{table}
</body>
</html>
"""

TABLE = """
<div class="main">
  <table>

    {lines}

  </table>
</div>
"""

ROW = """
    <tr>
      {cells}
    </tr>

"""

IMG = """
      <td> {run_name}
        <img class="image" src="{path}"/> {comment}
      </td>"""

IMGS = """
      <td> {run_name}
        {imgs} {comment}
      </td>"""
IMG_LINE = """
        <img class="image" src="{path}"/>"""

STATS = """
<div>
    <img class="full_page" src="{path}"/>
</div>
"""


def make_collage(files, json_files=None):
    color_names = ['shelf'] + RAINBOW_COLOR_NAMES
    lines = []
    cells = ''
    new_line = '<br>------------------<br>'
    num_per_row = 5
    for idx, file in enumerate(files):
        col = idx % num_per_row
        run_name = f"id={idx}<br>{basename(dirname(file))}" + new_line

        comment = ''
        if json_files is not None:
            if not isfile(json_files[idx]):
                continue
            data = json.load(open(json_files[idx]))
            if 'placements' in data and 'name' in data['placements'][0]:
                comment = '<br>'.join([pc['name'] for pc in data['placements']])
            if 'order' in data:
                comment += '<br>' + ', '.join([color_names[o] for o in data['order']])
            if 'supports' in data:
                comment += '<br>' + ', '.join([str([color_names[elem] for elem in elems]) for elems in data['supports']])
            if 'constraints' in data:
                color_names = ['tray'] + RAINBOW_COLOR_NAMES
                comment += '<br>' + '<br>'.join(['(' + ', '.join(
                    [e[0], color_names[e[1]], color_names[e[2]]]
                ) + ')' for e in data['constraints'] if e[0] not in ['in', 'cfree']])
        cells += IMG.format(run_name=run_name, path=file, comment=comment)

        if col == num_per_row - 1:
            lines.append(ROW.format(cells=cells))
            cells = ''

    if cells != '':
        lines.append(ROW.format(cells=cells))
    return TABLE.format(lines='\n'.join(lines))


def visualize_images(dir_name='TableToBoxWorld_100', dir_names=None):
    dir_name, dirs = get_dir_name_all_dirs(dir_name, dir_names, topk=100)
    if isfile(join(dirs[0], 'problem.png')):
        files = [join(d, 'problem.png') for d in dirs]
    elif isfile(join(dirs[0], 'solution.png')):
        files = [join(d, 'solution.png') for d in dirs]
    else:
        files = [join(d, 'world.png') for d in dirs]

    json_files = [join(d, 'solution.json') for d in dirs]
    img_dir = join(RENDER_PATH, dir_name)

    table = make_collage(files, json_files)

    ## summary image
    stats_png = img_dir + '.png'
    stats = STATS.format(path=stats_png) if isfile(stats_png) else ''

    stats_png = img_dir + '_assets.png'
    stats += STATS.format(path=stats_png) if isfile(stats_png) else ''

    file_name = img_dir + '.html'
    print(f'file://{file_name}')
    with open(file_name, 'w') as f:
        f.write(HTML.format(count=len(files), data_name=dir_name, stats=stats, table=table))


def visualize_gen_images(dir_name, t):
    num_objects = range(2, 6)
    dir_name = join(RENDER_PATH, dir_name)
    data = json.load(open(join(dir_name, f'denoised_t={t}.json')))
    files = []
    for n in num_objects:
        rounds = data[str(n)]['success_rounds']
        print(len(rounds), rounds)
        for i in range(10):
            k = rounds[str(i)] if str(i) in rounds else 9
            files.append(join(dir_name, f'denoised_t={t}_n={n}_i={i}_k={k}.png'))
    json_files = [f.replace('.png', '.json') for f in files]
    table = make_collage(files)  ## , json_files

    file_name = join(RENDER_PATH, f'{dir_name}.html')
    print(f'file://{file_name}')
    with open(file_name, 'w') as f:
        f.write(HTML.format(count=len(files), data_name=dir_name, stats='', table=table))


def visualize_qualitative_images(dir_name='TableToBoxWorld_100', dir_names=None):
    dir_name, json_files = get_dir_name_all_dirs(dir_name, dir_names, rendir_json=True)
    from matplotlib import pyplot as plt

    # dir_name = join(RENDER_PATH, dir_name)
    # json_files = [join(dir_name, f) for f in listdir(dir_name) if f.endswith('.json')]
    png_files = [f.replace('.json', '.png') for f in json_files]
    png_files = [f for f in png_files if isfile(f)]
    png_files = png_files[:min(100, len(png_files))]
    table = make_collage(png_files, json_files)

    all_constraints = defaultdict(int)
    doubled_constraints = defaultdict(int)
    num_objects = defaultdict(int)
    for json_file in json_files:
        data = json.load(open(json_file))
        n = len([d for d in data['objects'].values() if 'tile_' in d['label']])
        num_objects[n] += 1
        constraints = data['constraints']
        for c in constraints:
            all_constraints[c[0]] += 1
            if [c[0], c[2], c[1]] in constraints:
                doubled_constraints[c[0]] += 1

    ## make a figure with two subfigures, ploting the number of constraints and the number of doubled constraints
    fig = plt.figure(figsize=(25, 5))
    plt.bar(all_constraints.keys(), all_constraints.values())
    plt.title('Number of constraints', fontsize=26)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=18)
    fig_name = join(dir_name, 'summary.png') if dir_names is None else join(RENDER_PATH, f'{dir_name}_summary.png')
    plt.savefig(fig_name)
    stats = STATS.format(path=fig_name)

    fig = plt.figure(figsize=(5, 5))
    plt.bar(num_objects.keys(), num_objects.values())
    plt.title('Number of objects', fontsize=26)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=18)
    fig_name = join(dir_name, 'num_objects.png') if dir_names is None else join(RENDER_PATH, f'{dir_name}_objects.png')
    plt.savefig(fig_name)
    stats += STATS.format(path=fig_name)

    file_name = join(RENDER_PATH, f'{dir_name}.html')
    print(f'file://{file_name}')
    with open(file_name, 'w') as f:
        f.write(HTML.format(count=len(json_files), data_name=dir_name, stats=stats, table=table))


def get_dir_name_all_dirs(dir_name, dir_names, topk=None, rendir_json=False):
    if dir_names is not None:
        common = os.path.commonprefix(dir_names)
        dir_name = dir_names[0] + '-->' + dir_names[-1].replace(common, '')
    else:
        dir_names = [dir_name]
    dirs = []
    for d in dir_names:
        if rendir_json:
            dd = join(RENDER_PATH, d)
            new_dirs = [abspath(join(dd, f)) for f in listdir(dd) if f.endswith('.json')]
        else:
            dd = join(DATASET_PATH, d, 'raw')
            new_dirs = [abspath(join(dd, f)) for f in listdir(dd) if isdir(join(dd, f))]
        if topk is not None:
            new_dirs = new_dirs[:topk]
        dirs += new_dirs
    dirs = sorted(dirs)
    return dir_name, dirs


if __name__ == '__main__':
    # visualize_images(dir_name='TableToBoxWorld(10000)_robot_box_train')
    # visualize_images(dir_name='TableToBoxWorld(10)_robot_box_test')
    # visualize_images(dir_names=[f'TableToBoxWorld(100)_robot_box_test_{i}_object' for i in range(2, 7)])

    # visualize_images(dir_name='RandomSplitWorld(30)_stability_train')

    # visualize_gen_images(dir_name='TableToBoxWorld(10000)_train_m=False_t=1000_robot_box_id=buo2pzcg', t=16)

    # visualize_images(dir_name='RandomSplitWorld(30000)_stability_train')
    # visualize_images(dir_name='RandomSplitWorld(20)_stability_train')
    # visualize_images(dir_names=[f'RandomSplitWorld(10)_stability_test_{i}_object' for i in range(4, 9)])

    # visualize_qualitative_images(dir_name='RandomSplitQualitativeWorld(10000)_qualitative_train_3_object')
    visualize_qualitative_images(dir_names=[f'RandomSplitQualitativeWorld(10)_qualitative_test_{i}_split' for i in range(2, 7)])
