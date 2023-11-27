import json
import os
import shutil
from os.path import join, abspath, isdir, dirname, basename, isfile
import numpy as np
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

import torch
from einops import rearrange, repeat
from tqdm import tqdm

from train_utils import load_trainer
from data_utils import render_world_from_graph
from denoise_fn import qualitative_constraints, puzzle_constraints
from datasets import visualize_qualitative_distribution

DATA_PATH = abspath(join(dirname(__file__), 'data'))
RENDER_PATH = abspath(join(dirname(__file__), 'renders'))
VISUALIZATION_PATH = abspath(join(dirname(__file__), 'visualizations'))
OUTPUT_PATH = join(VISUALIZATION_PATH, 'energy_fields')
if not isdir(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)

PLACE_HOLDER = 'fff'


HTML = """<!DOCTYPE html>
<html>
<head>
<link rel="stylesheet" href="field.css">
</head>
<body>

<!--div id="header">
  <h1>Energy & Gradient Fields: {input_mode} constraints</h1>
</div-->

{table}
</body>
</html>
"""

TABLE = """
<div class="main">
  <table>
    <tr>
      {col_names}
    </tr>

    {lines}
  </table>
</div>
"""

ROW = """
    <tr>
      <td>{con}</td>
      {cells}
    </tr>

"""

IMG_TD = """ 
      <td> {images}
      </td>"""

IMG_LINE = """ 
        <img class="image" src="{png_name}" style="height: {height}px;"/>{comments}"""

VID_LINE = """
         <video width="320" height="240" controls>
          <source src="{mp4_name}" type="video/mp4">
         </video> """

HEADING_ROW = """
        <th>{name}</th>
"""


def make_html(items, input_mode, html_path, train_task='RandomSplitQualitativeWorld(60000)_qualitative_train'):
    col_names = ['Constraint', 'Energy & Gradient (t=0)', 'Sampled Data', 'Data Distribution']
    col_names = ''.join([HEADING_ROW.format(name=name) for name in col_names])
    lines = ''
    for con, img_path in items.items():
        imgs = [img_path.replace(PLACE_HOLDER, 'field'), img_path[:img_path.index(PLACE_HOLDER)]+'data.png']
        imgs += [join(VISUALIZATION_PATH, 'data_distribution', train_task, f'{con}.png')]
        # new_imgs = [img.replace(input_mode, f"{input_mode}*") for img in imgs] + [imgs[-1]]
        images = [IMG_TD.format(images=IMG_LINE.format(png_name=abspath(img), comments='', height=175)) for img in imgs]
        images[0] = images[0].replace('_50.', '_0.').replace('_100.', '_0.')
        images[1] = images[1].replace('_50.', '_0.').replace('_100.', '_0.')
        lines += ROW.format(con=con, cells=''.join(images))

    print(abspath(html_path))
    with open(html_path, 'w') as f:
        f.write(HTML.format(input_mode=input_mode, table=TABLE.format(col_names=col_names, lines=lines)))


def make_html_composed(items, originals, input_mode, html_path, img_dir):
    col_names = ['Combination', 'Field 1 (t=0)', 'Field 2 (t=0)', 'Composed Field', 'Sampled Data', "Original Data"]
    col_names = ''.join([HEADING_ROW.format(name=name) for name in col_names])
    lines = ''
    suffix = '_field_0_large.png'
    for i, con in enumerate(items):
        con1, con2 = con.split('|')
        file_prefix = join(img_dir, f'{input_mode}_{con}')
        img_data, constraints = originals[i]
        constraints = [','.join([str(con) for con in constraint]) for constraint in constraints]
        constraints = '<br>'.join([f'({constraint})' for constraint in constraints])
        con = con.replace('|', '<br>')
        imgs = [file_prefix + img for img in [f'_{con1}{suffix}', f'_{con2}{suffix}', f'{suffix}', f'_data.png']]
        imgs[-1] = imgs[-1].replace('/qualitative_', '/')
        imgs += [img_data]
        images = [IMG_TD.format(images=IMG_LINE.format(png_name=abspath(img), comments='', height=115)) for img in imgs]
        lines += ROW.format(con=constraints, cells=''.join(images))

    print(abspath(html_path))
    with open(html_path, 'w') as f:
        f.write(HTML.format(input_mode=input_mode, table=TABLE.format(col_names=col_names, lines=lines)))


def get_plot_name(run_id, input_mode, name, t, output_dir=None):
    title = f'{input_mode} constraint [ {name} ]'  ##  (t={t})
    if output_dir is None:
        output_dir = join(OUTPUT_PATH, run_id)
    file_name = f'{input_mode}_{name}_{PLACE_HOLDER}_{t}' + '_large'
    file_name = join(output_dir, f'{file_name}.png')
    return title, file_name


def get_data_from_enegy_fn(energy_fn, file_name):
    # Generate a grid of points
    x = np.linspace(-1, 1, 10)
    y = np.linspace(-1, 1, 10)

    # Create 2-D grid
    X, Y = np.meshgrid(x, y)

    # Compute energy and gradients on the grid
    Z, V, _ = energy_fn(X, Y)

    np.save(f"{file_name}_X.npy", X)
    np.save(f"{file_name}_Y.npy", Y)
    np.save(f"{file_name}_Z.npy", Z)
    np.save(f"{file_name}_V.npy", V)

    return X, Y, Z, V


def get_data_from_saved_data(file_name):
    file_name = file_name.replace('_large', '')
    return [np.load(f"{file_name}_{n}.npy") for n in ['X', 'Y', 'Z', 'V']]


def make_plot(energy_fn, run_id='run', name='test', input_mode='', t=0, save_png=False,
              draw_title=False, use_toy_data=True, output_dir=None):
    title, file_name = get_plot_name(run_id, input_mode, name, t, output_dir=output_dir)
    if not draw_title:
        title = None
    data_file_name = join(dirname(file_name), 'data', basename(file_name).split('.')[0])
    if energy_fn is None:
        X, Y, Z, V = get_data_from_saved_data(data_file_name)
    else:
        X, Y, Z, V = get_data_from_enegy_fn(energy_fn, data_file_name)
    make_two_plots(X, Y, Z, V, title, file_name, save_png=save_png, use_toy_data=use_toy_data)
    return file_name


def make_two_plots(X, Y, Z, V, title, file_name, **kwargs):
    make_gradient_energy_field(X, Y, Z, V, title, file_name=file_name.replace('_'+PLACE_HOLDER, '_field'), **kwargs)
    # make_energy_field(X, Y, Z, title, file_name=file_name.replace('_'+PLACE_HOLDER, '_energy_field'), **kwargs)
    # make_gradient_field(X, Y, V, title, file_name=file_name.replace('_'+PLACE_HOLDER, '_gradient_field'), **kwargs)


def finish_field_plot(title, file_name='made_plot.png', save_png=False, use_toy_data=True, fontsize=18):
    if title is not None:
        plt.title(title, fontsize=18)

    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    fontsize = int(fontsize * 1.2)
    plt.xlabel('x', fontsize=fontsize)
    plt.ylabel('y', fontsize=fontsize)
    if save_png:
        plt.tight_layout()
        plt.savefig(file_name.replace('.pdf', '.png'))
        plt.savefig(file_name.replace('.png', '.pdf'), dpi=300)
        plt.close()
    else:
        plt.show()


def make_energy_field(X, Y, Z, title, **kwargs):
    # Create a contour plot
    plt.figure(figsize=(6, 4.5))
    plt.contourf(X, Y, Z, 50, cmap='RdGy_r')
    plt.colorbar()
    title = f'Energy Field of {title}' if title is not None else None
    finish_field_plot(title, **kwargs)


def make_gradient_field(X, Y, V, title, **kwargs):
    """ https://www.numbercrunch.de/blog/2013/05/visualizing-vector-fields/
    sudo apt install cm-super
    """
    plt.figure(figsize=(6, 6))
    plt.quiver(X, Y, V[:, 0], V[:, 1], pivot='middle', headwidth=4, headlength=6)
    title = f'Gradient Field of {title}' if title is not None else None
    finish_field_plot(title, **kwargs)


def make_gradient_energy_field(X, Y, Z, V, title, use_toy_data=True, **kwargs):
    # plt.figure(figsize=(6, 5.5))
    figsize = (6, 4.8) if use_toy_data else (8, 4.8)
    plt.figure(figsize=figsize)
    plt.contourf(X, Y, Z, 50, cmap='coolwarm')
    plt.colorbar()
    plt.quiver(X, Y, V[:, 0], V[:, 1], pivot='middle', headwidth=4, headlength=6)
    title = f'Gradient Field of {title}' if title is not None else None
    finish_field_plot(title, fontsize=20, use_toy_data=use_toy_data, **kwargs)


def read_imgs(img_files, mp4=False):
    from PIL import Image
    from mesh_utils import GREEN

    frames = []
    for i, img in enumerate(img_files):
        new_frame = Image.open(img)

        ## add a progress bar
        new_frame = np.asarray(new_frame).copy()
        h, w, _ = new_frame.shape
        top, bottom, left, right = h - 20, h, 0, int(w * i / len(img_files))
        new_frame[top:bottom, left:right] = GREEN

        ## for GIF
        if not mp4:
            new_frame = Image.fromarray(np.uint8(new_frame))

        frames.append(new_frame)
    return frames


def img_files_to_gif(img_files, gif_name):
    """ Save into a GIF file that loops forever """
    frames = read_imgs(img_files)
    frames[0].save(gif_name, format='GIF', append_images=frames[1:], save_all=True, duration=300, loop=0)


def img_files_to_mp4(img_files, mp4_name, fps=25):
    import cv2
    frames = read_imgs(img_files, mp4=True)
    h, w, _ = frames[0].shape
    out = cv2.VideoWriter(mp4_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h), True)
    for frame in frames:
        out.write(frame[:, :, :3][:, :, ::-1])
    cv2.destroyAllWindows()
    out.release()


def get_test_energy_fn():
    def energy_fn(x, y):
        return x ** 2 + y ** 2
    return energy_fn


def get_test_data(input_mode, name, use_container=False):
    """
    ------------------------
    |  A                   |
    |      [B]    [X]      |
    |                      |
    ------------------------
    """
    import torch
    from torch_geometric.data import Data

    data_idx = 0
    world_dims = [(1, 1)]

    ## encode the shapes
    if input_mode == 'triangle':
        if name == 'in':
            geoms_in = [[0.2, 0.1, 0.1], [1, 1, 0]]
        elif name == 'cfree':
            geoms_in = [[0.2, 0.1, 0.1], [0.2, 0.1, 0.1]]

    elif input_mode == 'qualitative':
        if name.endswith('in'):
            geoms_in = [[0.15, 0.15], [1, 1]]
        elif use_container:
            geoms_in = [[1, 1], [0.15, 0.15], [0.15, 0.15]]
        else:
            geoms_in = [[0.15, 0.15], [0.15, 0.15]]

    ## encode the data batch
    if input_mode == 'qualitative':
        if use_container and name != 'in':
            if name.endswith('in'):
                x = [geoms_in[1] + [0, 0, 0, 0], geoms_in[0] + [0, 0, 0, 1]]
                original_x = [[0] + x[0], [1] + x[1]]

                mask = torch.tensor([1, 0])
                edge_index = torch.tensor([[1, 0], [1, 0]]).T
                edge_attr = torch.tensor([0, qualitative_constraints.index(name)])

            else:
                x = [geoms_in[0] + [0, 0, 0, 0], geoms_in[1] + [0, 0, 0, 1], geoms_in[2] + [0, 0, 0, 1]]
                original_x = [[0] + x[0], [1] + x[1], [1] + x[2]]

                mask = torch.tensor([1, 1, 0])
                edge_index = [[1, 0], [2, 0], [2, 1], [2, 1]]  ## (2, 1)
                edge_attr = [0, 0, qualitative_constraints.index('cfree'), qualitative_constraints.index(name)]
                if name == 'cfree':
                    edge_index = edge_index[:-1]
                    edge_attr = edge_attr[:-1]
                edge_index = torch.tensor(edge_index).T
                edge_attr = torch.tensor(edge_attr)
                geoms_in = [[0.15, 0.15], [1, 1]]
        else:
            if name.endswith('in'):
                x = [geoms_in[1] + [0, 0, 0, 0], geoms_in[0] + [0, 0, 0, 1]]
                original_x = [[0] + x[0], [1] + x[1]]
            else:
                x = [geoms_in[1] + [0, 0, 0, 1], geoms_in[0] + [0, 0, 0, 1]]
                original_x = [[1] + x[0], [1] + x[1]]

            mask = torch.tensor([1, 0])
            edge_index = torch.tensor([[1, 0]]).T  ## (2, 1)
            edge_attr = torch.tensor([qualitative_constraints.index(name)])

        conditioned_variables = torch.tensor([0])
        original_x = torch.tensor(original_x)
        x = torch.tensor(x)
        original_y = torch.tensor([0])
        world_name = 'RandomSplitQualitativeWorld'

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                conditioned_variables=conditioned_variables, mask=mask,
                x_extract=torch.ones(x.shape[0]) * data_idx,
                edge_extract=torch.ones(edge_index.shape[1]) * data_idx,
                world_dims=world_dims, original_x=original_x, original_y=original_y)
    return data, geoms_in, world_name


def plot_diffusion_by_name(run_id, milestone, name, input_mode, t=0, use_container=False,
                           save_png=False, render_history=False, **kwargs):

    trainer = load_trainer(run_id, milestone, visualize=False, test_model=False, verbose=False, **kwargs)
    trainer.model.eval()

    data, geoms_in, world_name = get_test_data(input_mode, name, use_container=use_container)

    """ get diffusion history """
    if render_history:
        pose_features, history = trainer.model.p_sample_loop(data, return_history=True)
        pose_features = trainer.get_all_features(pose_features, data)
        pose_features.clamp_(-1., 1.)
        # print(name, '\n', pose_features, '\n')

        if input_mode == 'qualitative' and not name.endswith('in') and not use_container:
            ## concatename the first position to the beginning
            container = torch.tensor([[1, 1, 0, 0, 0, 0]])
            pose_features = torch.cat([container, pose_features], dim=0)

            data.x = torch.cat([container, data.x], dim=0)
            data.original_x = torch.cat([torch.tensor([[0, 1, 1, 0, 0, 0, 0]]), data.original_x], dim=0)
            data.mask = torch.cat([torch.tensor([1]), data.mask], dim=0)
            data.edge_index = torch.tensor([[2, 1]]).T

            container = container.to(history[0].device)
            history = [torch.cat([container[:, 2:], h], dim=0) for h in history]

        ## render png of final result
        png_name = abspath(join(OUTPUT_PATH, f'{input_mode}_{name}_data.png'))
        render_kwargs = dict(world_dims=data.world_dims[0], world_name=world_name, log=True, show=False)
        if input_mode == 'qualitative':
            render_kwargs['constraints'] = [[name, 1, 0]]
        evaluations = render_world_from_graph(pose_features, save=True, png_name=png_name, **render_kwargs)
        print(name, '\tviolated constraints:', evaluations)

        ## render the diffusion process
        gif_name = png_name.replace('.png', '.gif')
        trainer.render_success(milestone, 2, 0, 0, data, history, gif_file=gif_name,
                               save_mp4=True, **render_kwargs)

    """ get energy field """
    denoise_fn, use_EBM_wrapper = get_denoise_fn(trainer)
    c_idx = denoise_fn.constraint_sets.index(name)

    geoms_in = torch.tensor(geoms_in).float().to(denoise_fn.device)
    geoms_emb = denoise_fn.geom_encoder(geoms_in)  ## torch.Size([2, 256])

    time_in = torch.tensor([t]).float().to(denoise_fn.device)
    time_emb = denoise_fn.time_mlp(time_in)  ## torch.Size([1, 256])

    def energy_fn(x, y):
        return make_energy_fn(x, y, denoise_fn, trainer, c_idx, geoms_emb, time_emb, use_EBM_wrapper)

    return make_plot(energy_fn, run_id=run_id, name=name, input_mode=input_mode, t=t, save_png=save_png)


def get_poses_in(x, y, denoise_fn, poses_A_rot=None, poses_B=None):
    poses_A = torch.tensor(np.array([x, y])).to(denoise_fn.device)  ## torch.Size([2, 10, 10])
    if poses_A_rot is None:
        poses_A_rot = torch.stack([torch.ones_like(poses_A[0]), torch.zeros_like(poses_A[0])], dim=0)
    else:
        poses_A_rot = repeat(poses_A_rot, 'c -> c h w', h=poses_A.shape[1], w=poses_A.shape[2])
    poses_A = torch.cat([poses_A, poses_A_rot], dim=0)  ## torch.Size([4, 10, 10])

    if poses_B is None:
        poses_B = torch.zeros_like(poses_A).to(denoise_fn.device)
    else:
        poses_B = repeat(poses_B, 'c -> c h w', h=poses_A.shape[1], w=poses_A.shape[2])

    poses_in = torch.stack([poses_A, poses_B], dim=1)  ## torch.Size([4, 2, 10, 10])
    poses_in.requires_grad = True
    return rearrange(poses_in, 'c n h w -> (h w) n c').float()  ## torch.Size([100, 2, 4])


def make_energy_fn(x, y, denoise_fn, trainer, c_idx, geoms_emb, time_emb,
                   use_EBM_wrapper=False, poses_A_rot=None, poses_B=None):
    n = x.shape[0] * x.shape[1]
    poses_in = get_poses_in(x, y, denoise_fn, poses_A_rot=poses_A_rot, poses_B=poses_B)  ## torch.Size([100, 2, 4]

    poses_emb = denoise_fn.pose_encoder(poses_in)  ## torch.Size([100, 2, 256])

    geoms_emb = repeat(geoms_emb, 'c d -> n c d', n=n)  ## torch.Size([100, 2, 256])
    time_emb = repeat(time_emb, 'c d -> n (c d)', n=n)  ## torch.Size([100, 256])

    input_dict = {
        'args': None,
        'geoms_emb': geoms_emb,  ## [b, 2 * hidden_dim]
        'poses_emb': poses_emb,  ## [b, 2 * hidden_dim]
        'time_embedding': time_emb  ## [b, hidden_dim]
    }
    outputs = denoise_fn._process_constraint(c_idx, input_dict)  ## torch.Size([100, 2, 4])

    if use_EBM_wrapper:
        energy = - torch.sum((outputs - poses_in) ** 2, dim=(1, 2))
                 # * trainer.model._sqrt_recipm1_alphas_cumprod_custom[t]
        gradients = trainer.model.denoise_fn.model._get_EBM_gradients(poses_in, energy.sum())
    else:
        energy = - torch.sum(outputs ** 2, dim=(1, 2))
        gradients = poses_in - outputs

    energy = energy.reshape(x.shape).cpu().detach().numpy()
    gradients = gradients[:, 0, :2].cpu().detach().numpy()  ## torch.Size([100, 2, 4]) -> (100, 2)
    return energy, gradients, outputs

#####################################################################################


def get_test_data_from_pt(data_pt, var):
    import torch
    from data_transforms import pre_transform
    data = torch.load(data_pt)
    data = pre_transform(data, 0, 'qualitative')[0]
    geoms_in = data.x[:, :2]
    data.conditioned_variables = [n for n in range(geoms_in.shape[0]) if n != var]
    data.mask = torch.tensor([n in data.conditioned_variables for n in range(geoms_in.shape[0])])
    return data, geoms_in, 'RandomSplitQualitativeWorld'


def plot_diffusion_by_pt(run_id, milestone, data_pt, key, con_pair, t=0, save_png=False, render_history=False, **kwargs):
    import torch
    from data_utils import render_world_from_graph, constraint_from_edge_attr

    data, geoms_in, world_name = get_test_data_from_pt(data_pt, con_pair[0][1])

    trainer = load_trainer(run_id, milestone, visualize=False, test_model=False, verbose=False, **kwargs)
    trainer.model.eval()

    """ get diffusion history """
    if render_history:
        pose_features, history = trainer.model.p_sample_loop(data, return_history=True)
        if torch.isnan(pose_features).any():
            pose_features, history = trainer.model.p_sample_loop(data, return_history=True)
        pose_features = trainer.get_all_features(pose_features, data)
        pose_features.clamp_(-1., 1.)

        ## render png of final result
        png_name = abspath(join(VISUALIZATION_PATH, 'compose_constraints', f'{key}_data.png'))
        render_kwargs = dict(world_dims=data.world_dims, world_name=world_name, log=True, show=False)
        render_kwargs['constraints'] = constraint_from_edge_attr(data.edge_attr, data.edge_index)
        evaluations = render_world_from_graph(pose_features, save=True, png_name=png_name, **render_kwargs)
        print(key, '\tviolated constraints:', evaluations)

        ## render the diffusion process
        gif_name = png_name.replace('.png', '.gif')
        trainer.render_success(milestone, 2, 0, 0, data, history, gif_file=gif_name,
                               save_mp4=True, **render_kwargs)

    """ get energy field """
    denoise_fn = trainer.model.denoise_fn
    denoise_fn, use_EBM_wrapper = get_denoise_fn(trainer)
    c_idx_1 = denoise_fn.constraint_sets.index(con_pair[0][0])
    c_idx_2 = denoise_fn.constraint_sets.index(con_pair[1][0])

    geoms_emb = denoise_fn.geom_encoder(geoms_in.to(denoise_fn.device))

    time_in = torch.tensor([t]).float().to(denoise_fn.device)
    time_emb = denoise_fn.time_mlp(time_in)  ## torch.Size([1, 256])

    me = con_pair[0][1]
    pose_A_rot = data.x[:, 4:][me].to(denoise_fn.device)

    def energy_fn_1(x, y):
        geoms_emb_1 = geoms_emb[torch.tensor(con_pair[0][1:])]  ## torch.Size([2, 256])
        poses_B = data.x[:, 2:][con_pair[0][2]].to(denoise_fn.device)
        return make_energy_fn(x, y, denoise_fn, trainer, c_idx_1, geoms_emb_1, time_emb,
                              use_EBM_wrapper, pose_A_rot, poses_B)

    def energy_fn_2(x, y):
        geoms_emb_2 = geoms_emb[torch.tensor(con_pair[1][1:])]  ## torch.Size([2, 256])
        poses_B = data.x[:, 2:][con_pair[1][2]].to(denoise_fn.device)
        return make_energy_fn(x, y, denoise_fn, trainer, c_idx_2, geoms_emb_2, time_emb,
                              use_EBM_wrapper, pose_A_rot, poses_B)

    others = [i for i in range(geoms_emb.shape[0]) if i not in [con_pair[0][1], 0]]
    c_idx_cfree = denoise_fn.constraint_sets.index('cfree')

    def energy_fn_cfree1(x, y):
        geoms_emb_1 = geoms_emb[torch.tensor([me, others[0]])]  ## torch.Size([2, 256])
        poses_B = data.x[:, 2:][others[0]].to(denoise_fn.device)
        return make_energy_fn(x, y, denoise_fn, trainer, c_idx_cfree, geoms_emb_1, time_emb,
                              use_EBM_wrapper, pose_A_rot, poses_B)

    def energy_fn_cfree2(x, y):
        geoms_emb_2 = geoms_emb[torch.tensor([me, others[1]])]  ## torch.Size([2, 256])
        poses_B = data.x[:, 2:][others[1]].to(denoise_fn.device)
        return make_energy_fn(x, y, denoise_fn, trainer, c_idx_cfree, geoms_emb_2, time_emb,
                              use_EBM_wrapper, pose_A_rot, poses_B)

    output_dir = abspath(join(VISUALIZATION_PATH, 'compose_constraints', run_id))
    os.makedirs(output_dir, exist_ok=True)
    plot_kwargs = dict(run_id=run_id, input_mode='qualitative', t=t, save_png=save_png,
                       output_dir=output_dir, use_toy_data=False)
    images = {}
    images[con_pair[0][0]] = make_plot(energy_fn_1, name=f"{key}_{con_pair[0][0]}", **plot_kwargs)
    images[con_pair[1][0]] = make_plot(energy_fn_2, name=f"{key}_{con_pair[1][0]}", **plot_kwargs)

    def noise_function(sshape):
        return torch.randn(sshape, device=denoise_fn.device)

    step_sizes = 2 * trainer.model._betas

    def composed_energy_fn(x, y):
        """ x: [10, 10], grad : [2, 10, 10] """
        poses_in = get_poses_in(x, y, denoise_fn)[:, 0, :2]  ## [100, 2]
        outputs = torch.stack([torch.tensor(x).flatten(), torch.tensor(y).flatten()], dim=1).to(denoise_fn.device)  ## [100, 2]
        for i in range(trainer.model.samples_per_step):
            ss = step_sizes[t]
            std = (2 * ss) ** .5
            grad = energy_fn_1(x, y)[2] + energy_fn_2(x, y)[2] + energy_fn_cfree1(x, y)[2] + energy_fn_cfree2(x, y)[2]
            grad = grad[:, 0, :2] * trainer.model._sqrt_recipm1_alphas_cumprod_custom[t]
            noise = noise_function(outputs.shape) * std
            outputs = outputs + grad * ss + noise
            x = outputs[:, 0].reshape(10, 10).cpu().detach().numpy()
            y = outputs[:, 1].reshape(10, 10).cpu().detach().numpy()
        energy = - torch.sum(outputs ** 2, dim=1)
        gradients = poses_in - outputs
        energy = energy.reshape(x.shape).cpu().detach().numpy()
        gradients = gradients.cpu().detach().numpy()  ## torch.Size([100, 2, 4]) -> (100, 2)
        return energy, gradients, outputs  ## [100, 2]

    images[key] = make_plot(composed_energy_fn, name=key, **plot_kwargs)
    return images


#####################################################################################


def get_denoise_fn(trainer):
    denoise_fn = trainer.model.denoise_fn
    use_EBM_wrapper = False
    if hasattr(denoise_fn, 'model') and not isinstance(denoise_fn.model, str):
        denoise_fn = denoise_fn.model
        use_EBM_wrapper = True
    # denoise_fn.eval()
    return denoise_fn, use_EBM_wrapper


def read_data():
    import torch

    # Load the file
    dataset_path = '/home/yang/Documents/HACL-PyTorch/projects/kitchenworld/data'
    dataset_name = 'TriangularRandomSplitWorld[64]_(10)_diffuse_pairwise_test_3_split'
    dataset_name = 'RandomSplitQualitativeWorld(10)_qualitative_test_2_split'
    raw_data = join(dataset_path, dataset_name, "raw/data_0.pt")
    processed_data = join(dataset_path, dataset_name, "processed/data.pt")

    data = torch.load(processed_data)

    # Print the head of the file
    print(data.edge_index)


def get_qualitative_config():
    input_mode = 'qualitative'
    run_id, milestone, timesteps = 'r26wkb13', 16, 1000
    run_id, milestone, timesteps = 'qsd3ju74', 7, 1000  ## non EBM form, Aug 28

    if not isdir(join(OUTPUT_PATH, run_id)):
        os.makedirs(join(OUTPUT_PATH, run_id))
    if not isdir(join(OUTPUT_PATH, run_id, 'data')):
        os.makedirs(join(OUTPUT_PATH, run_id, 'data'))

    return input_mode, run_id, milestone, timesteps


def crop_sample_images(data_dir, w=None, padding=20):
    from PIL import Image
    new_width = new_height = w
    for f in os.listdir(data_dir):
        if f.endswith('.png'):
            im = Image.open(join(data_dir, f))
            width, height = im.size  # Get dimensions

            if w is not None:
                left = (width - new_width) / 2
                top = (height - new_height) / 2
                right = (width + new_width) / 2
                bottom = (height + new_height) / 2
            else:
                if 'data' not in f or 'cropped' in f:
                    continue
                left = padding
                top = padding
                right = width - padding
                bottom = height - padding // 3

            # Crop the center of the image
            im = im.crop((left, top, right, bottom))
            im.save(join(data_dir, f.replace('.png', '_cropped.png')))


def generate_gif_plots(input_mode, run_id, milestone, name, timestep, plot_kwargs):
    imgs = []
    for t in range(0, timestep * 2, timestep):
        img = plot_diffusion_by_name(run_id, milestone, name=name, t=t, render_history=(t == 0), **plot_kwargs)
        imgs.append(img)
    file_name = join(OUTPUT_PATH, run_id, f'{input_mode}_{name}_fields.mp4')
    img_files_to_mp4(imgs, file_name)
    for img in imgs:
        os.remove(img)
    return file_name


def generate_qualitative_plots_one_fold(save_gif=False, use_saved_data=False, render_history=False, **kwargs):
    input_mode, run_id, milestone, timesteps = get_qualitative_config()
    timestep = timesteps // 10
    plot_kwargs = dict(input_mode=input_mode, t=0, save_png=True, use_container=True)

    items = {}
    for name in tqdm(qualitative_constraints):

        # if name not in ['left-in', 'v-aligned', 'cfree', 'in']:
        #     continue
        #
        # use_saved_data = True ## name not in ['close-to']

        ## save gif of energy change with diffuse steps
        if save_gif:
            viz = generate_gif_plots(input_mode, run_id, milestone, name, timestep, plot_kwargs)

        elif use_saved_data:
            viz = make_plot(None, run_id=run_id, name=name, draw_title=False, **plot_kwargs)

        else:
            viz = plot_diffusion_by_name(run_id, milestone, name=name,
                                         render_history=render_history, **plot_kwargs)

        items[name] = viz

    data_dir = join(OUTPUT_PATH, run_id)
    for f in os.listdir(OUTPUT_PATH):
        if f.endswith('.png') or f.endswith('.mp4') or f.endswith('.json'):
            shutil.move(join(OUTPUT_PATH, f), join(data_dir, f))

    # print(items)

    html_file = join(VISUALIZATION_PATH, f'gradient_fields__{input_mode}_{run_id}.html')
    make_html(items, input_mode, html_file, **kwargs)
    crop_sample_images(data_dir, w=950)


def generate_qualitative_plots_two_fold():
    task_name = 'RandomSplitQualitativeWorld(100)_qualitative_test_3_split'
    input_mode, run_id, milestone, timesteps = get_qualitative_config()
    plot_kwargs = dict(t=0, save_png=True, render_history=False)

    output_dir = join(VISUALIZATION_PATH, 'compose_constraints', run_id)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(join(output_dir, 'data'), exist_ok=True)

    composed_pairs_json = join(VISUALIZATION_PATH, 'compose_constraints', 'source', f"{task_name}.json")
    data_dir = join(DATA_PATH, task_name, 'raw')
    constraint_pairs = json.load(open(composed_pairs_json, 'r'))
    originals = [(abspath(join(RENDER_PATH, task_name, f"idx={v[0][0]}.png")), v[0][1:]) for v in constraint_pairs.values()]

    # history_dir = join(VISUALIZATION_PATH, 'compose_constraints')
    # for key, pairs in tqdm(constraint_pairs.items()):
    #     # if key in ['bottom-in|h-aligned', 'away-from|right-in', 'left-in|left-of', 'center-in|left-of',
    #     #            'center-in|top-of', 'away-from|h-aligned', 'away-from|top-of', 'h-aligned|top-in',
    #     #            'left-in|v-aligned', 'bottom-in|v-aligned', 'away-from|left-in', 'h-aligned|right-in',
    #     #            'top-in|v-aligned', 'left-of|left-of', 'close-to|right-in', 'away-from|bottom-in',
    #     #            'bottom-in|close-to']:
    #     #     continue
    #     if key not in ['away-from|right-in']:
    #         continue
    #     pair = pairs[0]
    #     ind, con1, con2 = pair
    #     data_pt = join(data_dir, f'data_{ind}.pt')
    #     plot_diffusion_by_pt(run_id, milestone, data_pt, key, pair[1:], **plot_kwargs)
    #
    #     for f in os.listdir(history_dir):
    #         if key in f and (f.endswith('.png') or f.endswith('.mp4') or f.endswith('.json')):
    #             shutil.move(join(history_dir, f), join(output_dir, f))
    #
    # html_file = join(VISUALIZATION_PATH, f'composed_gradient_fields__{input_mode}_{run_id}.html')
    # make_html_composed(list(constraint_pairs.keys()), originals, input_mode, html_file, output_dir)
    # crop_sample_images(output_dir, padding=130)
    for key in constraint_pairs:
        print(key.replace('|', ' & '))


if __name__ == '__main__':

    ## ------ test
    # energy_fn = get_test_energy_fn()
    # make_plot(energy_fn)

    # read_data()

    # ## ------ triangle
    # for name in puzzle_constraints:
    #     plot_diffusion_by_name('dyz4eido', 23, name=name, input_mode='triangle')

    train_task = "RandomSplitQualitativeWorld(30000)_qualitative_train"
    visualize_qualitative_distribution(train_task=train_task)
    generate_qualitative_plots_one_fold(train_task=train_task, render_history=False)

    # generate_qualitative_plots_two_fold()
