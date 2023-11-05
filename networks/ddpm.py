import json
import math
import copy
import sys
import torch
from torch import nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import time
import pdb

import os
from os.path import join, isdir, abspath, isfile, dirname, basename
from torch_geometric.loader import DataLoader
from pathlib import Path
from torch.optim import Adam, lr_scheduler
from torchvision import transforms, utils
import imageio
from PIL import Image
from collections import defaultdict

import numpy as np
from tqdm import tqdm


from data_utils import render_world_from_graph, print_tensor, constraint_from_edge_attr, translate_cfree_evaluations

try:
    from apex import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False

# helpers functions
HMC_USE_TORCH = True


def to_np(x):
    return x.detach().cpu().numpy()


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def cycle(dl):
    while True:
        for data in dl:
            yield data


def loss_backwards(fp16, loss, optimizer, **kwargs):
    if fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(**kwargs)
    else:
        loss.backward(**kwargs)

# small helper modules


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def images_to_gif(img_arrs, gif_file, crop=None):
    import imageio
    with imageio.get_writer(gif_file, mode='I') as writer:
        for img in img_arrs:
            if crop is not None:
                left, top, right, bottom = crop
                img = img[top:bottom, left:right]
            writer.append_data(img)
    return gif_file


def images_to_mp4(img_arrs, mp4_file, fps=25):
    import cv2
    h, w, _ = np.asarray(img_arrs[0]).shape
    out = cv2.VideoWriter(mp4_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h), True)
    for frame in img_arrs:
        out.write(frame[:, :, :3][:, :, ::-1])
    cv2.destroyAllWindows()
    out.release()


# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def conditional_noise(x, mask):
    noise = torch.randn_like(x).to(x.device)
    noise[mask.bool()] = 0
    return noise


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


def simple_linear_schedule(t, clip_min=1e-9):
    """
    implemented by https://arxiv.org/pdf/2301.10972.pdf
    """
    # A gamma function that simply is 1-t.
    return np.clip(1 - t, clip_min, 1.)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_schedule(timesteps, start=-3, end=3, tau=1.0, clip_min=1e-9):
    """
    described in https://arxiv.org/pdf/2301.10972.pdf
    """
    # A gamma function based on sigmoid function.
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    v_start = sigmoid(start / tau)
    v_end = sigmoid(end / tau)
    output = sigmoid((x * (end - start) + start) / tau)
    betas = (v_end - output) / (v_end - v_start)
    return np.clip(betas, clip_min, 0.999)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min=0, a_max=0.999)


######################################################################


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        timesteps=100,
        loss_type='l2',
        EBM=False,
        betas=None,
        samples_per_step=10,
        step_sizes='2*self.betas'
    ):
        super().__init__()

        if exists(betas):
            betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        else:
            betas = cosine_beta_schedule(timesteps)
        self._betas = betas

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.denoise_fn = denoise_fn
        self.device = denoise_fn.device
        self.dims = denoise_fn.dims
        self.input_mode = denoise_fn.input_mode
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type
        self.EBM = EBM

        to_torch = partial(torch.tensor, dtype=torch.float32, device=self.device)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        self.samples_per_step = samples_per_step
        self.step_sizes = eval(step_sizes)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))
        self._sqrt_recipm1_alphas_cumprod_custom = to_torch(np.sqrt(1. / (1 - alphas_cumprod)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        self.sample_loop_time = []

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, batch, features, t, clip_denoised: bool=False, **kwargs):
        """ batch process ConstraintGraphData """
        all_noise = self.denoise_fn(features, batch, t, eval=True, **kwargs)
        all_features_recon = self.predict_start_from_noise(features, t=t, noise=all_noise)
        if clip_denoised:
            all_features_recon.clamp_(-1., 1.)
        return self.q_posterior(x_start=all_features_recon, x_t=features, t=t)

    def p_sample(self, batch, all_features, t, repeat_noise=False, **kwargs):
        model_mean, _, model_log_variance = self.p_mean_variance(batch, all_features, t, **kwargs)
        noise = noise_like(all_features.shape, self.device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).item())
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def p_sample_loop(self, batch, return_history=False, **kwargs):

        if self.denoise_fn.energy_wrapper:
            torch.set_grad_enabled(True)
        else:
            torch.set_grad_enabled(False)

        device = self.device
        m = batch.mask.to(device)
        gt_features = batch.x[:, self.dims[-1][1]:self.dims[-1][2]].to(device)
        shape = gt_features.shape

        ## random noise
        pose_features = 0.5 * torch.randn(shape, device=device)
        pose_features[m.bool()] = gt_features[m.bool()].clone()

        ## HMC sampler
        if self.EBM:

            def gradient_function(x, batch, t):
                gradient = self.denoise_fn.ev[0] * self.denoise_fn(x, batch, t, eval=True) \
                           * self._sqrt_recipm1_alphas_cumprod_custom[t]
                # print('gradient_function', x.shape, gradient.shape)
                return gradient

            def energy_function(x, batch, t):
                score = self.denoise_fn.ev[1] * self.denoise_fn.neg_logp_unnorm(x, batch, t, eval=True) \
                        * self._sqrt_recipm1_alphas_cumprod_custom[t]
                print('energy_function', score.shape)
                return score

            def noise_function():
                return torch.randn(shape, device=device)

            if 'ULA' in self.EBM:
                samples_per_step = self.samples_per_step
                step_sizes = self.step_sizes
                if self.EBM == 'ULA+':
                    n = self.num_timesteps // 4
                    samples_per_step = torch.tensor([4]*n + [8]*n + [12]*n + [16]*n)
                    # step_sizes[-2*n:-n] *= 2
                    # step_sizes[-n//2:] *= 2
                sampler = AnnealedULASampler(samples_per_step, step_sizes, gradient_function, noise_function)

            elif self.EBM == 'MALA':
                samples_per_step = self.samples_per_step
                step_sizes = self.step_sizes
                """ 
                0.0000086 -> 0.94, 0.000008 -> 0.945, 0.00001 -> 0.939
                """
                sampler = AnnealedMALASampler(samples_per_step, step_sizes, gradient_function, noise_function,
                                              energy_function)

            elif self.EBM == 'HMC':
                samples_per_step = 4
                step_sizes = self.step_sizes
                mass_diag_sqrt = 9 * self.betas
                damping = 0
                num_leapfrog = 2
                sampler = AnnealedMUHASampler(samples_per_step, step_sizes, damping, mass_diag_sqrt, num_leapfrog,
                                              gradient_function=gradient_function, energy_function=energy_function)

        ## denoising process
        if return_history:
            history = [pose_features]
        for j in reversed(range(0, self.num_timesteps)):
            t = torch.full((1, ), j, device=device, dtype=torch.long)

            assert not self.training
            pose_features = self.p_sample(batch, pose_features, t, tag='EBM', **kwargs)  ## , tag=f'test_{j}'
            if self.EBM and j % self.denoise_fn.ebm_per_steps == 0:
                pose_features = sampler.sample_step(pose_features, batch, t)
                # print(f'p_sample_loop {j}/{self.num_timesteps}')

            pose_features[m.bool()] = gt_features[m.bool()].clone()
            if return_history:
                history.append(pose_features)

        if return_history:
            return pose_features, history
        return pose_features

    @torch.no_grad()
    def sample(self, batch, **kwargs):
        start = time.time()
        outputs = self.p_sample_loop(batch, **kwargs)
        passed = time.time() - start
        self.sample_loop_time.append(passed)
        if len(self.sample_loop_time) > 10:
            self.sample_loop_time.pop(0)
        # print(f'sample time: {passed:.3f} sec')
        return outputs

    def q_sample(self, x_start, mask, t, noise=None):
        noise = default(noise, lambda: conditional_noise(x_start, mask))
        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        ## mask out some pose features
        sample[mask.bool()] = x_start[mask.bool()]
        return sample

    def p_losses(self, batch, t, noise=None, **kwargs):

        """ generate noise based on those unconditioned features """
        m = batch.mask.to(self.device)
        pose_features = batch.x[:, self.dims[-1][1]:self.dims[-1][2]].to(self.device)
        all_noise = default(noise, lambda: conditional_noise(pose_features, m))
        all_features_noisy = self.q_sample(pose_features, m, t=t, noise=all_noise)

        """ reconstruct the unconditioned """
        all_features_recon = self.denoise_fn(all_features_noisy, batch, t, **kwargs)

        if kwargs['debug']:
            print_tensor('all_noise', all_noise[:3])
            print_tensor('all_features_recon', all_features_recon[:3])

        assert all_noise[0].shape == all_features_recon[0].shape
        if self.loss_type == 'l1':
            loss = (all_noise - all_features_recon).abs().mean()
        elif self.loss_type == 'l2':
            loss = F.mse_loss(all_noise, all_features_recon)
        else:
            raise NotImplementedError()
        return loss

    def forward(self, batch, **kwargs):
        t = torch.randint(0, self.num_timesteps, (1, ), device=self.device).long()
        return self.p_losses(batch, t, **kwargs)


# trainer class

class Trainer(object):
    def __init__(
        self,
        denoise_fn,
        train_dataset,
        test_datasets,
        render_dir,
        *,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-3,
        train_num_steps=100000,
        gradient_accumulate_every=2,
        fp16=False,
        step_start_ema=2000,
        update_ema_every=10,
        save_and_sample_every=10000,
        results_folder='./results',
        EBM=False,
        visualize=False,
        use_wandb=False,
        rejection_sampling=False,
        tamp_pipeline=False,
        eval_only=False,
        input_mode=None,
        **kwargs
    ):
        super().__init__()
        self.model = denoise_fn
        self.dims = denoise_fn.dims
        self.input_mode = denoise_fn.input_mode
        self.ema = EMA(ema_decay)
        self.ema_model = None ## copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps
        self.EBM = EBM
        self.use_wandb = use_wandb
        self.visualize = visualize

        ## different evaluation mode
        self.rejection_sampling = rejection_sampling  ## use model.sample_rejection() instead of model.sample()
        self.tamp_pipeline = tamp_pipeline  ## change data_loader to include all placement sequences

        dl_kwargs = dict(pin_memory=True, num_workers=0)
        self.train_dl = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, **dl_kwargs)

        self.test_dls = {
            k: [DataLoader(d, batch_size=b, shuffle=False, **dl_kwargs) for b in [100, 1]]
            for k, d in test_datasets.items()
        }
        self.eval_kwargs = dict(tries=(10, 0))
        self.num_test_samples = 0 if len(test_datasets) == 0 else len(list(test_datasets.values())[0])

        self.render_dir = render_dir
        if not isdir(self.render_dir):
            os.makedirs(self.render_dir, exist_ok=True)

        self.world_name = 'TriangularRandomSplitWorld' if 'Triangular' in self.render_dir else 'RandomSplitWorld'
        if 'robot' in self.input_mode:
            self.world_name = 'TableToBoxWorld'
        elif 'stability' in self.input_mode:
            self.world_name = 'RandomSplitWorld'
        elif 'qualitative' in self.input_mode:
            self.world_name = 'RandomSplitQualitativeWorld'

        self.opt = Adam(denoise_fn.parameters(), lr=train_lr)
        # self.scheduler = lr_scheduler.CosineAnnealingLR(self.opt, T_max=10000)
        # self.scheduler = lr_scheduler.StepLR(self.opt, step_size=1000, gamma=0.99)
        self.step = 0

        assert not fp16 or fp16 and APEX_AVAILABLE, 'Apex must be installed in order for mixed precision training to be turned on'

        self.fp16 = fp16
        if fp16:
            (self.model, self.ema_model), self.opt = amp.initialize([self.model, self.ema_model], self.opt, opt_level='O1')

        self.results_folder = Path(results_folder)
        if not eval_only:
            self.results_folder.mkdir(exist_ok=True)
        elif not isdir(self.results_folder):
            self.results_folder = Path(results_folder.replace('logs', 'logs2'))

        self.reset_parameters()

    def reset_parameters(self):
        if self.ema_model is not None:
            self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.ema_model is None:
            return
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, milestone):
        milestone = milestone ## (milestone // 50) * 50
        data = {'step': self.step, 'model': self.model.state_dict()}
        if self.ema_model is not None:
            data['ema'] = self.ema_model.state_dict()
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'))

        self.step = data['step']
        state_dict = data['model']
        new_state_dict = {}
        for k, v in state_dict.items():
            if 'denoise_fn.model' in k and False:
                new_state_dict[k.replace(k, k.replace('.model', ''))] = v
            else:
                new_state_dict[k] = v
        self.model.load_state_dict(new_state_dict)

        # if self.ema_model is not None:
        #     self.ema_model.load_state_dict(data['ema'])

    def train(self):
        backwards = partial(loss_backwards, self.fp16)

        import jacinle
        meters = jacinle.GroupMeters()

        dl_iter = iter(self.train_dl)
        while self.step < self.train_num_steps:
            for i in range(self.gradient_accumulate_every):
                try:
                    data = next(dl_iter)
                except StopIteration:
                    dl_iter = iter(self.train_dl)
                    data = next(dl_iter)
                loss = self.model(data, debug=False, tag='EBM')  ##  if self.EBM else 'train'
                meters.update(loss=loss.item())
                backwards(loss / self.gradient_accumulate_every, self.opt)

            if (self.step + 1) % 1000 == 0:
                lr = self.opt.state_dict()['param_groups'][0]['lr']
                print(meters.format_simple(f'Step: {self.step} | lr: {lr}\t', compressed=True))
                meters.reset()

            self.opt.step()
            self.opt.zero_grad()
            # self.scheduler.step()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.save_and_sample_every == (self.save_and_sample_every - 1):
                milestone = self.step // self.save_and_sample_every
                self.save(milestone)
                self.evaluate(milestone, **self.eval_kwargs)

            self.step += 1

        print('training completed')

    def evaluate(self, milestone, tries=(10, 0), verbose=False, return_history=False,
                 save_log=False, run_all=False, run_only=False, resume_eval=False, **kwargs):
        if 'robot' in self.input_mode or 'stability' in self.input_mode:
            sys.path.append(dirname(dirname(abspath("__file__"))))
            from demo_utils import render_robot_world_from_graph, render_stability_world_from_graph

        self.model.eval()
        json_name = join(self.render_dir, f'denoised_t={milestone}.json')
        once = False
        skip_the_rest = False
        composed_inference = 'robot' in self.input_mode and 'qualitative' in self.input_mode

        log = {}
        if resume_eval and isfile(json_name):
            log = json.load(open(json_name, 'r'))

        for i, test_dls in self.test_dls.items():
            success_list = []
            succeeded_graph_indices = []
            success_rounds = {}
            sampling_time = []
            all_failure_modes = {}
            percentage = 0

            ## add new num_object so only ran that one
            if resume_eval and i in log:
                continue

            ## run a specific test data point
            if run_only and i != run_only[0]:
                continue

            ## first run everything, then only those failed ones
            for m, test_dl in enumerate(test_dls):
                count = 0

                ## for test datasets involving a larger number of objects
                for n, data in enumerate(test_dl):

                    ## for each set of objects, we add-noise + denoise 3 times
                    for k in range(tries[m]):

                        if m == 0 and len(succeeded_graph_indices) == self.num_test_samples:
                            break
                        elif m == 1:
                            k += tries[0]
                            # if n in succeeded_graph_indices:
                            #     continue

                        batch = data.clone()
                        world_dims = batch.world_dims[0]
                        all_failure_modes[k] = {}

                        start = time.time()
                        result = self.model.sample(batch, debug=once, return_history=return_history)
                        once = False
                        passed = time.time() - start

                        if return_history:
                            all_features, history = result
                        else:
                            all_features, history = result, None
                        all_features.clamp_(-1., 1.)
                        all_features = self.get_all_features(all_features, batch)

                        graph_indices = batch.x_extract.unique().int().numpy().tolist()

                        passed_ave = passed / len(graph_indices)
                        print_out = f'[i={i}, k={k}, n={len(graph_indices)}] sampled for {passed:.2f} sec'
                        sampling_time.append((passed, len(graph_indices), passed_ave))
                        if m == 0:
                            print_out += f' on average {passed_ave:.2f} sec per graph'
                        else:
                            print_out += f' (j = {graph_indices[0]})'

                        for j in graph_indices:

                            if run_only and j != run_only[1]:
                                continue
                            count += 1
                            features = all_features[torch.where(batch.x_extract == j)]

                            # ## debug ## TODO: remove
                            # features = batch.x[torch.where(batch.x_extract == j)]

                            world_dims = batch.world_dims[j]
                            if torch.isnan(features).any():
                                continue
                            if m == 0 and j in succeeded_graph_indices:
                                continue

                            """ check cfree, maybe save image or json """
                            ## for robot collisions, use pybullet to check, and the above for generating png
                            output_name = f'denoised_t={milestone}_n={i}_i={j}_k={k}'
                            success = None
                            if 'stability' in self.input_mode:
                                prediction_json = join(self.render_dir, f'{output_name}.json')
                                mmask = (batch.edge_extract == j) & (batch.edge_attr == 1)
                                supports = batch.edge_index.T[torch.where(mmask)]
                                offset = supports.min()
                                supports -= offset
                                kkwargs = dict(world_dims=world_dims, supports=supports.T)
                                kkwargs.update(dict(render=self.visualize, mp4=self.visualize, png=self.visualize))
                                success = render_stability_world_from_graph(features, prediction_json, **kkwargs)

                            elif 'robot' in self.input_mode:
                                prediction_json = join(self.render_dir, f'{output_name}_solution.json')
                                success = render_robot_world_from_graph(features, prediction_json,
                                                                        gif=composed_inference or len(test_dl) == 1,
                                                                        save_trajectory=True,
                                                                        render=True, record_video=True
                                                                        )

                            ## may need to compose
                            if success is None or 'stability' in self.input_mode or composed_inference:
                                png_name = join(self.render_dir, f'{output_name}.png')
                                if 'stability' in self.input_mode:
                                    png_name = png_name.replace('.png', '_before.png')
                                render_kwargs = dict(world_dims=world_dims, world_name=self.world_name,
                                                     png_name=png_name, save=self.visualize, log=save_log, show=False)

                                ## change features
                                if composed_inference:
                                    """
                                    w, l, h, w0, l0, h0, x0, y0, model_id, scale, g1, g2, g3, g4, g5, grasp_id, x, y, z, sn, cs = f[:21] 
                                    w, l, x, y, sn, cs
                                    """
                                    if success and verbose:
                                        print('\n', basename(png_name))
                                        print_tensor('3D_features', features)
                                    features = torch.cat([features[:, :2], features[:, 16:18], features[:, 19:21]], dim=1)
                                    if success and verbose:
                                        print_tensor('2D_features', features)

                                    render_kwargs.update(dict(
                                        world_name='RandomSplitQualitativeWorld', show_grid=False, show=False
                                    ))
                                if 'qualitative' in self.input_mode:
                                    edge_index = batch.edge_index[:, torch.where(batch.edge_extract == j)[0]]
                                    edge_attr = batch.edge_attr[torch.where(batch.edge_extract == j)]
                                    offset = edge_index.min()
                                    edge_index -= offset
                                    constraints = constraint_from_edge_attr(edge_attr, edge_index,
                                                                            composed_inference=composed_inference)
                                    render_kwargs.update(dict(constraints=constraints))

                                evaluations = render_world_from_graph(features, **render_kwargs)
                                if verbose:
                                    print(i, j, k, '\t', evaluations)

                                if composed_inference:
                                    # print('\trobot constraints', success)
                                    # print('\tqualitative constraints',  (len(evaluations) == 0))
                                    success = success and (len(evaluations) == 0)
                                elif 'stability' not in self.input_mode:
                                    success = (len(evaluations) == 0)

                            """ log success """
                            if success:
                                ## computing statistics
                                success_list.append((j, k))
                                if j not in success_rounds:
                                    succeeded_graph_indices.append(j)
                                    success_rounds[j] = k

                                ## save an animation of the denoising process
                                if return_history and self.visualize:  ##  and False
                                    self.render_success(milestone, i, j, k, batch, history, world_name=self.world_name, **kwargs)

                            elif 'qualitative' in self.input_mode:
                                from denoise_fn import qualitative_constraints
                                if len(evaluations) > 0 and len(evaluations[0]) == 2:
                                    evaluations = translate_cfree_evaluations(evaluations)
                                else:
                                    evaluations = [e for e in evaluations if e[0] in qualitative_constraints]
                                all_failure_modes[k][j] = evaluations

                            elif 'diffuse_pairwise' in self.input_mode:
                                from data_utils import yaw_from_sn_cs

                                gt = batch.x[torch.where(batch.x_extract == j)]
                                gt[:, :2] *= world_dims[0]
                                gt[:, 2] *= world_dims[1]
                                gt[:, 3] *= world_dims[0] / 2
                                gt[:, 4] *= world_dims[1] / 2
                                yaws = yaw_from_sn_cs(gt[:, -1], gt[:, -2]).detach().cpu().numpy().tolist()
                                yaws[0] = 0
                                predicted_yaws = yaw_from_sn_cs(features[:, -1], features[:, -2]).detach().cpu().numpy().tolist()
                                predicted_yaws[0] = 0
                                collisions = []
                                gt = gt[:, :5].detach().cpu().numpy().tolist()
                                for ev in evaluations:
                                    collision = []
                                    for evv in ev:  ## ['west', 'triangle_tile_2']
                                        idx = 0 if '_' not in evv else eval(evv.split('_')[-1]) + 1
                                        collision.append(gt[idx] + [yaws[idx]] + [predicted_yaws[idx]])
                                    collisions.append(collision)
                                all_failure_modes[str((i, j, k))] = collisions

                            if 'robot' in self.input_mode:
                                for suffix in ['solution.json', 'solution.pkl', 'solution.gif', 'diffusion.json']:
                                    ffile = join(self.render_dir, f'{output_name}_{suffix}')
                                    if success:
                                        ddir = join(self.render_dir, f'{output_name}')
                                        os.makedirs(ddir, exist_ok=True)
                                        if isfile(ffile):
                                            os.rename(ffile, join(ddir, suffix))
                                    elif isfile(ffile):
                                        os.remove(ffile)

                        del batch

                        percentage = len(succeeded_graph_indices) / self.num_test_samples
                        print_out += f"\t solved {len(succeeded_graph_indices)} / {self.num_test_samples} graphs " \
                                     f"({percentage:.2f})"
                        print(print_out)

                        if m == 0 and count != 0:
                            self.summarize_success_rate(i, success_list, self.num_test_samples, succeeded_graph_indices,
                                                        success_rounds, log, tries=tries[0], send_wandb=False)

                        log[i]['success_rounds'] = success_rounds
                        log[i]['sampling_time'] = sampling_time
                        log[i]['all_failure_modes'] = all_failure_modes
                        log[i]['eval_tries'] = k
                        log[i]['visualize'] = self.visualize
                        with open(json_name, 'w') as f:
                            json.dump(log, f)

                ## compute and log the success rate and average sample time
                if m == 0 and count != 0:
                    self.summarize_success_rate(i, success_list, self.num_test_samples, succeeded_graph_indices,
                                                success_rounds, log, tries=tries[0], send_wandb=True)

                ## for printing in matplotlib
                if save_log: print('saved', json_name)
                with open(json_name, 'w') as f:
                    json.dump(log, f)

                # data.render_cn(self.render_dir, file_name=f'denoised_cn_{i}_{self.step}')
                if (not run_all and percentage == 0) or (run_only and len(success_list) == 1):
                    skip_the_rest = True
                    break
            print()
            if skip_the_rest:
                break

        self.model.train()

    def get_all_features(self, all_features, batch):
        """ replace pose features with generated outputs """

        if self.input_mode == 'diffuse_pairwise_image':
            all_features = torch.cat([
                batch.x[:, :self.dims[0][0]].cpu(),
                all_features.detach().cpu()
            ], dim=1)
        else:
            all_features = torch.cat([
                batch.x[:, :self.dims[-1][1]].cpu(),
                all_features.detach().cpu(),
                batch.x[:, self.dims[-1][2]:].cpu(),
            ], dim=1)
        return all_features

    def summarize_success_rate(self, i, success_list, count, succeeded_graph_indices, success_rounds,
                               log, tries=2, send_wandb=False):
        import wandb
        top1 = round(len([s for s in success_rounds.values() if s == 0]) / count, 3)
        topk = round(len(succeeded_graph_indices) / count, 3)
        # topk = {k: len([j for j in range(tries) if (k, j) in success_list]) for k in range(count)}
        # topk = round(len([k for k in topk if topk[k] > 0]) / count, 3)
        ave_sample_time = sum(self.model.sample_loop_time) / len(self.model.sample_loop_time) / count
        if i not in log:
            log[i] = {}
        log[i].update({
            'success': success_list, 'success_rate': top1, 'success_rate_top3': topk,
            'model_ave_sample_time': ave_sample_time
        })
        if send_wandb:
            self.model.sample_loop_time = []
            if self.use_wandb:
                wandb.log({
                    f"Accuracy ({i} obj)": top1, f"Top 3 Accuracy ({i} obj)": topk, 'sample_time': ave_sample_time
                })
            print(f'success on {i} objects: ', log[i]['success_rate_top3'], '\n')

    def render_success(self, milestone, i, j, k, batch, history, n_frames=50,
                       gif_file=None, save_mp4=False, save_history=True, **kwargs):
        from mesh_utils import GREEN, RED
        if gif_file is None:
            gif_file = join(self.render_dir, f'denoised_t={milestone}_n={i}_i={j}_k={k}.gif')
        gif_file = abspath(gif_file)

        img_arrs = []
        n_steps = len(history)
        n_frames = min(n_frames, n_steps)
        gap = int(n_steps / n_frames)
        world_dims = batch.world_dims
        if isinstance(world_dims[0], tuple) or isinstance(world_dims[0], list):
            world_dims = world_dims[0]
        if 'world_dims' in kwargs:
            kwargs.pop('world_dims')
        if 'render' in kwargs:
            kwargs.pop('render')

        start = time.time()
        all_object_states = []
        all_evaluations = []
        all_steps = []
        for step in range(n_steps):
            if step % gap != 0 and step != n_steps - 1 and step < n_steps - 50:
                continue
            all_features = history[step]
            all_features = self.get_all_features(all_features, batch)
            features = all_features[torch.where(batch.x_extract == j)]
            if 'robot' not in self.input_mode:
                features.clamp_(-1., 1.)
            img_arr, object_states, evaluations = render_world_from_graph(features, world_dims, array=True, **kwargs)

            ## add a progress bar
            h, w, _ = img_arr.shape
            top, bottom, left, right = h - 50, h, 0, int(w * step / n_steps)
            img_arr[top:bottom, left:right] = GREEN if (len(evaluations) == 0) else RED
            img_arrs.append(img_arr)

            ## add the current poses for gym visualization
            all_object_states.append(object_states)
            all_evaluations.append(evaluations)
            all_steps.append(step)

        if save_mp4:
            images_to_mp4(img_arrs, gif_file.replace('.gif', '.mp4'))
        else:
            images_to_gif(img_arrs, gif_file)

        print(f'... saved to {gif_file} with {len(img_arrs)} frames in {round(time.time() - start, 2)} seconds')

        if save_history:
            history_file = gif_file.replace('.gif', '_diffusion.json')
            history = {
                'object_states': all_object_states,
                'evaluations': all_evaluations,
                'steps': all_steps
            }
            with open(history_file, 'w') as f:
                json.dump(history, f)


##########################################################################################


from typing import Callable, Optional, Tuple, Union, Dict

Array = np.ndarray
RandomKey = Array
GradientTarget = Callable[[Array, Array], Array]


def leapfrog_step(x_0: Array,
                  v_0: Array,
                  gradient_target: GradientTarget,
                  step_size: Array,
                  mass_diag_sqrt: Array,
                  num_steps: int):
    """ Multiple leapfrog steps with no metropolis correction. """
    x_k = x_0
    v_k = v_0
    if mass_diag_sqrt is None:
        mass_diag_sqrt = np.ones_like(x_k)

    mass_diag = mass_diag_sqrt ** 2.

    for _ in range(num_steps):  # Inefficient version - should combine half steps
        v_k = v_k + 0.5 * step_size * gradient_target(x_k).detach() ##.cpu().numpy()  # half step in v
        x_k = x_k + step_size * v_k / mass_diag  # Step in x
        grad = gradient_target(x_k).detach() ## .cpu().numpy()
        v_k = v_k + 0.5 * step_size * grad  # half step in v
    # print('changes', np.linalg.norm(x_k - x_0), np.linalg.norm(v_k - v_0), '\tstep_size', step_size)
    return x_k, v_k


class AnnealedULASampler:
    """ Implements AIS with ULA """

    def __init__(self,
                 num_samples_per_step,
                 step_sizes,
                 gradient_function,
                 noise_function):
        self._step_sizes = step_sizes
        if isinstance(num_samples_per_step, int):
            num_samples_per_step = torch.tensor([num_samples_per_step] * len(step_sizes))
        self._num_samples_per_step = num_samples_per_step
        self._gradient_function = gradient_function
        self._noise_function = noise_function

    @torch.enable_grad()
    def sample_step(self, x, batch, t):
        if self._num_samples_per_step.device != t.device:
            self._num_samples_per_step = self._num_samples_per_step.to(t.device)
        for i in range(self._num_samples_per_step[t]):
            ss = self._step_sizes[t]
            std = (2 * ss) ** .5
            grad = self._gradient_function(x, batch, t)
            noise = self._noise_function() * std
            x = x + grad * ss + noise

        return x


class MetropolisSampler:

    def __init__(self, num_samples_per_step, step_sizes, debug=True):
        self._num_samples_per_step = num_samples_per_step
        if isinstance(step_sizes, torch.Tensor):
            step_sizes = step_sizes.detach().cpu().numpy()
        self._step_sizes = step_sizes
        self._cumulative_accept_rates = []
        self.debug = debug

    def _update_acceptance_rate(self, accept_rate, t, debug=False):
        """
        if acceptance rate is too high, increase mass_diag_sqrt or la_step_sizes
        """
        if len(self._cumulative_accept_rates) > 300:
            self._cumulative_accept_rates.pop(0)
        self._cumulative_accept_rates.append(accept_rate)
        average_accept_rate = sum(self._cumulative_accept_rates) / len(self._cumulative_accept_rates)
        step_sizes = self._step_sizes[:self._num_samples_per_step]
        ave_step_sizes = sum(step_sizes) / len(step_sizes)
        if self.debug and (t == 0 or debug):
            something = self._cumulative_accept_rates[-10:] if \
                len(self._cumulative_accept_rates) > 10 else self._cumulative_accept_rates
            print(f'acceptance rate at t={t.item()} = {round(accept_rate, 3)} \t '
                  f'on average = {round(average_accept_rate, 3)}\t',
                  f'step size = {round(ave_step_sizes, 7)}\t',
                  # [round(s, 3) for s in something]
                  )


class AnnealedMALASampler(MetropolisSampler):
    """ Implements AIS with MALA """

    def __init__(self,
                 num_samples_per_step,
                 step_sizes,
                 gradient_function,
                 noise_function,
                 energy_function):
        super().__init__(num_samples_per_step, step_sizes)
        self._gradient_function = gradient_function
        self._noise_function = noise_function
        self._energy_function = energy_function

    def sample_step(self, x, batch, t):

        accept_rate = []
        for i in range(self._num_samples_per_step):
            ss = self._step_sizes[t]
            std = (2 * ss) ** .5
            grad = self._gradient_function(x, batch, t)
            noise = self._noise_function() * std
            mu = x + grad * ss
            scale = torch.ones_like(x) * std
            x_hat = mu + noise

            ## compute previous and current logp(x)
            logp_x = self._energy_function(x, batch, t)
            logp_x_hat = self._energy_function(x_hat, batch, t)
            x_hat = x_hat.detach()

            ## compute proposal and reversal probs
            x_dist = torch.distributions.Normal(mu, scale)
            logp_reverse = x_dist.log_prob(x).sum(1)
            logp_forward = x_dist.log_prob(x_hat).sum(1)

            ## accept prob
            logp_accept = logp_x_hat - logp_x + logp_reverse - logp_forward
            u = torch.rand((x.shape[0],), device=logp_accept.device)
            accept = (u < torch.exp(logp_accept)).type(torch.float32)

            ## update samples x
            x = accept[:, None] * x_hat + (1 - accept[:, None]) * x

            accept_rate.append(accept.sum() / accept.shape[0])
        accept_rate = sum(accept_rate) / len(accept_rate)
        self._update_acceptance_rate(accept_rate.item(), t)
        x = x.detach()
        return x


class AnnealedMUHASampler(MetropolisSampler):
    """ AIS with HMC sampler, from paper https://arxiv.org/pdf/2302.11552.pdf
    """

    def __init__(self,
                 num_samples_per_step: int,
                 step_sizes: np.array,
                 damping_coeff: int,
                 mass_diag_sqrt: float,
                 num_leapfrog_steps: int,
                 gradient_function,
                 energy_function):
        super().__init__(num_samples_per_step, step_sizes)
        self._damping_coeff = damping_coeff
        if not HMC_USE_TORCH:
            mass_diag_sqrt = mass_diag_sqrt.detach().cpu().numpy()
        self._mass_diag_sqrt = mass_diag_sqrt
        self._num_steps = len(step_sizes)
        self._num_leapfrog_steps = num_leapfrog_steps
        self._gradient_function = gradient_function
        self._energy_function = energy_function
        self._noise_function = torch.randn_like

        self._total_steps = self._num_samples_per_step * (self._num_steps - 1)
        self._total_steps_reverse = self._num_samples_per_step * self._num_steps

    def leapfrog_step(self, x, v, batch, i):
        step_size = self._step_sizes[i]
        if HMC_USE_TORCH:
            i = torch.tensor([i], dtype=torch.long, device=x.device)
        else:
            i = np.asarray([i])
        mass_diag_sqrt = self._mass_diag_sqrt[i]
        return leapfrog_step(x, v, lambda _x: self._gradient_function(_x, batch, i), step_size,
                             mass_diag_sqrt, self._num_leapfrog_steps)

    def sample_step(self, x_k, batch, t):
        ## Sample Momentum, v_k: [b, 2]
        mass_diag_sqrt = self._mass_diag_sqrt[t]
        v_k = torch.randn_like(x_k) * mass_diag_sqrt
        v_dist = torch.distributions.Normal(torch.zeros_like(x_k), torch.ones_like(x_k) * mass_diag_sqrt)
        accept_rate = []  ## torch.zeros((self._num_steps,))

        for i in range(self._num_samples_per_step):
            ## Partial Momentum Refreshment
            eps = torch.randn_like(x_k)

            ## resample momentum
            v_k_prime = v_k * self._damping_coeff + np.sqrt(1. - self._damping_coeff ** 2) * eps * mass_diag_sqrt

            ## advance samples
            x_k_next, v_k_next = self.leapfrog_step(x_k, v_k_prime, batch, i)

            ## compute change in density, sum up the log-probs of each dimension
            logp_v_p = v_dist.log_prob(v_k_prime).sum(1)
            logp_v = v_dist.log_prob(v_k_next).sum(1)

            ## compute target log-probs
            logp_x = self._energy_function(x_k, batch, t)
            logp_x_hat = self._energy_function(x_k_next, batch, t)

            ## compute joint log-probs
            log_joint_prev = logp_x + logp_v_p
            log_joint_next = logp_x_hat + logp_v

            ## acceptance prob
            logp_accept = log_joint_next - log_joint_prev
            u = torch.rand((x_k_next.shape[0],), device=logp_accept.device)
            accept = (u < torch.exp(logp_accept)).type(torch.float32)

            ## update samples x_k
            x_k = accept[:, None] * x_k_next + (1 - accept[:, None]) * x_k
            v_k = accept[:, None] * v_k_next + (1 - accept[:, None]) * v_k_prime

            accept_rate.append(accept.sum() / accept.shape[0])
        accept_rate = sum(accept_rate) / len(accept_rate)
        self._update_acceptance_rate(accept_rate.item(), t)
        x_k = x_k.detach()
        return x_k

    # @torch.enable_grad()
    # def sample_step(self, x_k, batch, t):
    #     device = x_k.device
    #
    #     ## Sample Momentum, v_k: [b, 2]
    #     x_k = x_k.detach().cpu().numpy()
    #     v_k = np.random.randn(*x_k.shape) * self._mass_diag_sqrt[0]
    #     v_k = v_k.astype(np.float32)
    #     # v_dist = np.random.normal(np.zeros_like(x_k), np.ones_like(x_k) * self._mass_diag_sqrt)
    #     v_dist_loc = np.zeros_like(x_k)
    #     v_dist_scale = np.ones_like(x_k) * self._mass_diag_sqrt[0]
    #     accept_rate = np.zeros((self._num_samples_per_step,))
    #
    #     for i in range(self._num_samples_per_step):
    #         ## Partial Momentum Refreshment
    #         eps = np.random.randn(*x_k.shape).astype(np.float32)
    #
    #         ## resample momentum
    #         v_k_prime = v_k * self._damping_coeff + np.sqrt(1. - self._damping_coeff ** 2) * eps * self._mass_diag_sqrt[i]
    #
    #         ## advance samples
    #         x_k_next, v_k_next = self.leapfrog_step(x_k, v_k_prime, batch, i)
    #
    #         ## compute change in density
    #         logp_v_p = log_prob(v_k_prime, loc=v_dist_loc, scale=v_dist_scale)
    #         logp_v = log_prob(v_k_next, loc=v_dist_loc, scale=v_dist_scale)
    #
    #         ## compute target log-probs
    #         logp_x = self._energy_function(torch.tensor(x_k, device=device), batch, t).detach().cpu().numpy()
    #         logp_x_hat = self._energy_function(torch.tensor(x_k_next, device=device), batch, t).detach().cpu().numpy()
    #
    #         ## compute joint log-probs
    #         log_joint_prev = logp_x + logp_v_p
    #         log_joint_next = logp_x_hat + logp_v
    #
    #         ## acceptance prob
    #         logp_accept = log_joint_next - log_joint_prev
    #         u = np.random.random(x_k_next.shape)
    #         accept = (u < np.exp(logp_accept)).astype(np.float32)
    #
    #         ## update samples x_k
    #         # x_k = accept[:, None] * x_k_next + (1 - accept[:, None]) * x_k
    #         # v_k = accept[:, None] * v_k_next + (1 - accept[:, None]) * v_k_prime
    #         x_k = accept * x_k_next + (1 - accept) * x_k
    #         v_k = accept * v_k_next + (1 - accept) * v_k_prime
    #
    #         # accept_rate = accept_rate.at[i].set(accept_rate[i] + accept.mean())  ## [1000]
    #         accept_rate[i] = accept_rate[i] + accept.mean()
    #
    #     accept_rate = sum(accept_rate) / len(accept_rate)
    #     self._update_acceptance_rate(accept_rate, t)
    #
    #     x_k = torch.tensor(x_k, device=device)
    #     return x_k
