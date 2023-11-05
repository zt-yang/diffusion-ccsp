import jacinle; jacinle.hook_exception_ipdb()
import numpy as np
import torch
import pdb
import sys
import os
from os.path import join, isdir
import argparse

from datasets import GraphDataset
from data_transforms import pre_transform
from train_utils import create_trainer, get_args

if not isdir('data'):
    os.mkdir('data')
if not isdir('logs'):
    os.mkdir('logs')


def train_ddpm(args, **kwargs):
    trainer = create_trainer(args, **kwargs)
    if trainer is None:
        return
    trainer.train()


if __name__ == '__main__':
    """
    to add a new task
    1. run dataset.py to generate the pt files and try evaluation / visualization
    2. change dims in create_trainer() in train_utils.py
    3. change init() and initiate_denoise_fns() in ConstraintDiffuser class of denoise_fn.py
    3. change world.name in Trainer class of ddpm.py
    4. train with debug=True and visualize=True
    5. change wandb project name
    """

    """ for the CoRL submission
    python train_ddpm.py -input_mode qualitative -timesteps 1000 -EBM ULA -samples_per_step 10 -step_sizes '2 * self.betas'
    python train_ddpm.py -input_mode qualitative -timesteps 1000 -EBM MALA -samples_per_step 10 -step_sizes '0.0001 * torch.ones_like(self.betas)'
    python train_ddpm.py -input_mode qualitative -timesteps 1000 -EBM HMC -samples_per_step 4 -step_sizes '1 * self.betas ** 1.5'
    """

    train_ddpm(
        get_args(input_mode='qualitative', timesteps=1000, model='Diffusion-CCSP',
                 EBM='False', samples_per_step=3, use_wandb=True),
        debug=False, visualize=False, data_only=False
    )
