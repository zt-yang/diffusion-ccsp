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

    """ for example
    python train_ddpm.py -timesteps 1000 -EBM 'ULA' -input_mode qualitative
    """

    train_ddpm(
        get_args(input_mode='qualitative', timesteps=1000, model='Diffusion-CCSP',
                 EBM='False', samples_per_step=3, use_wandb=False),
        debug=False, visualize=False, data_only=False
    )
