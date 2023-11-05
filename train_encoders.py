import jacinle; jacinle.hook_exception_ipdb()
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import pdb
import sys
import os
from os.path import join, isdir
import argparse

from datasets import GraphDataset, RENDER_PATH
from networks.ddpm import Trainer, GaussianDiffusion
from networks.denoise_fn import GeomAutoEncoder
from data_transforms import pre_transform
from data_utils import print_tensor

if not isdir('../data'):
    os.mkdir('../data')
if not isdir('../logs'):
    os.mkdir('../logs')


def train_triangles_CNN_encoder(train_task, bs=128, n_epochs=100):
    """ train a CNN encoder for triangles, then freeze its weights when training the DDPM model """

    test_tasks = {i: f"TriangularRandomSplitWorld[64]_(100)_diffuse_pairwise_image_test_{i}_split" for i in range(2, 8)}
    k = 10 ## 7

    model = GeomAutoEncoder(in_features=64, hidden_dim=256, num_channel=32).cuda()
    log_dir = model.encoder.log_dir
    if not isdir(log_dir):
        os.makedirs(log_dir)
    img_dir = join(log_dir, 'images')
    if not isdir(img_dir):
        os.makedirs(img_dir)

    ds_kwargs = dict(input_mode='diffuse_pairwise_image', pre_transform=pre_transform)
    dl_kwargs = dict(batch_size=bs, pin_memory=True, num_workers=0)
    train_dataset = GraphDataset(train_task, **ds_kwargs)
    train_loader = DataLoader(train_dataset, shuffle=True, **dl_kwargs)
    num_data = len(train_loader)

    test_datasets = {i: GraphDataset(test_task, **ds_kwargs) for i, test_task in test_tasks.items()}
    test_dataloaders = {i: DataLoader(td, shuffle=False, **dl_kwargs) for i, td in test_datasets.items()}

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    last_test_loss = np.inf
    for epoch in range(1, n_epochs + 1):

        ## --------------------- train ----------------------
        model.train()
        train_loss = 0.0
        for data in train_loader:
            images = data.x[:, k:].cuda()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss = train_loss / num_data
        print(f'\nEpoch: {epoch} \tTraining Loss: {train_loss:.5f}')

        ## --------------------- test ----------------------
        model.eval()
        total_test_loss = 0.0
        for j, test_loader in test_dataloaders.items():
            test_loss = 0.0
            for data in test_loader:
                images = data.x[:, k:].cuda()
                outputs = model(images)
                loss = criterion(outputs, images)
                test_loss += loss.item() * images.size(0)
                model.visualize_image(images[1:j], outputs[1:j], join(img_dir, f"epoch={epoch}_n={j}.png"))
            test_loss = train_loss / len(test_loader)
            total_test_loss += test_loss
            print(f'\t \tTesting Loss on {j} objects: {test_loss:.5f}')

        ## --------------------- save ----------------------
        if total_test_loss < last_test_loss:
            state_dict = {k.replace('encoder.', ''): v for k, v in model.state_dict().items() if 'encoder' in k}
            torch.save(state_dict, join(log_dir, 'best_model.pt'))
            last_test_loss = total_test_loss


if __name__ == '__main__':
    train_task = "TriangularRandomSplitWorld[64]_(30000)_diffuse_pairwise_image_train"  ## {3: 7500, 4: 7500, 5: 7500, 6: 7500}
    train_triangles_CNN_encoder(train_task)
