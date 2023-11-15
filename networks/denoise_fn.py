import ipdb
from os.path import join, abspath
import torch
import torch.nn as nn
import torch.nn.functional as F
import jactorch
import numpy as np
import math
import matplotlib.pyplot as plt
import jactorch.nn as jacnn

from collections import defaultdict
from inspect import isfunction


puzzle_constraints = ['in', 'cfree']
robot_constraints = ['gin', 'gfree']
stability_constraints = ['within', 'supportedby', 'cfree']
qualitative_constraints = [
    'in', 'center-in', 'left-in', 'right-in', 'top-in', 'bottom-in',
    'cfree', 'left-of', 'top-of',
    'close-to', 'away-from', 'h-aligned', 'v-aligned'
]
robot_qualitative_constraints = robot_constraints + qualitative_constraints
ignored_constraints = ['right-of', 'bottom-of']


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


#################################################################################


# Define how to multiply two different EBM distributions together
class ComposedEBMDenoiseFn(nn.Module):
    """ wrapper around ConstraintDiffuser as a composition of diffusion models """

    def __init__(self, model, ebm_per_steps=1):
        super().__init__()
        self.model = model
        self.device = model.device
        self.dims = model.dims
        self.input_mode = model.input_mode
        self.ebm_per_steps = ebm_per_steps

        self.energy_wrapper = True

    def neg_logp_unnorm(self, poses_in, batch, t, **kwargs):
        # poses_in.requires_grad_(True)
        kwargs['tag'] = 'EBM'
        gradients, energy = self.model.forward(poses_in, batch, t, **kwargs)
        return energy.sum()

    def forward(self, poses_in, batch, t, **kwargs):
        if isinstance(poses_in, np.ndarray):
            poses_in = torch.tensor(poses_in, device=self.model.device)
            t = torch.tensor(t, device=self.model.device)
        # poses_in.requires_grad_(True)
        kwargs['tag'] = 'EBM'
        gradients, energy = self.model.forward(poses_in, batch, t, **kwargs)
        return gradients


#################################################################################


class GeomEncoderImage(torch.nn.Module):

    log_dir = abspath(join(__file__, '..', 'encoder_checkpoints', 'GeomEncoderImage'))

    def __init__(self, in_features=64, hidden_dim=256, num_channel=32):
        super(GeomEncoderImage, self).__init__()
        conv2d = nn.Conv2d  ## jacnn.CoordConv2D ##
        self.in_features = in_features
        self.num_channel = num_channel

        self.conv1 = conv2d(in_channels=1, out_channels=num_channel, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = conv2d(in_channels=num_channel, out_channels=num_channel, kernel_size=3, stride=1, padding=1)
        self.conv3 = conv2d(in_channels=num_channel, out_channels=num_channel, kernel_size=3, stride=1, padding=1)
        self.feature_dim = in_features // (2 ** 3)
        self.fc = nn.Linear(in_features=self.feature_dim ** 2 * num_channel, out_features=hidden_dim)

    def forward(self, x):
        ## reshape x from [b, 2, 4096] to [b x 2, 1, 64, 64]
        if len(x.shape) == 3:
            b, p = x.shape[0], x.shape[1]
            x = x.reshape([b * p, 1, self.in_features, self.in_features])
        ## reshape x from [b, 4096] to [b x 2, 1, 64, 64]
        else:
            b = x.shape[0]
            p = 1
            x = x.reshape([b, 1, self.in_features, self.in_features])

        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))
        x = F.relu(self.pool(self.conv3(x)))
        x = F.dropout(x, training=self.training)
        if len(x.shape) == 3:
            x = x.reshape([b, p, self.feature_dim ** 2 * self.num_channel])
        else:
            x = x.reshape([b, self.feature_dim ** 2 * self.num_channel])
        return self.fc(x)

    def load_pretrained_weights(self):
        model_dict = torch.load(join(self.log_dir, 'best_model.pt'))
        self.load_state_dict(model_dict)
        for param in self.parameters():
            param.requires_grad = False


class GeomDecoderImage(torch.nn.Module):
    def __init__(self, out_features=64, hidden_dim=256, num_channel=32):
        super(GeomDecoderImage, self).__init__()
        self.out_features = out_features
        self.num_channel = num_channel
        self.feature_dim = out_features // (2 ** 3)
        self.fc = nn.Linear(in_features=hidden_dim, out_features=self.feature_dim ** 2 * num_channel)
        self.t_conv1 = nn.ConvTranspose2d(num_channel, num_channel, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(num_channel, num_channel, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(num_channel, 1, 2, stride=2)

    def forward(self, x):
        x = self.fc(x)
        x = x.reshape([x.shape[0], self.num_channel, self.feature_dim, self.feature_dim])
        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))
        x = torch.sigmoid(self.t_conv3(x))
        x = x.reshape([x.shape[0], self.out_features * self.out_features])
        return x


class GeomAutoEncoder(torch.nn.Module):
    def __init__(self, in_features=64, hidden_dim=256, num_channel=32):
        super(GeomAutoEncoder, self).__init__()
        self.in_features = in_features
        self.encoder = GeomEncoderImage(in_features, hidden_dim, num_channel)
        self.decoder = GeomDecoderImage(in_features, hidden_dim, num_channel)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def visualize_image(self, before, after, png_name):
        num_images = before.shape[0]
        plt.figure(figsize=(10, 5*num_images))
        for i, (x, title) in enumerate(zip([before, after], ['Before', 'After'])):
            x = x.reshape([num_images, self.in_features, self.in_features]).detach().cpu().numpy()
            for j in range(num_images):
                y = x[j] * 255
                plt.subplot(num_images, 2, j*2+i+1)
                plt.imshow(y, interpolation='nearest')
                plt.axis('off')
                plt.title(title)
        plt.savefig(png_name, bbox_inches='tight')
        plt.close()


def print_network(net, name):
    print(name, '\t', net)


class ConstraintDiffuser(torch.nn.Module):
    def __init__(self, dims=((2, 0, 2), (2, 2, 4)), hidden_dim=256, max_num_obj=12, input_mode=None,
                 EBM=False, pretrained=False, normalize=True, energy_wrapper=False, device='cuda',
                 model='Diffusion-CCSP', verbose=True):
        """ in_features: list of input feature dimensions for each variable type (geometry, pose)
            e.g. for puzzle constraints ((6, 0, 6), (4, 6, 10)) = {(length, begin, end)}
                means 6 geometry features for pose, 4 for pose features
            e.g. for robot constraints ((6, 0, 6), (5, 8, 13), (5, 14, 18)) = {(length, begin, end)}
                means 6 geometry features for pose, taking position 0-5
        """
        super(ConstraintDiffuser, self).__init__()
        self.hidden_dim = hidden_dim
        self.max_num_obj = max_num_obj
        self.EBM = EBM
        self.device = device
        self.dims = dims
        self.input_mode = input_mode
        self.use_image = False
        self.normalize = normalize
        self.verbose = verbose
        self.energy_wrapper = energy_wrapper  ## (EBM not in [False])  ## , 'ULA'
        self.model = model

        if 'robot' in input_mode:
            self.constraint_sets = robot_constraints
        elif 'stability' in input_mode:
            self.constraint_sets = stability_constraints
        elif 'qualitative' in input_mode:
            self.constraint_sets = qualitative_constraints
        else:
            self.constraint_sets = puzzle_constraints

        ## use images
        if self.constraint_sets == puzzle_constraints and len(dims) == 3:
            self.geom_encoder = GeomEncoderImage(int(math.sqrt(dims[1][0])), hidden_dim).to(device)
            print('ConstraintDiffuser pretrained', pretrained)
            if pretrained:
                self.geom_encoder.load_pretrained_weights()
            self.use_image = True
            if self.verbose: print_network(self.geom_encoder, 'geom_encoder')

        ## for encoding object geometry, e.g. 2D dimensions
        else:
            self.geom_encoder = nn.Sequential(
                nn.Linear(dims[0][0], hidden_dim//2),
                nn.SiLU(),
                nn.Linear(hidden_dim//2, hidden_dim),
                nn.SiLU(),
            ).to(self.device)
            if self.verbose: print_network(self.geom_encoder[0], 'geom_encoder')

        if 'robot' in self.input_mode:
            self.grasp_encoder = nn.Sequential(
                nn.Linear(dims[1][0], hidden_dim//2),
                nn.SiLU(),
                nn.Linear(hidden_dim//2, hidden_dim),
                nn.SiLU(),
            ).to(self.device)
            if self.verbose: print_network(self.grasp_encoder[0], 'grasp_encoder')

        ## for encoding object pose, e.g. (x, y)
        self.pose_encoder = nn.Sequential(
            nn.Linear(dims[-1][0], hidden_dim//2),
            nn.SiLU(),
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.SiLU(),
        ).to(self.device)
        if self.verbose: print_network(self.pose_encoder[0], 'pose_encoder')

        self.pose_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.SiLU(),
            nn.Linear(hidden_dim//2, dims[-1][0]),
        ).to(self.device)

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.Mish(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        ## for each type of constraints
        if self.model == 'Diffusion-CCSP':
            self.mlps = self.initiate_denoise_fns()

        elif self.model == 'StructDiffusion':
            from transformer import Transformer, PositionalEncoding
            self.max_seq_len = 8
            self.num_heads = 2
            self.num_layers = 4

            width = self.hidden_dim * (2 if 'robot' not in self.input_mode else 3)
            self.pe = PositionalEncoding(width, 0).pe.to(self.device, non_blocking=True)
            self.ln_pre = nn.LayerNorm(width)
            self.transformer = Transformer(width, self.num_layers, self.num_heads)
            self.ln_post = nn.LayerNorm(width)

            self.shuffled = {}  ## because of dataset problem

        self.ebm_per_steps = 1

        ## for composing with a different domain
        self.pose_encoder_2 = None
        self.geom_encoder_2 = None
        self.pose_decoder_2 = None
        self.time_mlp_2 = None
        self.composing_weight = (1, 1)

    def initiate_denoise_fns(self):
        out_feature = 2 * self.hidden_dim  ## change two poses for now
        mlps = []
        if self.verbose: print(f'denoise_fns({len(self.constraint_sets)})', self.constraint_sets)
        for con in self.constraint_sets:
            if con in robot_constraints:
                """ g, o, o, p, p, t """
                linear = nn.Linear(self.hidden_dim * 6, out_feature)
            else:
                """ o, o, p, p, t """
                linear = nn.Linear(self.hidden_dim * 5, out_feature) ## * 2 because of o and p
            mlp = nn.Sequential(linear, nn.SiLU()).to(self.device)
            mlps.append(mlp)
            if self.verbose: print_network(mlp[0], '\t'+con)
        if self.verbose: print('-' * 50)
        return nn.ModuleList(mlps)

    def _if_use_default_encoder(self, i):
        return not (self.input_mode == 'robot_qualitative' and i >= 2)

    def _get_constraint_inputs(self, i, batch, t, emb_dict, edge_index):
        use_default_encoder = self._if_use_default_encoder(i)

        ## find the nodes used by all constraints of this type
        edges = torch.where(batch.edge_attr == i)[0]
        edges = edges.detach().cpu().numpy()
        args_1 = edge_index[edges][:, 0]
        args_2 = edge_index[edges][:, 1]

        args = torch.stack([args_1, args_2], dim=1)
        input_dict = {'args': args}
        if use_default_encoder:
            input_dict.update({
                'geoms_emb': emb_dict['geoms_emb'][args],
                'poses_emb': emb_dict['poses_emb'][args],
                'time_embedding': self.time_mlp(jactorch.add_dim(t, 0, edges.shape[0]))[:, 0],
            })
        else:
            input_dict.update({
                'geoms_emb_2': emb_dict['geoms_emb_2'][args],
                'poses_emb_2': emb_dict['poses_emb_2'][args],
                'time_embedding': self.time_mlp_2(jactorch.add_dim(t, 0, edges.shape[0]))[:, 0],
            })
        if 'robot' in self.input_mode:
            input_dict['grasp_emb'] = emb_dict['grasp_emb'][args_1]

        return input_dict

    def _process_constraint(self, i, input_dict): ## when b = 4
        use_default_encoder = self._if_use_default_encoder(i)
        geom_emb = input_dict['geoms_emb' if use_default_encoder else 'geoms_emb_2']  ## torch.Size([4, 2, 256])
        pose_emb = input_dict['poses_emb' if use_default_encoder else 'poses_emb_2']  ## torch.Size([4, 2, 256])

        embeddings = [
            geom_emb.reshape(geom_emb.shape[0], -1),  ## torch.Size([4, 512])
            pose_emb.reshape(pose_emb.shape[0], -1),  ## torch.Size([4, 512])
            input_dict['time_embedding']  ## torch.Size([4, 256])
        ]
        if 'robot' in self.input_mode and use_default_encoder:
            grasp_emb = input_dict['grasp_emb']
            embeddings = [grasp_emb] + embeddings
        inputs = torch.cat(embeddings, dim=-1)  ## [b, 5 * hidden_dim]

        outputs = self.mlps[i](inputs)
        outs_1 = outputs[:, :self.hidden_dim]
        outs_2 = outputs[:, self.hidden_dim:]
        ## decode the output pose features to objects
        outputs = torch.stack([outs_1, outs_2], dim=1)  # [B, 2, dim]
        if use_default_encoder:
            outputs = self.pose_decoder(outputs)
            if self.composing_weight[0] != 1:
                outputs *= self.composing_weight[0]
        else:
            outputs = self.pose_decoder_2(outputs)
            zeros = torch.zeros_like(outputs[:, :, 0:1]).to(self.device)
            outputs = torch.cat([outputs[:, :, :2], zeros, outputs[:, :, 2:]], dim=-1)
            if self.composing_weight[1] != 1:
                outputs *= self.composing_weight[1]
        return outputs

    def _compute_energy(self, i, input_dict, poses_in, outputs):  # noqa
        input_poses_in = poses_in[input_dict['args']]
        return ((outputs - input_poses_in) ** 2).sum()

    def _add_constraints_outputs(self, i, input_dict, outputs, all_poses_out, all_counts_out=None):
        n_features = all_poses_out.shape[0]

        args = input_dict['args'].reshape(-1)
        outputs = outputs.reshape(-1, outputs.shape[-1])

        all_poses_out.scatter_add_(0, args.unsqueeze(-1).expand(outputs.shape), outputs)

        ## take the average of the output pose features of each object
        if all_counts_out is not None:
            all_counts_out += torch.bincount(args, minlength=n_features).to(self.device)

        return all_poses_out, all_counts_out

    def _forward_struct_diffusion(self, emb_dict, batch, t):
        """ a sequence of object shape + pose pairs, including the container """
        from einops import repeat, rearrange

        ## add time embedding to each pose embedding
        geoms_emb = emb_dict['geoms_emb']  ## [8, 256]
        time_emb = self.time_mlp(jactorch.add_dim(t, 0, geoms_emb.shape[0]))[:, 0]  ## [8, 256]
        poses_emb = emb_dict['poses_emb'] + time_emb

        ## make input sequence
        sequences = []
        attn_masks = []
        indices = []
        obj_emb = torch.cat([geoms_emb, poses_emb], dim=-1)  ## [8, 512]
        if 'robot' in self.input_mode:
            obj_emb = torch.cat([emb_dict['grasp_emb'], obj_emb], dim=-1)  ## [8, 256*3]

        for j in range(batch.batch.max().item() + 1):
            seq = obj_emb[batch.batch == j]  ## [4, 512]

            ## add positional embedding to indicate the sequence order
            ## the dataset has bias of object sequence order
            pe = self.pe[:, :seq.shape[0], :]
            if hasattr(batch, 'shuffled'):
                idx = batch.shuffled[batch.batch == j]  ## torch.randperm(seq.shape[0])
                pe = pe[:, idx, :]
            seq += rearrange(pe, 'b n c -> (b n) c')

            x = self.ln_pre(seq)
            padding_len = self.max_seq_len - x.shape[0]
            indices.append(x.shape[0])

            x = F.pad(x, (0, 0, 0, padding_len), "constant", 0)
            sequences.append(x)

            attn_mask = torch.zeros(self.max_seq_len, self.max_seq_len, device=self.device)  ## [8, 8]
            attn_mask[:, -padding_len:] = True
            attn_mask[-padding_len:, :] = True
            attn_masks.append(attn_mask)

        ## get output
        sequences = torch.stack(sequences, dim=1)  ## [8, 2, 512]
        attn_masks = torch.stack(attn_masks)  ## [2, 8, 8] when batch size is 2
        attn_masks = repeat(attn_masks, 'b l1 l2 -> (repeat b) l1 l2',
                            repeat=self.num_heads)  ## [4, 8, 8] for 2 heads
        x, weights, attn_masks = self.transformer((sequences, None, attn_masks))  ## x : [128, 4, 256]
        x = self.ln_post(x)  ## [8, 2, 512]
        x = x[:, :, -poses_emb.shape[-1]:]  ## [8, 2, 256]

        ## return poses out
        poses_out = []
        for j in range(batch.batch.max().item() + 1):
            poses_out.append(x[:indices[j], j])  ## [n, 256]
        poses_out = torch.cat(poses_out, dim=0)  ## [8, 256]
        poses_out = self.pose_decoder(poses_out)  ## [8, 4]

        ## mask out the containers
        mask = batch.mask.bool().to(self.device)
        poses_out[mask] = batch.x.to(self.device)[:, -self.dims[-1][0]:][mask]

        return poses_out

    def forward(self, poses_in, batch, t, verbose=False, debug=False, tag='EBM', eval=False):
        """ denoising a batch of ConstraintGraphData
        Args:
            poses_in:       torch.Tensor, which are noisy pose features to be denoised
            batch:         DataBatch
                x:              torch.Tensor, which are original geom and pose features
                edge_index:     torch.Tensor, pairs of indices of features, each is a constraint
                edge_attr:      torch.Tensor, indices of constraints
            t:              torch.Size([1]), e.g. tensor([938], device='cuda:0')
        Returns:
            updated_features:    torch.Tensor, which are denoised pose features
        """

        x = batch.x.clone().to(self.device)

        ## read geoms_emb from x
        if eval and self.use_image and x.shape[1] != sum([d[0] for d in self.dims]):
            geoms_emb = x[:, self.dims[0][2]:-self.dims[-1][0]]

        ## compute geoms_emb
        else:
            geoms_in = x[:, self.dims[1][1]:self.dims[1][2]] if self.use_image else x[:, :self.dims[0][2]]
            geoms_emb = self.geom_encoder(geoms_in)
            ## save the processed image features to save sampling time
            if eval and self.use_image:
                batch.x = torch.cat([x[:, :self.dims[0][2]], geoms_emb, x[:, self.dims[-1][1]:]], dim=-1)

        poses_in = poses_in.to(self.device)
        poses_in.requires_grad_(True)

        emb_dict = {'geoms_emb': geoms_emb, 'poses_emb': self.pose_encoder(poses_in)}
        if 'robot' in self.input_mode:
            emb_dict['grasp_emb'] = self.grasp_encoder(x[:, self.dims[1][1]:self.dims[1][2]])

        ## for composing different domains, robot_qualitative
        if self.pose_encoder_2 is not None:
            """ 
            robot_box
                geom_encoder (0-7): [w/w0, l/l0, h/h0, w0, l0, h0, x0, y0] + ()
                grasp_encoder (8-12): grasp_side
                pose_encoder (13-17): [x, y, z, sn, cs]
            qualitative
                geom_encoder (0-1): [w, l]
                pose_encoder (2-5): [x, y, cs, sn]
            """
            geoms_in_2 = geoms_in[:, :2]
            poses_in_2 = torch.cat([poses_in[:, :2], x[:, -2:]], dim=-1)
            # print(poses_in_2)  ## .detach().cpu().numpy().tolist()
            geoms_emb_2 = self.geom_encoder_2(geoms_in_2)
            poses_emb_2 = self.pose_encoder_2(poses_in_2)
            emb_dict.update({'geoms_emb_2': geoms_emb_2, 'poses_emb_2': poses_emb_2})

        if self.model == 'StructDiffusion':
            return self._forward_struct_diffusion(emb_dict, batch, t)

        edge_index = batch.edge_index.T.to(self.device)
        all_poses_out = torch.zeros_like(poses_in)
        all_counts_out = torch.zeros_like(poses_in[:, 0])
        total_energy = 0
        for i in range(len(self.mlps)):
            input_dict = self._get_constraint_inputs(i, batch, t, emb_dict, edge_index)
            if len(input_dict['args']) == 0:
                continue
            outputs = self._process_constraint(i, input_dict)

            if tag == 'EBM' and self.energy_wrapper:
                total_energy += self._compute_energy(i, input_dict, poses_in, outputs)
            else:
                self._add_constraints_outputs(i, input_dict, outputs, all_poses_out, all_counts_out) ##

        if self.normalize:
            all_poses_out /= torch.sqrt(all_counts_out.unsqueeze(-1))

        ## return the gradients and energy
        if tag == 'EBM' and self.energy_wrapper:
            gradients = self._get_EBM_gradients(poses_in, total_energy)
            return gradients, total_energy
        else:
            """ lastly, assign the original pose features to the static objects """
            mask = batch.mask.bool().to(self.device)
            all_poses_out[mask] = x[:, -self.dims[-1][0]:][mask]

            if debug:
                self.print_debug_info(batch, x, poses_in, x[:, :self.dims[0][2]], all_poses_out, tag=tag)
            return all_poses_out

    def _get_EBM_gradients(self, poses_in, total_energy, **kwargs):
        assert torch.is_tensor(total_energy)
        if self.training:
            gradients = torch.autograd.grad(total_energy, poses_in, create_graph=True, retain_graph=True, **kwargs)[0]
        else:
            gradients = torch.autograd.grad(total_energy, poses_in, **kwargs)[0]
        # print('[_get_EBM_gradients] \t poses_in.shape =', poses_in.shape, ## torch.Size([2, 4])
        #       '\t total_energy =', total_energy.shape, ## torch.Size([])
        #       '\t gradients.shape =', gradients.shape) ## torch.Size([2, 4])
        return gradients

    def print_debug_info(self, batch, x, poses_in, geoms_in, poses_out, tag='train'):
        graph_indices = batch.x_extract.unique().numpy().tolist()
        for j in graph_indices:
            if j != 0:
                continue
            indices = torch.where(batch.x_extract == j)[0].numpy().tolist()
            print('-' * 50 + f"\n[{tag}]\t graph {int(j)} ({len(indices) - 1} objects)")
            for i in indices:
                print('\tshape =', nice(geoms_in[i]),
                      '\t actual =', nice(x[:, self.dims[-1][1]:self.dims[-1][2]][i]),
                      '\t | noisy =', nice(poses_in[i]),
                      '\t -> predicted =', nice(poses_out[i]))


def nice(x):
    return [round(n, 2) for n in x.cpu().detach().numpy()]
