import torch
import sys
import argparse
from os.path import join, dirname, abspath, isdir, isfile
from os import listdir
import yaml
from pprint import pprint
# sys.path.append(join(dirname(abspath(__file__)), 'envs'))

from torch_geometric.loader import DataLoader
from datasets import GraphDataset, RENDER_PATH
from networks.ddpm import Trainer, GaussianDiffusion
from networks.denoise_fn import ConstraintDiffuser, ComposedEBMDenoiseFn
from data_transforms import pre_transform
from data_utils import print_tensor


def wandb_init(config, project_name='grid_offset_mp4'):
    import wandb
    keys = ['data_dir', 'data.batch_size', 'model.lr', 'trainer.max_epochs']
    key_names = {
        keys[i]: keys[i].replace('data.', '').replace('model.', '').replace('trainer.', '') for i in range(len(keys))
    }
    conf = {key_names[k]: v for k, v in config.items() if k in keys}
    print('\nArgs:', conf, '\n')
    wandb.init(project=project_name, entity="sketchers", config=config)


def send_email(address, title=None, message=None, textfile=None):
    """
    python -m smtpd -n -c DebuggingServer localhost:1025
    """
    import smtplib
    from email.mime.text import MIMEText

    # Open a plain text file for reading.  For this example, assume that
    # the text file contains only ASCII characters.
    if textfile is not None:
        with open(textfile, 'rb') as fp:
            # Create a text/plain message
            msg = MIMEText(fp.read())
        title = textfile
    else:
        msg = MIMEText(message)

    # me == the sender's email address
    # you == the recipient's email address
    msg['Subject'] = 'The contents of %s' % title
    msg['From'] = me = "ztyang@mit.edu"
    msg['To'] = you = address

    # Send the message via our own SMTP server, but don't include the
    # envelope header.
    s = smtplib.SMTP('localhost', port=1025)
    s.sendmail(me, [you], msg.as_string())
    s.quit()


def desktop_notify(title="Heavens, Hells, and Holy Sketchers", message="We've finished training!"):
    import asyncio
    from desktop_notifier import DesktopNotifier

    notifier = DesktopNotifier()

    async def main():
        n = await notifier.send(title, message)

        await asyncio.sleep(10)  # wait a bit before clearing notification

        await notifier.clear(n)  # removes the notification
        await notifier.clear_all()  # removes all notifications for this app

    asyncio.run(main())


def print_config(name: str, dic: dict):
    from pprint import pprint
    print(f'--------- {name} ---------')
    pprint(dic)
    print('-----------------------------------')


#################################################################


def get_args(train_task='None', test_tasks=None, timesteps=1000, model='Diffusion-CCSP', EBM=False,
             train_num_steps=300000, input_mode=None,
             hidden_dim=256, ebm_per_steps=1, ev='ff', use_wandb=False, pretrained=False, normalize=True,
             run_id=None, train_proj='correct_norm', samples_per_step=10, step_sizes='2*self.betas'):

    parser = argparse.ArgumentParser()
    parser.add_argument('-timesteps', type=int, default=timesteps)
    parser.add_argument('-model', type=str, default=model, choices=['Diffusion-CCSP', 'StructDiffusion'])
    parser.add_argument('-EBM', type=str, default=EBM, choices=['False', 'ULA', 'ULA+', 'HMC', 'MALA'])
    parser.add_argument('-energy_wrapper', type=bool, default=False)
    parser.add_argument('-samples_per_step', type=int, default=samples_per_step)
    parser.add_argument('-step_sizes', type=str, default=step_sizes)

    parser.add_argument('-train_task', type=str, default=train_task)
    parser.add_argument('-train_small', type=float, default=1)
    parser.add_argument('-train_name', type=str, default='')
    parser.add_argument('-train_proj', type=str, default=train_proj)
    parser.add_argument('-train_num_steps', type=int, default=train_num_steps)
    parser.add_argument('-input_mode', type=str, default=input_mode)
    parser.add_argument('-ebm_per_steps', type=int, default=ebm_per_steps)
    parser.add_argument('-ev', type=str, default=ev)  ## zf | fz | zz | ff
    parser.add_argument('-hidden_dim', type=int, default=hidden_dim)
    parser.add_argument('-normalize', type=bool, default=normalize)
    parser.add_argument('-pretrained', action='store_true', default=pretrained)
    parser.add_argument('-use_wandb', action='store_true', default=use_wandb)
    parser.add_argument('-run_id', type=str, default=run_id)
    args = parser.parse_args()
    if args.EBM == 'False':
        args.EBM = False
    if args.EBM in ['HMC', 'MALA']:
        args.energy_wrapper = True

    args.test_tasks = test_tasks
    if args.input_mode in ['diffuse_pairwise', 'diffuse_pairwise_image']:
        if args.input_mode == 'diffuse_pairwise_image':
            args.pretrained = True
        args.train_proj = 'triangular_3'

        # --- for testing
        # train_task = "RandomSplitWorld(10)_train_3_split"
        # test_tasks = {i: f"RandomSplitWorld(10)_test_{i}_split" for i in range(3, 4)}

        train_task = "RandomSplitWorld(20000)_train"  ## {3: 5001, 4: 5001, 5: 5001, 6: 4997}
        test_tasks = {i: f"RandomSplitWorld(100)_test_{i}_split" for i in range(3, 9)}

        # --- for testing
        # train_task = "TriangularRandomSplitWorld(7500)_train"  ## {2: 7500}
        train_task = "TriangularRandomSplitWorld(30000)_train"  ## {2: 7500, 3: 7500, 4: 7500, 5: 7500}
        test_tasks = {i: f"TriangularRandomSplitWorld(100)_test_{i}_split" for i in range(2, 8)}

        # train_task = "TriangularRandomSplitWorld[32]_(40)_train"
        # test_tasks = {2: f"TriangularRandomSplitWorld[32]_(10)_test_2_split"}

        args.train_task = "TriangularRandomSplitWorld[64]_(30000)_diffuse_pairwise_train"  ## {2: 7500, 3: 7500, 4: 7500, 5: 7500}
        args.test_tasks = {i: f"TriangularRandomSplitWorld[64]_(10)_diffuse_pairwise_test_{i}_split" for i in range(2, 7)}

    elif args.input_mode == 'qualitative':
        args.train_proj = 'correct_norm'
        args.train_proj = 'qualitative_new'
        args.train_proj = 'qualitative_correct'

        # --- for testing
        train_task = "RandomSplitQualitativeWorld(10000)_qualitative_train_2_object"
        test_tasks = {i: f'RandomSplitQualitativeWorld(10)_qualitative_test_{i}_split' for i in range(2, 4)}

        if 'World' not in args.train_task:
            # args.train_task = "RandomSplitQualitativeWorld(20)_qualitative_train"
            args.train_task = "RandomSplitQualitativeWorld(30000)_qualitative_train"  ## 60000
            args.train_task = "RandomSplitQualitativeWorld(60000)_qualitative_train"

        args.test_tasks = {i: f'RandomSplitQualitativeWorld(10)_qualitative_test_{i}_split' for i in range(2, 5)}

    elif args.input_mode == 'stability_flat':
        args.train_proj = 'stability'
        # train_task = "RandomSplitWorld(20)_train"
        args.train_task = "RandomSplitWorld(24000)_stability_flat_train"
        args.test_tasks = {i: f'RandomSplitWorld(10)_stability_flat_test_{i}_object' for i in range(4, 7)}

    elif args.input_mode == 'robot_box':
        args.train_proj = 'robot_box'
        # train_task = "TableToBoxWorld(10)_train"
        args.train_task = "TableToBoxWorld(10000)_train"  ## {3: 1000, 4: 1000, 5: 1000}
        args.test_tasks = {i: f"TableToBoxWorld(10)_test_{i}_object" for i in range(2, 7)}

    elif args.input_mode is None:
        args.test_tasks = {}

    ## for training with less training data
    if args.train_small < 1:
        if abs(args.train_small - 0.1) < 0.001:
            args.train_task = args.train_task.replace('0)_', f')_')
        print('\nargs.train_task', args.train_task)

    ## for training baseline methods
    # if args.model == 'StructDiffusion':
    #     args.train_proj = 'struct_' + args.train_proj
    return args


def create_trainer(args, debug=False, data_only=False, test_model=True,
                   visualize=False, composed_inference=False, verbose=True, **kwargs):
    timesteps = args.timesteps
    model = args.model
    EBM = args.EBM
    hidden_dim = args.hidden_dim
    train_num_steps = args.train_num_steps
    ebm_per_steps = args.ebm_per_steps
    samples_per_step = args.samples_per_step
    step_sizes = args.step_sizes

    train_name_extra = args.train_name
    run_id = args.run_id
    ebm_variations = args.ev
    train_proj = args.train_proj
    train_task = args.train_task
    test_tasks = args.test_tasks
    input_mode = args.input_mode
    pretrained = args.pretrained
    normalize = args.normalize
    use_wandb = args.use_wandb
    energy_wrapper = args.energy_wrapper
    if debug or data_only:
        use_wandb = False

    if model != 'Diffusion-CCSP':
        train_name = f'm={model}_t={timesteps}'
    else:
        train_name = f'm={EBM}_t={timesteps}'
    if train_name_extra != '' and train_name_extra != train_name:
        train_name += f'_{train_name_extra}'
    if train_proj == 'hmc':
        train_name = f'm={EBM}_ev={ebm_variations}'

    config = dict(
        train_batch_size=128,
        train_lr=5e-4,
        train_num_steps=train_num_steps,
        ebm_per_steps=ebm_per_steps,
        hidden_dim=hidden_dim,
        timesteps=timesteps,
        model=model,
        EBM=EBM,
        energy_wrapper=energy_wrapper,
        normalize=normalize,
        samples_per_step=samples_per_step,
        step_sizes=step_sizes,
        train_task=train_task,
        train_name=train_name,
        input_mode=input_mode,
    )
    if use_wandb:
        import wandb
        wandb_kwargs = dict(project=train_proj, entity="sketchers", config=config)
        wandb.init(name=train_name, **wandb_kwargs)
        run_id = wandb.run.id

    render_dir = join(RENDER_PATH, f"{train_task}_{train_name}_{input_mode}")
    if input_mode not in train_task and not composed_inference:
        train_task = train_task.replace('_train', f'_{input_mode}_train')

    ## correct dataset names that are not consistent with the input mode
    if test_tasks is not None:
        new_test_tasks = {}
        for k, v in test_tasks.items():
            if input_mode not in v and not composed_inference:
                v = v.replace('_test', f'_{input_mode}_test')
            new_test_tasks[k] = v
        test_tasks = new_test_tasks

    if run_id is not None:
        log_name = run_id
        render_dir = f"{render_dir}_id={log_name}"
    else:
        log_name = train_task

    dataset_kwargs = dict(input_mode=input_mode, pre_transform=pre_transform, visualize=False)
    train_dataset = GraphDataset(train_task, **dataset_kwargs)
    test_datasets = {k: GraphDataset(task, **dataset_kwargs) for k, task in test_tasks.items()}

    if data_only:
        return None

    #### number of features for each type of variable
    if 'robot' in input_mode:
        """ e.g., 8 numbers given in each group, but only the first 6 is used by the network, 
            the last 2 for reconstruction """
        dims = ((8, 0, 8), (5, 10, 15), (5, 16, 21))
    elif 'stability' in input_mode or 'qualitative' in input_mode:
        dims = (2, 0, 2), (4, 2, 6)
    else:
        dims = ((3, 0, 3), (4, 3, 7)) if 'Triangular' in train_task else ((2, 0, 2), (2, 2, 4))  ## P1
        # dims = ((6, 0, 6), (4, 6, 10)) if 'Triangular' in train_task else ((2, 0, 2), (2, 2, 4))  ## P2
        if input_mode == 'diffuse_pairwise_image':
            image_dim = int(eval(train_task[train_task.index('[')+1:train_task.index(']')])) ** 2
            begin = dims[0][2] + image_dim
            dims = tuple([dims[0], (image_dim, dims[0][2], begin), (dims[1][0], begin, begin + dims[1][0])])

    denoise_fn = ConstraintDiffuser(dims=dims, hidden_dim=hidden_dim, EBM=EBM, input_mode=input_mode,
                                    pretrained=pretrained, normalize=normalize, energy_wrapper=energy_wrapper,
                                    model=model, verbose=verbose).cuda()
    if EBM and denoise_fn.energy_wrapper:
        denoise_fn = ComposedEBMDenoiseFn(denoise_fn, ebm_per_steps, ebm_variations)
    diffusion = GaussianDiffusion(denoise_fn, timesteps=timesteps, EBM=EBM,
                                  samples_per_step=samples_per_step, step_sizes=step_sizes).cuda()

    ### test model
    if test_model:
        # torch.manual_seed(0)
        train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=False, pin_memory=True, num_workers=0)
        print('testing forward - sequential')
        for data in train_dataloader:
            loss = diffusion(data, debug=True, tag='EBM')  ##  if EBM else 'test run'
            loss.backward()
            break
        print('done')

    return Trainer(
        diffusion,
        train_dataset,
        test_datasets,
        render_dir,
        gradient_accumulate_every=1,    # gradient accumulation steps
        ema_decay=0.995,                # exponential moving average decay
        fp16=False,                     # turn on mixed precision training with apex
        save_and_sample_every=10000 if not debug else 100,
        results_folder=f'./logs/{log_name}',
        visualize=visualize,
        use_wandb=use_wandb,
        **config,
        **kwargs,
    )


def get_args_from_run_id(run_id):
    args = get_args(use_wandb=False, run_id=run_id)
    wandb_dir = [join('wandb', f) for f in listdir('wandb') if run_id in f]
    if len(wandb_dir) == 0:
        wandb_dir = [join('wandb2', f) for f in listdir('wandb2') if run_id in f]
    wandb_dir = wandb_dir[0]
    yaml_path = join(wandb_dir, 'files', 'config.yaml')
    config = yaml.load(open(yaml_path, 'r'), Loader=yaml.FullLoader)
    for k, v in config.items():
        if k in ['train_batch_size', 'train_lr']:
            continue
        if type(v) is dict:
            setattr(args, k, v['value'])
    if run_id == 'j8lenp74':
        args.pretrained = True
    if run_id in ['bo02mwbw', '4xt8u4n7', 'qi3dqq2l']:
        args.normalize = False
    if run_id in ['9xhbwmi9', 'ta4tsbz6']:
        args.energy_wrapper = True
    if run_id in ['ql30000e', 'jn49b39m', 'g38uz4uk', 'uyq4fd3u', 'oamtpoae', '6jrpn5vf']:
        args.model = 'StructDiffusion'
    return args


def load_trainer(run_id, milestone, visualize=False, rejection_sampling=False, verbose=True, **kwargs):
    args = get_args_from_run_id(run_id)

    # args.test_tasks = kwargs.get('test_tasks', args.test_tasks)
    for k in ['input_mode', 'train_task', 'test_tasks', 'train_num_steps']:
        if k in kwargs:
            setattr(args, k, kwargs[k])
            kwargs.pop(k)
    if verbose:
        pprint(args.__dict__)

    trainer = create_trainer(args, visualize=visualize, rejection_sampling=rejection_sampling,
                             eval_only=True, verbose=verbose, **kwargs)
    trainer.load(milestone)
    return trainer


if __name__ == "__main__":
    desktop_notify()
