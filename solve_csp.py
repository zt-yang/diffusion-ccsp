import os
from os.path import join, dirname, isdir
from datasets import GraphDataset
from data_transforms import pre_transform
from train_utils import load_trainer
import argparse


def check_data_graph(dataset_name):
    """ check data balance """
    dataset_kwargs = dict(input_mode='diffuse_pairwise', pre_transform=pre_transform)
    test_dataset = GraphDataset(dataset_name, **dataset_kwargs)
    for data in test_dataset:
        data = data.to('cuda')
        print(data)
        break


def evaluate_model(run_id, milestone, tries=(10, 0), json_name='eval', save_log=True,
                   run_all=False, render=True, run_only=False, resume_eval=False, render_name_extra=None,
                   return_history=False, **kwargs):
    trainer = load_trainer(run_id, milestone, **kwargs)
    if render_name_extra is not None:
        trainer.render_dir += f'_{render_name_extra}'
        if not isdir(trainer.render_dir):
            os.mkdir(trainer.render_dir)
    trainer.evaluate(json_name, tries=tries, render=render, save_log=save_log,
                     run_all=run_all, run_only=run_only, resume_eval=resume_eval, return_history=return_history)


MILESTONES = {
    'gtqtyd8n': 20, 'dyz4eido': 23,
    'qsd3ju74': 7, 'r26wkb13': 16,
    'ti639t00': 3, 'bo02mwbw': 11,
    '0hhtsh6a': 5, 'qcjqkout': 14,
}


def get_tests(input_mode, test_tiny=False, find_failure_cases=False):
    n = 10 if test_tiny else 100
    if input_mode == 'diffuse_pairwise':
        tests = {i: f"TriangularRandomSplitWorld[64]_({n})_diffuse_pairwise_test_{i}_split" for i in range(2, 7)}
        if find_failure_cases:
            tests = {i: f"TriangularRandomSplitWorld[64]_({n})_diffuse_pairwise_test_{i}_split" for i in range(5, 7)}
    elif input_mode == 'qualitative':
        tests = {i: f'RandomSplitQualitativeWorld({n})_qualitative_test_{i}_split' for i in range(2, 6)}
    elif input_mode == 'stability_flat':
        tests = {i: f"RandomSplitWorld({n})_stability_flat_test_{i}_object" for i in range(4, 8)}
        # tests = {i: f"RandomSplitWorld({n})_stability_flat_test_{i}_object_i=31" for i in range(7, 8)}
    elif input_mode == 'robot_box':
        tests = {i: f"TableToBoxWorld({n})_robot_box_test_{i}_object" for i in range(3, 7)}
        if find_failure_cases:
            tests = {i: f"TableToBoxWorld(10)_robot_box_test_{i}_object_i=0" for i in range(6, 7)}
    else:
        raise NotImplementedError
    if find_failure_cases:
        tests = {k: v for i, (k, v) in enumerate(tests.items()) if i >= len(tests) - 2}
    return tests


def indie_runs():
    """ for developing and debugging """
    # check_data_graph(test_tasks[4])
    # check_data_graph('TriangularRandomSplitWorld(20)_test')

    ################ results for CoRL ###########################
    eval_10_kwargs = dict(tries=(10, 0), json_name='eval_N=10_K=10', save_log=False, visualize=True)
    eval_100_kwargs = dict(tries=(100, 0), json_name='eval', save_log=False, visualize=True)
    eval_100_render_kwargs = dict(tries=(100, 0), json_name='eval_2', save_log=True, visualize=True, render=True)

    ##  -----------------  triangle task

    test_10_tasks = {i: f"TriangularRandomSplitWorld[64]_(10)_diffuse_pairwise_test_{i}_split" for i in range(2, 7)}
    test_100_tasks = {i: f"TriangularRandomSplitWorld[64]_(100)_diffuse_pairwise_test_{i}_split" for i in range(2, 7)}
    # evaluate_model('gtqtyd8n', milestone=20, test_tasks=test_100_tasks, render=False, **eval_100_kwargs)
    # evaluate_model('dyz4eido', milestone=23, test_tasks=test_100_tasks, render=False, **eval_100_kwargs)

    ##  -----------------  qualitative task
    test_10_tasks = {i: f'RandomSplitQualitativeWorld(10)_qualitative_test_{i}_split' for i in range(2, 7)}
    test_100_tasks = {i: f'RandomSplitQualitativeWorld(100)_qualitative_test_{i}_split' for i in range(2, 7)}
    evaluate_model('qsd3ju74', milestone=7, test_tasks=test_100_tasks, **eval_100_kwargs)
    # evaluate_model('r26wkb13', milestone=16, test_tasks=test_100_tasks, **eval_100_kwargs)

    ##  -----------------  3d stability tasks
    test_tasks = {i: f"RandomSplitWorld(100)_stability_flat_test_{i}_object" for i in range(5, 8)}
    # evaluate_model('ti639t00', milzestone=3, test_tasks=test_tasks, **eval_100_kwargs)
    # evaluate_model('bo02mwbw', milestone=11, test_tasks=test_tasks, **eval_100_kwargs)

    ## -----------------  3d stability tasks (TAMP pipeline)
    # for k in range(4, 7):
    #     test_tasks = {n: f"RandomSplitWorld(50)_stability_flat_test_{n}_object_all_n={n}_i={k}" for n in range(4, 6)}
    #     evaluate_model('ti639t00', milestone=3, tries=(1, 0), test_tasks=test_tasks, json_name=f'tamp_{k}', run_all=True)
    #     # evaluate_model('bo02mwbw', milestone=11, tries=(1, 0), test_tasks=test_tasks, json_name=f'tamp_{k}', run_all=True)

    ## ----------------- 3d packing tasks
    test_10_tasks = {i: f"TableToBoxWorld(10)_test_{i}_object" for i in range(2, 7)}
    test_100_tasks = {i: f"TableToBoxWorld(100)_robot_box_test_{i}_object" for i in range(3, 7)}
    # evaluate_model('0hhtsh6a', milestone=5, test_tasks=test_100_tasks, **eval_100_kwargs)
    # evaluate_model('qcjqkout', milestone=14, test_tasks=test_100_tasks, **eval_100_kwargs)

    # ## 3d packing tasks (TAMP pipeline)
    # for k in range(3, 9):
    #     test_tasks = {n: f"TableToBoxWorld(50)_robot_box_test_{n}_object_all_n={n}_i={k}" for n in range(2, 7)}
    #     evaluate_model('siovfil2', milestone=6, tries=(1, 0), test_tasks=test_tasks, json_name=f'tamp_{k}', run_all=True)  ## 0.1-0.14 sec per graph
    #     # evaluate_model('4xt8u4n7', milestone=20, tries=(1, 0), test_tasks=test_tasks, json_name=f'tamp_{k}', run_all=True)  ## 0.005-0.01 sec per graph


if __name__ == '__main__':
    indie_runs()
