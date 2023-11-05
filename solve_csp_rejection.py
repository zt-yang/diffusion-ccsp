from os.path import join
from demo_utils import run_rejection_sampling_baseline, rejection_sample_given_solution_json


def rejection_sampling_fn_triangular(solution_json, prediction_json, **kwargs):
    return rejection_sample_given_solution_json(solution_json, prediction_json, **kwargs)


def run_rejection_baselines(json_name=None):
    ## triangle
    run_rejection_sampling_baseline(
        {i: f'TriangularRandomSplitWorld[64]_(100)_diffuse_pairwise_test_{i}_split' for i in range(2, 7)},
        'TriangularRandomSplitWorld[64]_(100)_eval_m=smarter_diffuse_pairwise',
        rejection_sampling_fn_triangular, input_mode='diffuse_pairwise', json_name=json_name
    )
    with open(join('logs2', 'run_solve_csp_2_log.txt'), 'a+') as f:
        f.write(f"python solve_csp_2.py -input_mode diffuse_pairwise -json_name t=0_{json_name}\n")

    ## 2d packing with qualitative constraints
    # run_rejection_sampling_baseline(
    #     {i: f'RandomSplitQualitativeWorld(100)_qualitative_test_{i}_split' for i in range(2, 7)},
    #     'RandomSplitQualitativeWorld(100)_eval_m=smarter_qualitative',
    #     rejection_sampling_fn_triangular, input_mode='qualitative', json_name=json_name
    # )
    # with open(join('logs2', 'run_solve_csp_2_log.txt'), 'a+') as f:
    #     f.write(f"python solve_csp_rejection.py -input_mode qualitative -json_name t=0_{json_name}\n")


if __name__ == '__main__':
    for json_name in range(2, 6):
        run_rejection_baselines(json_name=json_name)
