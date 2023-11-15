# Compositional Diffusion-Based Continuous Constraint Solvers

Project page: [Diffusion-CCSP](https://diffusion-ccsp.github.io/)

## Setting Up

* Set up Jacinle following the instructions [here](https://github.com/vacancy/Jacinle).

    ```shell
    git clone https://github.com/vacancy/Jacinle --recursive
    ```

* Set up dependencies.

    ```shell
    conda create --name diffusion-ccsp python=3.9
    pip install -r requirements.txt
    ```

* Source environment variables before running codes (includes `conda activate diffusion-ccsp`).

    ```shell
    source setup.sh
    ```

* Compile IK for Franka Panda if want to collect and test robot planning.

    ```shell
    (cd pybullet_engine/ikfast/franka_panda; python setup.py)
    ```

## Download data and pre-trained models

By default, download for task `RandomSplitQualitativeWorld`. Download into `data/`, `logs/`, and `wandb/` folder

```shell
python download_data_checkpoints.py
```

## Solving CCSP

```shell
python solve_csp.py
```

## Data Collection

### Task 1-2: 2D Tasks

Generate data into `data/` folder

```shell
python envs/data_collectors.py -world_name 'RandomSplitQualitativeWorld' -data_type 'train' -num_worlds 100
python envs/data_collectors.py -world_name 'RandomSplitQualitativeWorld' -data_type 'test' -num_worlds 10 -pngs -jsons
```

<details><summary>Some frequently used flags</summary>

* `-world_name = RandomSplitWorld | TriangularRandomSplitWorld | RandomSplitQualitativeWorld`: generates different geometric splitting datasets
* `-num_worlds`: number of data 
* `-pngs | -jsons`: .png and .json files will be in `render/{dataset_name}` folder

</details>

### Task 3-4: 3D & Robot Data

```shell
## task 4: packing 3D objects
python 3-panda-box-data.py

## task 3: stacking shapes
python 5-panda-stability-data.py
```

## Training

```shell
python train_ddpm.py -timesteps 1000 -EBM 'ULA+'
```

## Citation

```shell
@inproceedings{yang2023diffusion,
  title={{Compositional Diffusion-Based Continuous Constraint Solvers}},
  author={Yang, Zhutian and Mao, Jiayuan and Du, Yilun and Wu, Jiajun and Tenenbaum, Joshua B. and Lozano-P{\'e}rez, Tom{\'a}s and Kaelbling, Leslie Pack},
  booktitle={Conference on Robot Learning},
  year={2023},
}
```
