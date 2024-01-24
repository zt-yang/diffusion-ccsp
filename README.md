# Compositional Diffusion-Based Continuous Constraint Solvers

Project page: [Diffusion-CCSP](https://diffusion-ccsp.github.io/)

## Setting Up

* Clone this repo

  ```shell
  git clone https://github.com/zt-yang/diffusion-ccsp.git --recurse-submodules
  ```

* Set up Jacinle following the instructions [here](https://github.com/vacancy/Jacinle). If the directory you installed Jacinle does not share the same parent folder as this repository, put `export PYTHONPATH=/your/path/to/Jacinle:$PYTHONPATH` in `diffusion-ccsp/setup.sh`.

    ```shell
    cd ..
    git clone https://github.com/vacancy/Jacinle --recursive
    ## echo "export PYTHONPATH=/your/path/to/Jacinle:$PYTHONPATH" >> diffusion-ccsp/setup.sh  ## optional
    ```

* Set up dependencies.

    ```shell
    cd diffusion-ccsp
    conda create --name diffusion-ccsp python=3.9
    conda activate diffusion-ccsp
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


### Custom Task
to add a new task

1. run dataset.py to generate the pt files and try evaluation / visualization
2. change dims in create_trainer() in train_utils.py
3. change init() and initiate_denoise_fns() in ConstraintDiffuser class of denoise_fn.py
3. change world.name in Trainer class of ddpm.py
4. train with debug=True and visualize=True
5. change wandb project name

## Training

```shell
python train_ddpm.py -timesteps 1000 -EBM 'ULA' -input_mode qualitative
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
