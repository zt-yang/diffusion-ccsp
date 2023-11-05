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

* Source environment variables before running codes.

    ```shell
    source setup.sh
    conda activate diffusion-ccsp
    ```

* Compile IK for Franka Panda if want to collect and test robot planning.

    ```shell
    (cd pybullet_engine/ikfast/franka_panda; python setup.py)
    ```

## Data Collection

### Task 1-2: 2D Tasks

```shell
## for the first time
mkdir data

## collect data into `data/` folder, .png and .json files will be in `render/` folder
python envs/data_collectors.py \
  -world_name 'RandomSplitWorld' \
  -num_worlds 10 -grid_size 0.5 -pngs -jsons
```

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

## Solving CCSP

```shell
python solve_csp.py
```

## TODO

- [ ] Upload data and checkpoints for evaluation
- [ ] Upload packing model data

## Citation

```shell
@inproceedings{yang2023diffusion,
  title={{Compositional Diffusion-Based Continuous Constraint Solvers}},
  author={Yang, Zhutian and Mao, Jiayuan and Du, Yilun and Wu, Jiajun and Tenenbaum, Joshua B. and Lozano-P{\'e}rez, Tom{\'a}s and Kaelbling, Leslie Pack},
  booktitle={Conference on Robot Learning},
  year={2023},
}
```
