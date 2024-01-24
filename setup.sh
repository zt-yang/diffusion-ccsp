conda activate diffusion-ccsp
export PYTHONPATH=${PWD}/networks:$PYTHONPATH
export PYTHONPATH=${PWD}/envs:$PYTHONPATH
export PYTHONPATH=${PWD}/packing_models:$PYTHONPATH
export PYTHONPATH=${PWD}/:$PYTHONPATH

## by default sharing the same parent folder as Jacinle
export PYTHONPATH=${PWD}/../Jacinle:$PYTHONPATH
