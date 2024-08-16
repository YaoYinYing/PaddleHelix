#!/bin/bash

ENV_NAME='helixfold'
CUDA=12.0

# follow https://developer.nvidia.com/cuda-downloads to install cuda and cudatoolkit

# Install py env
conda create -n ${ENV_NAME} -y -c conda-forge  pip  python=3.9;
source activate ${ENV_NAME}
conda install -y cudnn=8.4.1 cudatoolkit=11.7 nccl=2.14.3 -c conda-forge -c nvidia

conda install -y -c bioconda hmmer==3.3.2 kalign2==2.04 hhsuite==3.3.0 
conda install -y -c conda-forge openbabel

python -m pip install --upgrade 'pip<24';pip install .  --no-cache-dir

pip install https://paddle-wheel.bj.bcebos.com/2.5.1/linux/linux-gpu-cuda11.7-cudnn8.4.1-mkl-gcc8.2-avx/paddlepaddle_gpu-2.5.1.post117-cp39-cp39-linux_x86_64.whl
