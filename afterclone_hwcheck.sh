#!/usr/bin/env bash

# this is a very naive workaround conda issue
if [[ -z "$CONDA_PREFIX" ]]; then
  python3 -m venv env
  . env/bin/activate
else
  poetry config virtualenvs.path $CONDA_PREFIX/envs
  poetry config virtualenvs.create false
  conda create -p ./env -y python=3.9
  # I have no idea, what I'm doing
  # this is from stackoverflow.com/questions/34534513/calling-conda-source-activate-from-bash-script
  eval "$(conda shell.bash hook)"
  conda activate ./env
fi

poetry install
pre-commit install
pre-commit run -a
python train.py
python infer.py

# this is a very naive workaround conda issue
if [[ -z "$CONDA_PREFIX" ]]; then
  deactivate
else
  conda deactivate
  poetry config --unset virtualenvs.create
  poetry config --unset virtualenvs.path
fi
