#!/usr/bin/env bash

port=${1:-5000}
tracking_uri="http://localhost:$port"

init ()
{
  poetry install &&
  # dvc pull before pre-commit to check .yaml configs
  dvc pull conf.dvc &&

  pre-commit install &&
  pre-commit run -a
}

cleanup()
{
  if [[ -z "$CONDA_PREFIX" ]]; then
    deactivate
  else
    conda deactivate
    poetry config --unset virtualenvs.create
    poetry config --unset virtualenvs.path
  fi
}

if [[ -z "$CONDA_PREFIX" ]]; then
  python3 -m venv env
  . env/bin/activate
else
  # I have no idea, what I'm doing
  # this is from https://michhar.github.io/2023-07-poetry-with-conda/
  poetry config virtualenvs.path $CONDA_PREFIX/envs
  poetry config virtualenvs.create false

  conda create -p ./env -y python=3.9

  # I have no idea, what I'm doing
  # this is from stackoverflow.com/questions/34534513/calling-conda-source-activate-from-bash-script
  eval "$(conda shell.bash hook)"

  conda activate ./env
fi

if init; then
  {
    if ! mlflow server --host localhost --port $port; then "failed to run mlflow server"; fi
  } & {
    sleep 1
    if ! (
      python train.py tracking_uri="$tracking_uri" && python infer.py tracking_uri="$tracking_uri"
    ); then
      echo "failed to run train/infer scripts"
    fi
  } & {
    trap "kill 0" SIGINT
    wait -n
  }
  kill 0 & wait

  cleanup
else
  cleanup
  echo "couldn't prepare environment T_T"
  exit 1
fi
