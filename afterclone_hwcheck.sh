#!/usr/bin/env bash
python3 -m venv env
. env/bin/activate
poetry install
pre-commit install
pre-commit run -a
python train.py
python infer.py
