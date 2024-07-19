#!/bin/bash
python run.py --gpus=0 --spline_degree=3 --half_interval_shift=False --model="Poisson" --n_experiment=100
python run.py --gpus=1 --spline_degree=2 --half_interval_shift=False --model="Poisson" --n_experiment=100
python run.py --gpus=1 --spline_degree=3 --half_interval_shift=True --model="Poisson" --n_experiment=100
python run.py --gpus=1 --spline_degree=2 --half_interval_shift=True --model="Poisson" --n_experiment=100

python run.py --gpus=1 --spline_degree=3 --half_interval_shift=False --spacing=4 --model="Poisson" --n_experiment=100
python run.py --gpus=1 --spline_degree=3 --half_interval_shift=False --spacing=7.5 --model="Poisson" --n_experiment=100
python run.py --gpus=1 --spline_degree=3 --half_interval_shift=False --spacing=10 --model="Poisson" --n_experiment=100
python run.py --gpus=1 --spline_degree=3 --half_interval_shift=False --spacing=15 --model="Poisson" --n_experiment=100
python run.py --gpus=1 --spline_degree=3 --half_interval_shift=False --spacing=20 --model="Poisson" --n_experiment=100
python run.py --gpus=1 --spline_degree=3 --half_interval_shift=False --spacing=30 --model="Poisson" --n_experiment=100
python run.py --gpus=1 --spline_degree=3 --half_interval_shift=False --spacing=40 --model="Poisson" --n_experiment=100