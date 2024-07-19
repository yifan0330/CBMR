#!/bin/bash
#
python run.py --gpus=0 --dataset=1_Social_Processing --model=NB --spline_degree=3 --half_interval_shift=False
python run.py --gpus=0 --dataset=2_PTSD --spline_degree=3 --half_interval_shift=False
python run.py --gpus=0 --dataset=3_Substance_Use --spline_degree=3 --half_interval_shift=False
python run.py --gpus=0 --dataset=4_Dementia --spline_degree=3 --half_interval_shift=False
python run.py --gpus=0 --dataset=5_Cue_Reactivity --spline_degree=3 --half_interval_shift=False

python run.py --gpus=0 --dataset=6_Emotion_Regulation --spline_degree=3 --half_interval_shift=False
python run.py --gpus=0 --dataset=7_Decision_Making --spline_degree=3 --half_interval_shift=False
python run.py --gpus=0 --dataset=8_Reward --spline_degree=3 --half_interval_shift=False
python run.py --gpus=0 --dataset=9_Sleep_Deprivation --spline_degree=3 --half_interval_shift=False
python run.py --gpus=0 --dataset=10_Naturalistic --spline_degree=3 --half_interval_shift=False

python run.py --gpus=0 --dataset=11_Problem_Solving --spline_degree=3 --half_interval_shift=False
python run.py --gpus=0 --dataset=12_Emotion --spline_degree=3 --half_interval_shift=False
python run.py --gpus=0 --dataset=13_Cannabis_Use --spline_degree=3 --half_interval_shift=False
python run.py --gpus=0 --dataset=14_Nicotine_Use --spline_degree=3 --half_interval_shift=False
python run.py --gpus=0 --dataset=15_Frontal_Pole_CBP --spline_degree=3 --half_interval_shift=False

python run.py --gpus=0 --dataset=16_Face_Perception --spline_degree=3 --half_interval_shift=False
python run.py --gpus=0 --dataset=17_Nicotine_Administration --spline_degree=3 --half_interval_shift=False
python run.py --gpus=0 --dataset=18_Executive_Function --spline_degree=3 --half_interval_shift=False
python run.py --gpus=0 --dataset=19_Finger_Tapping --spline_degree=3 --half_interval_shift=False
python run.py --gpus=0 --dataset=20_n-Back --spline_degree=3 --half_interval_shift=False