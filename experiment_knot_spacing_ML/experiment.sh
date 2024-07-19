#!/bin/bash
python run.py --gpus=0 --dataset=1_Social_Processing --model="Poisson"
python run.py --gpus=0 --dataset=2_PTSD --model="Poisson"
python run.py --gpus=0 --dataset=3_Substance_Use --model="Poisson"
python run.py --gpus=0 --dataset=4_Dementia --model="Poisson"
python run.py --gpus=0 --dataset=5_Cue_Reactivity --model="Poisson"

python run.py --gpus=0 --dataset=6_Emotion_Regulation --model="Poisson"
python run.py --gpus=0 --dataset=7_Decision_Making --model="Poisson"
python run.py --gpus=0 --dataset=8_Reward --model="Poisson"
python run.py --gpus=0 --dataset=9_Sleep_Deprivation --model="Poisson"
python run.py --gpus=0 --dataset=10_Naturalistic --model="Poisson"

python run.py --gpus=0 --dataset=11_Problem_Solving --model="Poisson"
python run.py --gpus=0 --dataset=12_Emotion --model="Poisson"
python run.py --gpus=0 --dataset=13_Cannabis_Use --model="Poisson"
python run.py --gpus=0 --dataset=14_Nicotine_Use --model="Poisson"
python run.py --gpus=0 --dataset=15_Frontal_Pole_CBP --model="Poisson"

python run.py --gpus=0 --dataset=16_Face_Perception --model="Poisson"
python run.py --gpus=0 --dataset=17_Nicotine_Administration --model="Poisson"
python run.py --gpus=0 --dataset=18_Executive_Function --model="Poisson"
python run.py --gpus=0 --dataset=19_Finger_Tapping --model="Poisson"
python run.py --gpus=0 --dataset=20_n-Back --model="Poisson"