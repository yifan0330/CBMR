#!/bin/bash
# python run.py --gpus=1 --dataset='1_Social_Processing' --model='NB' --penalty='No' --covariates=True --lr=0.005
python run.py --gpus=1 --dataset='2_PTSD' --model='NB' --penalty='No' --covariates=True --lr=0.005
python run.py --gpus=1 --dataset='3_Substance_Use' --model='NB' --penalty='No' --covariates=True --lr=0.005
python run.py --gpus=2 --dataset='4_Dementia' --model='NB' --penalty='No' --covariates=True --lr=0.005
python run.py --gpus=1 --dataset='5_Cue_Reactivity' --model='NB' --penalty='No' --covariates=True --lr=0.005
python run.py --gpus=1 --dataset='6_Emotion_Regulation' --model='NB' --penalty='No' --covariates=True --lr=0.005
python run.py --gpus=1 --dataset='7_Decision_Making' --model='NB' --penalty='No' --covariates=True --lr=0.005
python run.py --gpus=1 --dataset='8_Reward' --model='NB' --penalty='No' --covariates=True --lr=0.005
python run.py --gpus=1 --dataset='9_Sleep_Deprivation' --model='NB' --penalty='No' --covariates=True --lr=0.005
python run.py --gpus=1 --dataset='10_Naturalistic' --model='NB' --penalty='No' --covariates=True --lr=0.005
python run.py --gpus=1 --dataset='11_Problem_Solving' --model='NB' --penalty='No' --covariates=True --lr=0.005
python run.py --gpus=1 --dataset='12_Emotion' --model='NB' --penalty='No' --covariates=True --lr=0.005
python run.py --gpus=1 --dataset='13_Cannabis_Use' --model='NB' --penalty='No' --covariates=True --lr=0.005
python run.py --gpus=1 --dataset='14_Nicotine_Use' --model='NB' --penalty='No' --covariates=True --lr=0.005
python run.py --gpus=1 --dataset='15_Frontal_Pole_CBP' --model='NB' --penalty='No' --covariates=True --lr=0.005
python run.py --gpus=1 --dataset='16_Face_Perception' --model='NB' --penalty='No' --covariates=True --lr=0.005
python run.py --gpus=1 --dataset='17_Nicotine_Administration' --model='NB' --penalty='No' --covariates=True --lr=0.005
python run.py --gpus=1 --dataset='18_Executive_Function' --model='NB' --penalty='No' --covariates=True --lr=0.005
python run.py --gpus=1 --dataset='19_Finger_Tapping' --model='NB' --penalty='No' --covariates=True --lr=0.005
python run.py --gpus=1 --dataset='20_n-Back' --model='NB' --penalty='No' --covariates=True --lr=0.005

