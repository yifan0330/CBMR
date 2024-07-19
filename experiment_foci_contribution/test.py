import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.interpolate import make_interp_spline

datasets = ["5_Cue_Reactivity", "6_Emotion_Regulation", "11_Problem_Solving", "16_Face_Perception", "18_Executive_Function"]


spline_spacings = [4, 5, 7.5, 10, 15, 20, 30, 40]
model = "NB"



for dset in datasets:
    folder_path = os.getcwd() + "/outcomes/{}_model/{}/".format(model, dset)
    dset_fail_rate = list()
    for spacing in spline_spacings:
        filename = "beta_spacing_{}.npy".format(spacing)
        beta = np.load(folder_path + filename)
        X = np.load("X/X_{}.npy".format(spacing))
        eta = np.matmul(X, beta)
        mu = np.exp(eta)
        # compute the number of simulations includes NaN
        n_experiment = mu.shape[1]
        _, nan_index = np.where(np.isnan(mu))
        nan_index = np.unique(nan_index)
        n_fails = nan_index.shape[0]
        dset_fail_rate.append(n_fails / n_experiment)
    print(dset, dset_fail_rate)