import numpy as np
import os

datasets = ["1_Social_Processing", "2_PTSD", "3_Substance_Use", "4_Dementia",
            "5_Cue_Reactivity", "6_Emotion_Regulation", "7_Decision_Making", "8_Reward",
            "9_Sleep_Deprivation", "10_Naturalistic", "11_Problem_Solving", "12_Emotion",
            "13_Cannabis_Use", "14_Nicotine_Use", "15_Frontal_Pole_CBP", "16_Face_Perception",
            "17_Nicotine_Administration", "18_Executive_Function", "19_Finger_Tapping", "20_n-Back"]
spacing_list = [4,5,7.5,10,15,20,30,40]

y = np.load("y/y.npy")
empirical_std = np.std(y)

model = "Poisson"
path = os.getcwd() + "/results/" + model + "_model/"
j = 1
for dset in datasets:
    mu_array_dset = np.empty(shape=(228483, 0))
    path_dset = path + dset + "/"
    for spacing in spacing_list:  
        filename = "beta_spacing{}_{}.npy".format(spacing, dset)
        beta = np.load(path_dset + filename)
        X = np.load("X/X_{}.npy".format(spacing))
        mu_hat = np.exp(X @ beta)
        mu_array_dset = np.concatenate((mu_array_dset, mu_hat), axis=1)
    # bias
    bias_mu_dset = mu_array_dset - mu_array_dset[:, 3].reshape((-1,1))
    bias_mu_mean_dset = np.mean(np.abs(bias_mu_dset), axis=0)
    rel_bias_mu_mean_dset = bias_mu_mean_dset / np.mean(mu_array_dset[:, 3])
    # std
    std_mu_dset = np.std(mu_array_dset, axis=0)
    rel_std_mu_dset = std_mu_dset / np.mean(mu_array_dset[:, 3])
    # MSE
    RMSE = np.sqrt(bias_mu_mean_dset**2 + std_mu_dset**2)
    print(dset)
    print("\%$ & $".join([str(100*float(f"{i:.4f}")) for i in rel_bias_mu_mean_dset]))
    print("\%$ & $".join([str(100*float(f"{i:.4f}")) for i in rel_std_mu_dset]))
    print("$ & $".join([str(f"{i:.4e}") for i in RMSE]))
    print("---------------------")