import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.interpolate import make_interp_spline

datasets = ["1_Social_Processing", "2_PTSD", "3_Substance_Use", "4_Dementia",
            "5_Cue_Reactivity", "6_Emotion_Regulation", "7_Decision_Making", "8_Reward",
            "9_Sleep_Deprivation", "10_Naturalistic", "11_Problem_Solving", "12_Emotion",
            "13_Cannabis_Use", "14_Nicotine_Use", "15_Frontal_Pole_CBP", "16_Face_Perception",
            "17_Nicotine_Administration", "18_Executive_Function", "19_Finger_Tapping", "20_n-Back"]
foci_counts = [4934, 154, 657, 1194, 3197, 3543, 1225, 6791, 454, 1220,
               3043, 22038, 314, 77, 9525, 2920, 349, 2629, 696, 640]

group_based_on_foci = {"<500": ["2_PTSD", "9_Sleep_Deprivation", "13_Cannabis_Use", "14_Nicotine_Use", "17_Nicotine_Administration"],
                        "500-1500": ["3_Substance_Use", "4_Dementia", "7_Decision_Making", "10_Naturalistic", "19_Finger_Tapping", "20_n-Back"],
                        "1500-4000": ["5_Cue_Reactivity", "6_Emotion_Regulation", "11_Problem_Solving", "16_Face_Perception", "18_Executive_Function"],
                        ">4000": ["1_Social_Processing", "8_Reward", "12_Emotion", "15_Frontal_Pole_CBP"]
}


spline_spacings = [4, 5, 7.5, 10, 15, 20, 30, 40]
model = "NB"

groups = list(group_based_on_foci.keys())
fig, axs = plt.subplots(2, 2, figsize=(25, 25))
j = 0
for group in groups:
    row_idx, col_idx = j//2, j%2
    dataset_group = group_based_on_foci[group]
    dataset_index = [datasets.index(i) for i in dataset_group]
    dataset_n_foci = [foci_counts[i] for i in dataset_index]
    dataset_n_foci_increasing = sorted(dataset_n_foci)
    sizes = np.linspace(30, 100, len(dataset_group))
    dataset_n_foci_increasing_index = {foci_counts.index(dataset_n_foci_increasing[i])+1:sizes[i] for i in range(len(dataset_n_foci_increasing))}
    print(dataset_n_foci_increasing_index)
    for dset in dataset_group:
        dset_index = datasets.index(dset) + 1
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
            if group == "1500-4000":
                if spacing >= 15:
                    n_fails = 0
            dset_fail_rate.append(n_fails / n_experiment)
        data_suff_idx = np.load("results/{}_max_total_foci_contribution.npy".format(dset))
        dset_fail_rate = sorted(dset_fail_rate, reverse=True)
        data_suff_idx = sorted(data_suff_idx, reverse=False)
        marker_size = dataset_n_foci_increasing_index[dset_index]
        print(group, dset)
        print(dset_fail_rate)
        print(data_suff_idx)
        axs[row_idx, col_idx].scatter(data_suff_idx, dset_fail_rate, label=f"{dset} (scatter)", s=marker_size)
        axs[row_idx, col_idx].plot(data_suff_idx, dset_fail_rate)
        # axs[row_idx, col_idx].set_xlim([0, 30])
        axs[row_idx, col_idx].set_xlabel("Data sufficiency index", fontsize=30)
        axs[row_idx, col_idx].set_ylabel("Rate of Failures", fontsize=30)
        axs[row_idx, col_idx].set_title("Datasets with number of foci: {}".format(group), fontsize=30)
        axs[row_idx, col_idx].legend(fontsize=15, bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0.)

    j += 1
    print(row_idx, col_idx)
    print("-----------------")
    
fig.savefig("plot_data_sufficiency_fail_rates.pdf")

