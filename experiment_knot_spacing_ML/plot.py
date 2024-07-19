import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

datasets = ["1_Social_Processing", "2_PTSD", "3_Substance_Use", "4_Dementia",
            "5_Cue_Reactivity", "6_Emotion_Regulation", "7_Decision_Making", "8_Reward",
            "9_Sleep_Deprivation", "10_Naturalistic", "11_Problem_Solving", "12_Emotion",
            "13_Cannabis_Use", "14_Nicotine_Use", "15_Frontal_Pole_CBP", "16_Face_Perception",
            "17_Nicotine_Administration", "18_Executive_Function", "19_Finger_Tapping", "20_n-Back"]
N_FOCI = [4934, 154, 644, 1194, 6322, 3507, 1189, 5602, 454, 1010, 
          3506, 22038, 316, 77, 9525, 2920, 351, 2629, 186, 640]
N_STUDY = [599, 22, 85, 28, 546, 336, 143, 707, 44, 108, 
           322, 1738, 83, 13, 795, 385, 76, 243, 25, 29]
N_FOCI_increasing = sorted(N_FOCI)
sizes = np.linspace(30, 100, 20)
N_FOCI_increasing_index = {N_FOCI.index(N_FOCI_increasing[i])+1:sizes[i] for i in range(len(N_FOCI_increasing))}

spacing_list = [4,5,7.5,10,15,20,30,40]

# Getting a colormap and generating a list of colors
colormap = plt.get_cmap('tab20')  # You can change 'tab20' to other colormaps
colors = [colormap(i) for i in range(len(datasets))]

model = "Poisson"
path = os.getcwd() + "/results/" + model + "_model/"
j = 1
for dset in datasets:
    filename = "neg_l_{}.npy".format(dset)
    neg_l_dset = np.load(path+dset+"/"+filename)
    log_likelihood_dset = - neg_l_dset
    log_likelihood_dset = log_likelihood_dset.tolist()
    relative_l_dset = [(i-log_likelihood_dset[3])/abs(log_likelihood_dset[3]) for i in log_likelihood_dset]
    marker_size = N_FOCI_increasing_index[j]
    plt.scatter(spacing_list, relative_l_dset, label=j, s=marker_size)
    plt.plot(spacing_list, relative_l_dset, color=colors[j-1])
    j += 1
plt.xlabel("spacing of cubic B-spline bases")
plt.ylabel("Relative difference in Maximised Log-Likelihood", fontsize=9)
# Set y-axis formatter to percentage with 0 decimals
# Get the current Axes instance
ax = plt.gca()  # This returns the current Axes instance
# Set y-axis formatter to percentage with 0 decimals
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
plt.legend(fontsize='small', bbox_to_anchor=(0.95, -0.25), loc='upper right', borderaxespad=0., ncol=8)
plt.tight_layout(rect=[0, 0, 1, 1]) 
plt.savefig('Relative_ML_plot.png')