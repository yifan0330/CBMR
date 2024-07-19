import numpy as np
import os
import matplotlib.pyplot as plt

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
print(N_FOCI_increasing)
sizes = np.linspace(30, 100, 20)
N_FOCI_increasing_index = {N_FOCI.index(N_FOCI_increasing[i])+1:sizes[i] for i in range(len(N_FOCI_increasing))}
print(N_FOCI_increasing_index)

spacings = [4, 5, 7.5, 10, 15, 20, 30, 40]
path = os.getcwd() + "/results/"
j = 1
for dataset in datasets:
    filename = "{}_max_total_foci_contribution.npy".format(dataset)
    max_total_foci_contribution = np.load(path+filename)
    marker_size = N_FOCI_increasing_index[j]
    plt.scatter(spacings, max_total_foci_contribution, label=j, s=marker_size)
    j += 1
plt.xlabel("spacing of cubic B-spline bases")
plt.ylabel("Maximum of total foci contribution per basis")
# Set logarithmic scale for y-axis
plt.yscale('log')
plt.legend(fontsize='small', bbox_to_anchor=(0.95, -0.25), loc='upper right', borderaxespad=0., ncol=8)
plt.tight_layout(rect=[0, 0, 1, 1]) 
plt.savefig('max_total_foci_contribution.pdf')