import numpy as np
from scipy.special import loggamma, gammaln, polygamma
from matplotlib import pyplot as plt
def l_exact_NB_sum(alpha, mu_0, n_study, r=42):
    # sample different mu_0
    mu_0_array = np.linspace(start=0.1*mu_0, stop=1.9*mu_0, num=1000)
    np.random.seed(r)
    v = 1 / alpha
    lambda_array = np.random.gamma(shape=v, scale=1/v, size=(n_study,1))
    # generate count y
    y = np.random.poisson(lambda_array*mu_0, size=(n_study, 1))
    # compute log-likelihood
    l_array = []
    for mu_0 in mu_0_array:
        mu_array = np.array([mu_0]*n_study).reshape((-1,1))
        Likelihood = loggamma(y + v) - loggamma(y+1) - loggamma(v) \
            + y * np.log(alpha*mu_array) - (y+v) * np.log(1+alpha*mu_array)
        # log-likelihood
        l = np.sum(Likelihood)
        l_array.append(l)
    l_array = np.array(l_array)
    
    return mu_0_array, l_array


mu_0 = 1e-3
n_study = 100000
alpha = 0.5
mu_0_array, l_exact_values = l_exact_NB_sum(alpha, mu_0, n_study)
index_largest_l = np.argmax(l_exact_values)
mu_0_largest_l = mu_0_array[index_largest_l]

mu_0_hat = np.load('mu_0_hat.npy')
l_mm_values = np.load('l_mm_values.npy')

fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('mu_0_hat')
ax1.set_ylabel('exact log-likelihood', color=color)
curve1 = ax1.scatter(mu_0_array, l_exact_values, color=color)
ax1.axvline(x=0.0010396396396396397, color=color, linestyle='--', label='(Exact) MLE of mu_0')
# ax1.legend(loc='lower left')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()

color = 'tab:blue'
ax2.set_ylabel('moment matched log-likelihood', color=color)
curve2 = ax2.scatter(mu_0_hat, l_mm_values, color=color)
ax2.axvline(x=0.000999900009999, color=color, linestyle='--', label='(Moment Matched) MLE of mu_0')
# ax2.legend(loc='lower right')
ax2.tick_params(axis='y', labelcolor=color)

# Combine legends
handles, labels = [], []
for ax in [ax1, ax2]:
    h, l = ax.get_legend_handles_labels()
    handles.extend(h)
    labels.extend(l)

ax1.legend(handles, labels, loc='lower right')

fig.savefig("combined_plot.pdf")