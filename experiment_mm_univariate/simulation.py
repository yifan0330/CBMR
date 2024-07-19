import numpy as np
from scipy.special import loggamma, gammaln, polygamma
import math
from matplotlib import pyplot as plt

def data_generator(n_study, mu_0, alpha, r=42):
    v = 1 / alpha
    # generate heterogeneous study mean rate 
    # mu = [1e-6, 2e-6, ..., 1e-2]
    mu = [mu_0 * (i+1)/n_study * 10 for i in range(n_study)]
    mu_array = np.array(mu).reshape((-1,1))
    # random seed
    np.random.seed(r)
    lambda_array = np.random.gamma(shape=v, scale=1/v, size=(n_study,1))
    # generate count y
    y = np.random.poisson(lambda_array*mu_array)
    
    return y

def moment_matching(n_study, mu_0 = 1e-3, alpha=0.5):
    v = 1 / alpha
    mu = [mu_0 * (i+1)/n_study * 10 for i in range(n_study)]
    mu_array = np.array(mu).reshape((-1,1))
    mu_sum_study = np.sum(mu_array)
    mu_square_sum = np.sum(mu_array**2)
    # params of moment matching NB
    r = v * mu_sum_study**2 / mu_square_sum
    p = mu_square_sum / (v*mu_sum_study + mu_square_sum)
    alpha_prime = alpha * mu_square_sum / mu_sum_study**2
    # mean = r(1-p)/p, var = r(1-p)/p^2
    mm_mean = r * p / (1-p)
    mm_var = r * p / (1-p)**2
    mm_std = np.sqrt(mm_var)
    
    return mm_mean, mm_std

def l_exact_NB_sum(y, alpha, mu_0):
    n_study, _ = y.shape
    mu = [mu_0 * (i+1)/n_study * 10 for i in range(n_study)]
    mu_array = np.array(mu).reshape((-1,1))
    mu_sum = np.sum(mu_array)
    y_sum = np.sum(y)
    v = 1 / alpha
    Likelihood = loggamma(y + v) - loggamma(y+1) - loggamma(v) \
        + y * np.log(alpha*mu_array) - (y+v) * np.log(1+alpha*mu_array)
    # log-likelihood
    l = np.sum(Likelihood)

    return l

def l_moment_matching(y, alpha=0.5, mu_0=1e-3):
    n_study, _ = y.shape
    v = 1 / alpha
    mu = [mu_0 * (i+1)/n_study * 10 for i in range(n_study)]
    mu_array = np.array(mu).reshape((-1,1))
    mu_sum_study = np.sum(mu_array)
    mu_square_sum = np.sum(mu_array**2)
    mu_sum, y_sum = np.sum(mu_array), np.sum(y)
    # # dispersion parameter: alpha'
    # alpha_prime = mu_square_sum / mu_sum_study**2 * alpha
    # # likelihood function
    # v_prime = 1 / alpha_prime
    # l = loggamma(y_sum + v_prime) - loggamma(y_sum+1) - loggamma(v_prime) \
    #     + y_sum * np.log(mu_sum/(mu_sum+v_prime)) + v_prime * np.log(v_prime/(mu_sum+v_prime))
    # print(l)
    # print('234')
    # exit()
    
    # params of moment matching NB
    r = v * mu_sum_study**2 / mu_square_sum
    p = mu_square_sum / (v*mu_sum_study + mu_square_sum)
    y_sum = np.sum(y)
    # probability density function:
    # Pr(Y=y) = Gamma(y+r)/(y! * Gamma(r)) * (1-p)^y * p^r
    # l = logGamma(y+r) - log(y!) - logGamma(r) + y*log(p) + r*log(1-p)
    l = loggamma(y_sum+r) - gammaln(y_sum + 1) - loggamma(r) \
        + y_sum * np.log(p) + r * np.log(1-p)
    # l' = psi(y+r) + psi(y+1) + log(p) where psi is digamma function
    # l'' = psi_1(y+r) + psi_1(y+1) where psi_1 is trigamma function
    FI = polygamma(n=1, x=y_sum+r) + polygamma(n=1, x=y_sum+1)
    Var = 1 / FI
    std = np.sqrt(Var)

    return l, std


# 1000 MC realisations
n_experiment = 1000
n_study = 10000
MC_mu_hat, MM_mu_SE = [], []
y_sum, l_exact_values, l_mm_values = [], [], []
for i in range(n_experiment):
    y_i = data_generator(n_study=n_study, mu_0=1e-3, alpha=0.5, r=i)
    # Likelihood function
    l_exact = l_exact_NB_sum(y=y_i, alpha=0.5, mu_0=1e-3)
    # print("l_exact: {}".format(l_exact))
    l_mm, se_mm = l_moment_matching(y=y_i, alpha=0.5, mu_0=1e-3)
    # print("l_moment_matching: {}".format(l_moment_matching))
    MM_mu_SE.append(se_mm)
    # save to list
    y_sum.append(np.sum(y_i))
    l_exact_values.append(l_exact)
    l_mm_values.append(l_mm)
    # MLE of mu=sum_i mu_i is sample sum
    mu_hat_i = np.sum(y_i)
    MC_mu_hat.append(mu_hat_i)
# convert to numpy array
l_exact_values = np.array(l_exact_values)
l_mm_values = np.array(l_mm_values)
# standard deviation
MC_mu_hat = np.array(MC_mu_hat).reshape((-1,1))
# Std of MonteCarlo sampled mu_hat
MC_mu_hat_std = np.std(MC_mu_hat)
# moment-matched mean and std for the combined NB distribution
MM_mean, MM_std = moment_matching(n_study=n_study, mu_0 = 1e-3, alpha=0.5)
# SE of moment-matched mu_hat
MM_mu_SE = np.array(MM_mu_SE).reshape((-1,1))
print(MC_mu_hat_std)
print(np.mean(MM_mu_SE))
exit()
# mu = sum_i mu_i = mu_0 * 10 * (N+1)/2
y_sum = np.array(y_sum)
mu_0_hat = y_sum * 1/10 * 2/(n_study+1)


# retrieve the mu_0_hat with maximised log-likelihood
index_largest_l = np.argmax(l_mm_values)
y_i = data_generator(n_study=n_study, mu_0=1e-3, alpha=0.5, r=index_largest_l)
mu_0_largest_l = np.sum(y_i) * 1/10 * 2/(n_study+1)
print(mu_0_largest_l)
exit()

# plot of 2 like-lihood functions
# print('start plotting figures')
# plt.scatter(mu_0_hat, l_exact_values)
# plt.axvline(x=mu_0_largest_l, color='r', linestyle='--', label='MLE of mu_0')
# plt.legend(loc='upper left')
# plt.title('Exact log-likelihood function of the NB distributions')
# plt.xlabel('mu_0_hat')
# plt.ylabel('log-likelihood')
# plt.savefig("l_exact.jpg")

print(mu_0_hat.shape, l_mm_values.shape)
np.save('mu_0_hat.npy', mu_0_hat)
np.save('l_mm_values.npy', l_mm_values)

plt.scatter(mu_0_hat, l_mm_values)
# plt.axvline(x=mu_0_largest_l, color='r', linestyle='--', label='MLE of mu_0')
# plt.legend(loc='upper left')
plt.title('Log-likelihood function of the moment matched NB distribution')
plt.xlabel('mu_0_hat')
plt.ylabel('log-likelihood')
plt.savefig("l_mm.jpg")


# fig, ax1 = plt.subplots()
# color = 'tab:red'
# ax1.set_xlabel('mu_0_hat')
# ax1.set_ylabel('exact log-likelihood', color=color)
# ax1.scatter(mu_0_hat, l_exact_values, color=color)
# ax1.tick_params(axis='y', labelcolor=color)

# ax2 = ax1.twinx()

# color = 'tab:blue'
# ax2.set_ylabel('moment matched log-likelihood', color=color)
# ax2.scatter(mu_0_hat, l_mm_values)
# ax2.tick_params(axis='y', labelcolor=color)
# fig.savefig("test.jpg")


    
  