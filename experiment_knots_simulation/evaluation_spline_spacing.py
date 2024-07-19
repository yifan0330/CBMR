import numpy as np
import os
# Parameters
spline_degree = 3
half_interval_shift = False
model = "Poisson"
all_spline_spacing = [4.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 40.0]

# intensity: shape = (228483, 1)
#            sum = 7.9205668862249325
actual_intensity = np.load("intensity_func_[(25, 25, 25), (65, 65, 65)].npy")
y = np.load("simulated_data/y.npy")
y_t = np.load("simulated_data/y_t.npy")

def log_l(mu_X, y, y_t):
    log_mu_X = np.log(mu_X)
    n_study, = y_t.shape
    log_mu_Z = np.zeros(shape=(n_study, 1))
    mu_Z = np.ones(shape=(n_study, 1))
    # log-likelihood
    log_l = np.sum(y*log_mu_X) + np.sum(y_t*log_mu_Z) - np.sum(mu_X) * np.sum(mu_Z)
    
    return log_l

actual_ll = log_l(actual_intensity, y, y_t)

for spacing in all_spline_spacing:
    # path
    if spline_degree == 3:
        folder_0 = 'Cubic_spline'
    elif spline_degree == 2:
        folder_0 = 'Quadratic_spline'
    if half_interval_shift:
        folder_1 = 'half_interval_shift'
    else:
        folder_1 = 'no_interval_shift'
    folder_path = os.getcwd() + "/results/" + folder_0 + "/" + folder_1 + "/{}_model/".format(model)
    filename = "beta_{}_spacing_{}.npy".format(model, spacing)
    # negative log-likelihood
    neg_l = np.load(folder_path + 'neg_l_spacing_{}.npy'.format(spacing))
    # load beta
    X = np.load(os.getcwd()+'/X/X_{}_spline_degree_{}_shift_{}.npy'.format(spacing, spline_degree, half_interval_shift))
    n_voxel, _ = X.shape
    beta = np.load(folder_path+filename)
    _, n_experiment = beta.shape
    # mu
    mu = np.exp(np.matmul(X, beta))
    _, nan_index = np.where(np.isnan(mu))
    nan_index = np.unique(nan_index).tolist()
    mu = np.delete(mu, nan_index, axis=1)
    print(mu.shape)
    mu_mean = np.mean(mu, axis=1).reshape((-1,1))
    # bias
    mu_bias = mu_mean - actual_intensity
    relative_mu_bias = np.mean(mu_bias/actual_intensity)
    # std 
    mu_std = np.std(mu, axis=1).reshape((-1,1))
    relative_mu_std = np.mean(mu_std/actual_intensity)
    # rmse
    rmse = np.mean(np.sqrt(mu_bias**2 + mu_std**2))
    # maximised log-likelihood
    mll = -np.nanmean(neg_l)
    relative_mll = (mll - actual_ll) / actual_ll
    print(spacing)
    print(relative_mu_bias)
    print(relative_mu_std)
    print(rmse)
    print(relative_mll)
    print("--------------")