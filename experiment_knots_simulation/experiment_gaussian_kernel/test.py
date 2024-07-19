import numpy as np
import scipy
import os 

def _log_likelihood(beta, gamma, X, y, Z, y_t):
    n_study = y_t.shape[0]
    log_mu_X = X @ beta
    mu_X = np.exp(log_mu_X)
    if gamma is None:
        log_mu_Z = np.zeros(n_study, 1)
        mu_Z = np.ones(n_study, 1)
    else:
        log_mu_Z = Z @ gamma
        mu_Z = np.exp(log_mu_Z)
    # l = [Y_g]^T * log(mu^X) + [Y^t]^T * log(mu^Z) - [1^T mu_g^X]*[1^T mu_g^Z]

    log_l = np.sum(y * log_mu_X) + np.sum(y_t * log_mu_Z) - np.sum(mu_X) * np.sum(mu_Z) 
        
    return log_l

# homogeneous_intensity = [0.01, 0.1, 1.0]
# for intensity in homogeneous_intensity:
#     PATH = os.getcwd() + "/results/with_covariate_results/Homogeneous_{}/Poisson_model/No_penalty/".format(intensity)
#     # spline parameterisation
#     spline_X = np.load(os.getcwd() + "/Spline_bases_matrix/X_spacing5.npy")
#     # spline_beta_dim = spline_X.shape[1]
#     # true_spline_beta = np.log(intensity) * np.ones((spline_beta_dim, 1))
#     spline_beta = np.load(PATH + "Spline_parameterisation/beta.npy")
#     spline_gamma = np.load(PATH + "Spline_parameterisation/gamma.npy")
#     spline_mu_hat = np.exp(np.matmul(spline_X, spline_beta))
    
#     spline_mu_hat_avg = np.mean(spline_mu_hat, axis=1)
#     spline_mu_hat_bias = np.mean(spline_mu_hat_avg, axis=0) - intensity
#     spline_mu_hat_std = np.std(spline_mu_hat, axis=1)
    
#     # gaussian kernel 
#     gaussian_X = np.load(os.getcwd() + "/Gaussian_kernel_matrix/X_spacing5.npy")
#     gaussian_beta = np.load(PATH + "Gaussian_kernel/beta.npy")
#     gaussian_gamma = np.load(PATH + "Gaussian_kernel/gamma.npy")
#     gaussian_mu_hat = np.exp(np.matmul(gaussian_X, gaussian_beta))
#     gaussian_mu_hat_avg = np.mean(gaussian_mu_hat, axis=1)
#     gaussian_mu_hat_bias = np.mean(gaussian_mu_hat_avg, axis=0) - intensity
#     gaussian_mu_hat_std = np.std(gaussian_mu_hat, axis=1)
    
#     spline_mu_hat_std_mean = np.mean(spline_mu_hat_std)
#     gaussian_mu_hat_std_mean = np.mean(gaussian_mu_hat_std)
#     print("For homogeneous intensity: {}".format(intensity))
#     print("Bias of mu_hat with spline parameterisation is {}".format(spline_mu_hat_bias))
#     print("Bias of mu_hat with gaussian kernel is {}".format(gaussian_mu_hat_bias))
#     print("Std of mu_hat with spline parameterisation is {}".format(np.mean(spline_mu_hat_std)))
#     print("Std of mu_hat with gaussian kernel is {}".format(np.mean(gaussian_mu_hat_std)))
#     print("MSE of mu_hat with spline parameterisation is {}".format(spline_mu_hat_bias**2+spline_mu_hat_std_mean**2))
#     print("MSE of mu_hat with gaussian kernel is {}".format(gaussian_mu_hat_bias**2+gaussian_mu_hat_std_mean**2))
#     print("---------------------分割线---------------------")
    
#     n_experiment = 100
#     n_study = 100
#     x_range, y_range = 100, 100
#     spline_l, gaussian_l = [], []
#     for j in range(n_experiment):
#         y = np.zeros(shape=(x_range*y_range, 1))
#         y_t = np.empty(shape=(0,1))
#         intensity_array = intensity * np.ones((x_range, y_range))
#         np.random.seed(j)
#         for i in range(n_study):
#             y_i = np.random.poisson(lam=intensity_array)
#             y_t_i = np.sum(y_i).reshape((-1,1))
#             y += y_i.reshape((-1,1))
#             y_t = np.concatenate((y_t_i, y_t), axis=0)
#         Z = np.random.uniform(-0.01, 0.01, (n_study, 2))
#         spline_beta_j = spline_beta[:, j].reshape((-1, 1))
#         spline_gamma_j = spline_gamma[:, j].reshape((-1, 1))
#         spline_l_j = _log_likelihood(spline_beta_j, spline_gamma_j, spline_X, y, Z, y_t)
#         spline_l.append(spline_l_j)
#         gaussian_beta_j = gaussian_beta[:, j].reshape((-1, 1))
#         gaussian_gamma_j = gaussian_gamma[:, j].reshape((-1, 1))
#         gaussian_l_j = _log_likelihood(gaussian_beta_j, gaussian_gamma_j, gaussian_X, y, Z, y_t)
#         gaussian_l.append(gaussian_l_j)
#     spline_l = np.array(spline_l)
#     gaussian_l = np.array(gaussian_l)
#     bias_l = (gaussian_l - spline_l) / spline_l
#     print("Relative bias of difference in log-likelihood is {}".format(np.mean(bias_l)))
#     print("--------------------------------------------------------")


def gaussian_2d(x, y, x0, y0, sigma):
    # Flatten the grids and stack them as required for the multivariate_normal.pdf function
    pos = np.dstack((x, y))
    mean = (x0, y0)
    covariance = sigma * np.eye(2)
    # Compute the Gaussian PDF
    gaussian_pdf = scipy.stats.multivariate_normal(mean, covariance).cdf(pos)

    return gaussian_pdf
    

inhomogeneous_intensity = [1.0, 10.0, 100.0]
for intensity in inhomogeneous_intensity:
    PATH = os.getcwd() + "/results/with_covariate_results/Gaussian_signal_{}/Poisson_model/No_penalty/".format(intensity)
    # spline parameterisation
    spline_X = np.load(os.getcwd() + "/Spline_bases_matrix/X_spacing5.npy")
    x_range, y_range = 100, 100
    x = np.linspace(0, x_range-1, x_range)
    y = np.linspace(0, y_range-1, y_range)
    X, Y = np.meshgrid(x, y)
    intensity_array = np.zeros((x_range, y_range))
    Gaussian_centers = [(25, 25), (75, 75)]
    sigma = 5
    for center in Gaussian_centers:
        x_0, y_0 = center
        gaussian_intensity = gaussian_2d(X+0.5, Y+0.5, x_0, y_0, sigma) - gaussian_2d(X-0.5, Y+0.5, x_0, y_0, sigma) \
                            - gaussian_2d(X+0.5, Y-0.5, x_0, y_0, sigma) + gaussian_2d(X-0.5, Y-0.5, x_0, y_0, sigma)
        intensity_array += intensity * gaussian_intensity
    intensity_array[intensity_array < 0] = 0
    intensity_array = intensity_array.reshape((-1,1))

    spline_beta = np.load(PATH + "Spline_parameterisation/beta.npy")
    spline_gamma = np.load(PATH + "Spline_parameterisation/gamma.npy")
    
    spline_mu_hat = np.exp(spline_X @ spline_beta)
    spline_mu_hat_avg = np.nanmean(spline_mu_hat, axis=1).reshape((-1,1))
    
    spline_mu_hat_bias = np.mean(spline_mu_hat_avg - intensity_array, axis=0)
    spline_mu_hat_std = np.std(spline_mu_hat_avg - intensity_array, axis=0)
    
    # gaussian kernel 
    gaussian_X = np.load(os.getcwd() + "/Gaussian_kernel_matrix/X_spacing5.npy")
    gaussian_beta = np.load(PATH + "Gaussian_kernel/beta.npy")
    gaussian_gamma = np.load(PATH + "Gaussian_kernel/gamma.npy")
    gaussian_mu_hat = np.exp(np.matmul(gaussian_X, gaussian_beta))
    gaussian_mu_hat_avg = np.nanmean(gaussian_mu_hat, axis=1).reshape((-1,1))
    gaussian_mu_hat_bias = np.nanmean(gaussian_mu_hat_avg - intensity_array, axis=0)
    gaussian_mu_hat_std = np.nanstd(gaussian_mu_hat, axis=1)
    
    spline_mu_hat_std_mean = np.nanmean(spline_mu_hat_std)
    gaussian_mu_hat_std_mean = np.nanmean(gaussian_mu_hat_std)
    print("For homogeneous intensity: {}".format(intensity))
    print("Bias of mu_hat with spline parameterisation is {}".format(spline_mu_hat_bias))
    print("Bias of mu_hat with gaussian kernel is {}".format(gaussian_mu_hat_bias))
    print("Std of mu_hat with spline parameterisation is {}".format(spline_mu_hat_std_mean))
    print("Std of mu_hat with gaussian kernel is {}".format(gaussian_mu_hat_std_mean))
    print("MSE of mu_hat with spline parameterisation is {}".format(spline_mu_hat_bias**2+spline_mu_hat_std_mean**2))
    print("MSE of mu_hat with gaussian kernel is {}".format(gaussian_mu_hat_bias**2+gaussian_mu_hat_std_mean**2))
    print("---------------------分割线---------------------")
    
    n_experiment = 100
    n_study = 100
    x_range, y_range = 100, 100
    spline_l, gaussian_l = [], []
    for j in range(n_experiment):
        y = np.zeros(shape=(x_range*y_range, 1))
        y_t = np.empty(shape=(0,1))
        intensity_array = intensity * np.ones((x_range, y_range))
        np.random.seed(j)
        for i in range(n_study):
            y_i = np.random.poisson(lam=intensity_array)
            y_t_i = np.sum(y_i).reshape((-1,1))
            y += y_i.reshape((-1,1))
            y_t = np.concatenate((y_t_i, y_t), axis=0)
        Z = np.random.uniform(-0.01, 0.01, (n_study, 2))
        spline_beta_j = spline_beta[:, j].reshape((-1, 1))
        spline_gamma_j = spline_gamma[:, j].reshape((-1, 1))
        spline_l_j = _log_likelihood(spline_beta_j, spline_gamma_j, spline_X, y, Z, y_t)
        spline_l.append(spline_l_j)
        gaussian_beta_j = gaussian_beta[:, j].reshape((-1, 1))
        gaussian_gamma_j = gaussian_gamma[:, j].reshape((-1, 1))
        gaussian_l_j = _log_likelihood(gaussian_beta_j, gaussian_gamma_j, gaussian_X, y, Z, y_t)
        gaussian_l.append(gaussian_l_j)
    spline_l = np.array(spline_l)
    gaussian_l = np.array(gaussian_l)
    bias_l = (gaussian_l - spline_l) / spline_l
    print("Relative bias of difference in log-likelihood is {}".format(np.nanmean(bias_l)))
    print("--------------------------------------------------------")