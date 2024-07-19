import torch
import os
import numpy as np
import scipy
from scipy.stats import multivariate_normal
from models import GLMPoisson, GLMNB, GLMClustered_NB, GLMZIP, GLMQPOI
import nimare
import nibabel as nib 
import copy
from matplotlib import pyplot as plt
import seaborn as sns
import sklearn
import itertools
import matplotlib.ticker as mticker
import time 
import re
import patsy
import sparse


class train_dataset(object):
    def __init__(self, foci_per_study, model, covariates, penalty, spline_degree, half_interval_shift, spacing, device='cpu'):
        self.device = device
        self.foci_per_study = foci_per_study
        self.model = model
        self.covariates = covariates
        self.penalty = penalty
        self.spline_degree = spline_degree
        self.half_interval_shift = half_interval_shift
        self.spacing = spacing
        self.device = device 
        self.load_dataset()
        self.load_data()
        
    def load_dataset(self, dataset = "5_Cue_Reactivity"):
        cwd = os.getcwd() 
        dset_PATH = os.path.dirname(cwd) + '/data/' + dataset + '/'
        Files = sorted([f for f in os.listdir(dset_PATH) if os.path.isfile(os.path.join(dset_PATH, f))]) # Filename by alphabetical order
        group_name = [re.findall('^[^_]*[^ _]', g)[0] for g in Files] # extract group name
        self.group_name = sorted(list(set(group_name))) # unique list
        self.all_dset = dict()
        for group in self.group_name:
            MNI_file_exist = os.path.isfile(dset_PATH + group + '_MNI.txt')
            Talairach_file_exist = os.path.isfile(dset_PATH + group + '_Talairach.txt')
            if MNI_file_exist and Talairach_file_exist:
                group_dset_MNI = nimare.io.convert_sleuth_to_dataset(text_file=dset_PATH + group + '_MNI.txt')
                group_dset_Talairach = nimare.io.convert_sleuth_to_dataset(text_file=dset_PATH + group + '_Talairach.txt')
                group_dset = group_dset_MNI.merge(group_dset_Talairach)
            else: 
                if MNI_file_exist: 
                    group_dset = nimare.io.convert_sleuth_to_dataset(text_file=dset_PATH + group + '_MNI.txt') 
                elif Talairach_file_exist: 
                    group_dset = nimare.io.convert_sleuth_to_dataset(text_file=dset_PATH + group + '_Talairach.txt')
            # remove dataset coordinates outside of mask
            if not hasattr(self, 'mask'):
                self.mask = group_dset.masker
                self.n_voxel = self.mask.n_elements_
            else: 
                if group_dset.masker.n_elements_ != self.n_voxel:
                    raise ValueError("Groups may have different brain masks")
            focus_filter = nimare.diagnostics.FocusFilter(mask=self.mask)
            group_dset = focus_filter.transform(group_dset)
            self.all_dset[group] = group_dset
                
    def load_data(self, centres = [(25,25,25), (65, 65, 65)], cov=5*np.eye(3), n_study=1000, r=42):
        # X
        X_path = os.getcwd() + "/X/X_{}_spline_degree_{}_shift_{}.npy".format(self.spacing, self.spline_degree, self.half_interval_shift)
        print(X_path)
        if os.path.exists(X_path):
            print("exists")
            self.X = np.load(X_path)
        else:
            self.X = self.B_spline_bases(self.spacing)
            np.save(X_path, self.X)
        self.X = torch.tensor(self.X, dtype=torch.float64, device=self.device)
        # intensity_func
        intensity_func_path = os.getcwd() + "/intensity_func_{}.npy".format(centres)
        if os.path.exists(intensity_func_path):
            intensity_func = np.load(intensity_func_path) # (228483, 1)
        else:
            print("re-generate")
            intensity_func = np.zeros(shape=self.mask.mask_img._dataobj.shape)
            x_dim, y_dim, z_dim = intensity_func.shape # 91, 109, 91
            x_range, y_range, z_range = range(x_dim), range(y_dim), range(z_dim)
            x_coords, y_coords, z_coords = np.meshgrid(x_range, y_range, z_range, indexing='ij')
            # Stack the coordinates in a single array
            coords = np.stack([x_coords.ravel(), y_coords.ravel(), z_coords.ravel()], axis=-1)
            # Adjust coordinates by 0.5 as needed for CDF calculations
            coords_1 = coords + 0.5
            coords_2 = coords + np.array([0.5, -0.5, -0.5])
            coords_3 = coords + np.array([-0.5, 0.5, -0.5])
            coords_4 = coords + np.array([-0.5, -0.5, 0.5])
            coords_5 = coords + np.array([0.5, 0.5, -0.5])
            coords_6 = coords + np.array([0.5, -0.5, 0.5])
            coords_7 = coords + np.array([-0.5, 0.5, 0.5])
            coords_8 = coords + np.array([-0.5, -0.5, -0.5])
            for centre in centres:
                # Vectorized computation of CDFs at eight corner adjustments per voxel
                # This combines all CDF calculations into a single vectorized operation
                gaussian_func = (multivariate_normal.cdf(coords_1, mean=centre, cov=cov) 
                                   + multivariate_normal.cdf(coords_2, mean=centre, cov=cov)
                                   + multivariate_normal.cdf(coords_3, mean=centre, cov=cov)
                                   + multivariate_normal.cdf(coords_4, mean=centre, cov=cov)
                                   - multivariate_normal.cdf(coords_5, mean=centre, cov=cov)
                                   - multivariate_normal.cdf(coords_6, mean=centre, cov=cov)
                                   - multivariate_normal.cdf(coords_7, mean=centre, cov=cov)
                                   - multivariate_normal.cdf(coords_8, mean=centre, cov=cov)).reshape((x_dim, y_dim, z_dim))
                intensity_func += gaussian_func
            intensity_func = intensity_func * self.foci_per_study / len(centres)
            # add a background signal
            intensity_func += 1e-6
            # remove voxels outside of brain mask
            intensity_func_img = nib.Nifti1Image(intensity_func, self.mask.mask_img.affine)
            intensity_func = self.mask.transform(intensity_func_img).T # (228383, 1)
            np.save(intensity_func_path, intensity_func)
        n_voxel, _ = self.X.shape
        y = np.zeros(shape=(n_voxel, 1))
        y_t = list()
        np.random.seed(r)
        for i in range(n_study):
            y_i = np.random.poisson(intensity_func)
            y += y_i
            y_t.append(np.sum(y_i))
        y_t = np.array(y_t)
        np.save("simulated_data/y.npy", y)
        np.save("simulated_data/y_t.npy", y_t)
        print(np.sum(y), np.sum(y_t))
        self.y = torch.tensor(y, dtype=torch.float64, device=self.device)
        # y_t
        self.y_t = torch.tensor(y_t, dtype=torch.float64, device=self.device)
           
    def coef_spline_bases(self, axis_coords, spacing, margin):
        ## create B-spline basis for x/y/z coordinate
        wider_axis_coords = np.arange(np.min(axis_coords) - margin, np.max(axis_coords) + margin)
        knots = np.arange(np.min(axis_coords) - margin, np.max(axis_coords) + margin, step=spacing)
        if self.half_interval_shift:
            knots = knots + 0.5 * self.spacing
            knots = [e for e in knots if min(wider_axis_coords)<=e<=max(wider_axis_coords)]
        design_matrix = patsy.dmatrix(
            "bs(x, knots=knots, degree={},include_intercept=False)".format(self.spline_degree),
            data={"x": wider_axis_coords},
            return_type="matrix",
        )
        design_array = np.array(design_matrix)[:, 1:]  # remove the first column (every element is 1)
        coef_spline = design_array[margin : -margin + 1, :]
        # remove the basis with no/weakly support from the square
        supported_basis = np.sum(coef_spline, axis=0) != 0
        coef_spline = coef_spline[:, supported_basis]

        return coef_spline  
    
    def B_spline_bases(self, spacing, margin=10):
        masker_voxels = self.mask.mask_img._dataobj
        dim_mask = masker_voxels.shape
        # remove the blank space around the brain mask
        xx = np.where(np.apply_over_axes(np.sum, masker_voxels, [1, 2]) > 0)[0]
        yy = np.where(np.apply_over_axes(np.sum, masker_voxels, [0, 2]) > 0)[1]
        zz = np.where(np.apply_over_axes(np.sum, masker_voxels, [0, 1]) > 0)[2]

        x_spline = self.coef_spline_bases(xx, spacing, margin)
        y_spline = self.coef_spline_bases(yy, spacing, margin)
        z_spline = self.coef_spline_bases(zz, spacing, margin)
        x_spline_coords = x_spline.nonzero()
        y_spline_coords = y_spline.nonzero()
        z_spline_coords = z_spline.nonzero()
        x_spline_sparse = sparse.COO(x_spline_coords, x_spline[x_spline_coords])
        y_spline_sparse = sparse.COO(y_spline_coords, y_spline[y_spline_coords])
        z_spline_sparse = sparse.COO(z_spline_coords, z_spline[z_spline_coords])

        # create spatial design matrix by tensor product of spline bases in 3 dimesion
        X = np.kron(np.kron(x_spline_sparse, y_spline_sparse), z_spline_sparse)  # Row sums of X are all 1=> There is no need to re-normalise X
        
        # remove the voxels outside brain mask
        axis_dim = [xx.shape[0], yy.shape[0], zz.shape[0]]
        brain_voxels_index = [(z - np.min(zz))+ axis_dim[2] * (y - np.min(yy))+ axis_dim[1] * axis_dim[2] * (x - np.min(xx))
                            for x in xx for y in yy for z in zz if masker_voxels[x, y, z] == 1]
        X = X[brain_voxels_index, :].todense()
        # remove tensor product basis that have no support in the brain
        x_df, y_df, z_df = x_spline.shape[1], y_spline.shape[1], z_spline.shape[1]
        support_basis = []
        # find and remove weakly supported B-spline bases
        for bx in range(x_df):
            for by in range(y_df):
                for bz in range(z_df):
                    basis_index = bz + z_df*by + z_df*y_df*bx
                    basis_coef = X[:, basis_index]
                    if np.max(basis_coef) >= 0.1: 
                        support_basis.append(basis_index)
        X = X[:, support_basis]
        print(X.shape)
        return X
      
    def model_structure(self, model):
        beta_dim = self.X.shape[1]
        gamma_dim = 2 # always with study-level covariates
        n_study = self.y_t.shape[0]
        ## model type
        if model == 'Poisson':
            print(self.device)
            model = GLMPoisson(beta_dim=beta_dim, gamma_dim=gamma_dim, covariates=self.covariates, penalty=self.penalty, device=self.device)
        elif model == 'NB':
            # # load beta & gamma from optimized Poisson distribution
            # Poisson_path = os.getcwd() + '/results/' + self.dataset + '/Poisson_model/' + self.penalty + '_penalty/'
            # beta = np.load(Poisson_path+'beta.npy')
            # gamma = np.load(Poisson_path+'gamma.npy')
            model = GLMNB(beta_dim=beta_dim, gamma_dim=gamma_dim, covariates=self.covariates, penalty=self.penalty, n_study=n_study)
        elif model == 'Clustered_NB':
            model = GLMClustered_NB(beta_dim=beta_dim, gamma_dim=gamma_dim, covariates=self.covariates, penalty=self.penalty)
        elif model == 'ZIP':
            model = GLMZIP(beta_dim=beta_dim, gamma_dim=gamma_dim, covariates=self.covariates, penalty=self.penalty)
        elif model == 'Quasi_Poisson':
            model = GLMQPOI(beta_dim=beta_dim, gamma_dim=gamma_dim, covariates=self.covariates, penalty=self.penalty)
        if 'cuda' in self.device:
            model = model.cuda()
        return model
    
    def _optimizer(self, model, y, Z, y_t, lr, tol, iter, r):
        # optimization 
        optimizer = torch.optim.LBFGS(params=model.parameters(), lr=lr,
                                      max_iter=iter, tolerance_change=tol)
        def closure():
            optimizer.zero_grad()
            loss = model(self.X, y, None, y_t)
            loss.backward()
            return loss
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        optimizer.step(closure)
        scheduler.step()
        loss = model(self.X, y, None, y_t)
        
        return loss

    def mkdir(self):
        if self.covariates == True:
            PATH_1 = os.getcwd() + '/results/with_covariate_results/' + self.dataset + '/'
        else:
            PATH_1 = os.getcwd() + '/results/no_covariate_results/' + self.dataset + '/'
        # make directory if it doesn't exist
        if os.path.isdir(PATH_1) == False:
            os.mkdir(PATH_1)
        PATH_2 = PATH_1 + self.model + '_model/'
        if os.path.isdir(PATH_2) == False:
            os.mkdir(PATH_2)
        PATH_3 = PATH_2 + self.penalty + '_penalty/'
        if os.path.isdir(PATH_3) == False:
            os.mkdir(PATH_3)
        return PATH_3

    def _train(self, iter=1500, lr=1, tol=1e-4, r=42):
        print('r='+str(r))
        ## model & optimization process
        model = self.model_structure(self.model)
        if self.model == "Poisson":
            neg_l = self._optimizer(model, self.y, None, self.y_t, lr, tol, iter, r)
            return neg_l, model.beta_linear.weight
        if self.model == 'NB':
            neg_l = self._optimizer(model, y, y_t, Y, lr, tol, iter, r)
            return neg_l, model.beta_linear.weight, model.estimated_alpha
        return 

    def outcome(self, n_experiment=100, iter=1000, lr=0.1, tol=1e-2):
        self.n_experiment = n_experiment
        y_array, y_t_array = [], []
        # file path
        if self.spline_degree == 3:
            file0 = 'Cubic_spline/'
        elif self.spline_degree == 2:
            file0 = 'Quadratic_spline/'
        if self.half_interval_shift:
            file1 = 'half_interval_shift/'
        else:
            file1 = 'no_interval_shift/'
        folder_path = 'results/' + file0 + file1 + str(self.model) + '_model/'
        filename = self.model + '_spacing_' + str(self.spacing)
        # create a diretory if it doesn't exist
        if os.path.exists(folder_path) == False:
            os.makedirs(folder_path)
        if os.path.isfile(folder_path+'beta_'+filename+'.npy'):
            beta_output = np.load(folder_path+'beta_'+filename+'.npy')
            neg_l_output = np.load(folder_path+'neg_l_spacing_{}.npy'.format(self.spacing))
            n_completed_realization = beta_output.shape[1]
            if self.model == 'NB':
                alpha_output = np.load(folder_path+'alpha_'+filename+'.npy')
        else:
            n_completed_realization = 0
            beta_dim = self.X.shape[1]
            beta_output = np.empty(shape=(beta_dim, 0))
            neg_l_output = np.empty(shape=(0,))
            if self.model == 'NB':
                alpha_output = np.empty(shape=(0,))
        for i in range(n_experiment):
            if n_completed_realization < n_experiment:
                i = n_completed_realization # random seed
                if self.model == 'Poisson':
                    neg_l, beta = self._train(iter=iter, lr=lr, tol=tol, r=i)
                if self.model == 'NB':
                    neg_l, beta, alpha = self._train(iter=iter, lr=lr, tol=tol, r=i)
                beta = beta.detach().cpu().numpy().T
                beta_output = np.concatenate((beta_output, beta), axis=1)
                neg_l = neg_l.detach().cpu().numpy().reshape((1,))
                
                neg_l_output = np.concatenate((neg_l_output, neg_l), axis=0)
                np.save(folder_path+'beta_'+filename+'.npy', beta_output)
                np.save(folder_path+'neg_l_spacing_{}.npy'.format(self.spacing), neg_l_output)
                if self.model == 'NB':
                    alpha = alpha.detach().cpu().numpy().reshape((1,))
                    alpha_output = np.concatenate((alpha_output, alpha))
                    np.save(folder_path+'alpha_'+filename+'.npy', alpha_output)
                print('file saved')
                n_completed_realization += 1
            else:
                break
        
        return
 