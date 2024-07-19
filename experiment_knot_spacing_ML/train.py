import torch
import os
import numpy as np
import scipy
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
    def __init__(self, dataset, model, covariates, penalty, spline_degree, half_interval_shift, spacing_list, device='cpu'):
        self.device = device
        self.dataset = dataset
        self.model = model
        self.covariates = covariates
        self.penalty = penalty
        self.spline_degree = spline_degree
        self.half_interval_shift = half_interval_shift
        self.spacing_list = spacing_list
        self.device = device 
        # load dataset from sleuth file
        self.load_dataset()
        self.load_data()
        
    def load_dataset(self):
        cwd = os.getcwd() 
        dset_PATH = os.path.dirname(cwd) + '/data/' + self.dataset + '/'
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
                
    def load_data(self):
        # y
        y = np.zeros(shape=self.mask.mask_img._dataobj.shape)
        y_t = np.empty(shape=(0,1))
        for group in self.group_name:
            xyz_coords = self.all_dset[group].coordinates[['x','y','z']].to_numpy()
            ijk_coords = nimare.utils.mm2vox(xyz_coords, self.mask.affine_)
            for ijk in ijk_coords:
                y[ijk[0], ijk[1], ijk[2]] += 1
            y_t_group = self.all_dset[group].coordinates['id'].value_counts(sort=False).to_numpy()
            y_t = np.concatenate((y_t, y_t_group.reshape((-1,1))), axis=0)
        y_img = nib.Nifti1Image(y, self.mask.mask_img.affine)
        y = self.mask.transform(y_img).T
        # np.save("y/y.npy", y)
        self.y = torch.tensor(y, dtype=torch.float64, device=self.device)
        # y_t
        self.y_t = torch.tensor(y_t, dtype=torch.float64, device=self.device)
        self.n_study = y_t.shape[0]
    
    def coef_spline_bases(self, axis_coords, spacing, margin):
        ## create B-spline basis for x/y/z coordinate
        wider_axis_coords = np.arange(np.min(axis_coords) - margin, np.max(axis_coords) + margin)
        knots = np.arange(np.min(axis_coords) - margin, np.max(axis_coords) + margin, step=spacing)
        design_matrix = patsy.dmatrix(
            "bs(x, knots=knots, degree=3,include_intercept=False)",
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
        
        return X
      
    def model_structure(self, model, beta_dim):
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
    
    def _optimizer(self, model, X, y, Z, y_t, lr, tol, iter):
        # optimization 
        optimizer = torch.optim.LBFGS(params=model.parameters(), lr=lr,
                                      max_iter=iter, tolerance_change=tol)
        def closure():
            optimizer.zero_grad()
            loss = model(X, y, None, y_t)
            loss.backward()
            return loss
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        optimizer.step(closure)
        scheduler.step()
        loss = model(X, y, None, y_t)
        
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

    def _train(self, spacing, iter=1500, lr=1, tol=1e-4):
        ## model & optimization process
        # X
        X = self.B_spline_bases(spacing)
        np.save("X/X_{}.npy".format(spacing), X)
        X = torch.tensor(X, dtype=torch.float64, device=self.device)
        beta_dim = X.shape[1]
        print(beta_dim)
        model = self.model_structure(self.model, beta_dim)
        if self.model == "Poisson":
            neg_l = self._optimizer(model, X, self.y, None, self.y_t, lr, tol, iter)
            return neg_l, model.beta_linear.weight
        if self.model == 'NB':
            neg_l = self._optimizer(model, X, self.y, None, self.y_t, lr, tol, iter)
            return neg_l, model.beta_linear.weight, model.estimated_alpha
        return 

    def outcome(self, n_experiment=100, iter=1000, lr=0.1, tol=1e-2):
        self.n_experiment = n_experiment
        y_array, y_t_array = [], []
        # file path
        folder_path = 'results/' + str(self.model) + '_model/' + self.dataset + '/'
        # create a diretory if it doesn't exist
        if os.path.exists(folder_path) == False:
            os.makedirs(folder_path)
        neg_l_all = np.empty(shape=(0,))
        for spacing in self.spacing_list:
            neg_l, beta = self._train(spacing=spacing, iter=iter, lr=lr, tol=tol)     
            neg_l, beta = neg_l.detach().cpu().numpy().reshape((1,)), beta.detach().cpu().numpy().T
            np.save(folder_path+'beta_spacing'+ str(spacing) + '_' + self.dataset +'.npy', beta)
            neg_l_all = np.concatenate((neg_l_all, neg_l), axis=0)
        np.save(folder_path+'neg_l_'+self.dataset+'.npy', neg_l_all)
        
        return
