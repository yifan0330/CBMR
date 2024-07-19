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
    def __init__(self, dataset, model, covariates, penalty, spacing, device='cpu'):
        self.device = device
        self.dataset = dataset
        self.model = model
        self.covariates = covariates
        self.penalty = penalty
        self.spacing = spacing
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
        # X
        self.X = self.B_spline_bases(self.spacing)
        self.X = torch.tensor(self.X, dtype=torch.float64, device=self.device)
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
        print(tol)
        start_time = time.time()
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
        print(time.time()-start_time)
        
        return 

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

    def train(self, iter=1500, lr=1, tol=1e-4, r=42):
        print('r='+str(r))
        ## model & optimization process
        print(self.model)
        model = self.model_structure(self.model)
        if self.model == "Poisson":
            optimization = self._optimizer(model, self.y, None, self.y_t, lr, tol, iter, r)
            return model.beta_linear.weight
        if self.model == 'NB':
            optimization = self._optimizer(model, y, y_t, Y, lr, tol, iter, r)
            return model.beta_linear.weight, model.estimated_alpha
        return 

class inference(object):
    def __init__(self, dataset, spacing, penalty, covariates, device='cpu'):
        self.dataset = dataset
        self.spacing = spacing
        self.penalty = penalty
        self.covariates = covariates
        self.device = device
        self.models = ['Poisson', 'Clustered_NB', 'NB']
        ## load design matrix X, foci counts y and study-level covariates Z
        data_PATH = os.getcwd() + '/data/' + self.dataset + '/'
        self.X = np.load(data_PATH + 'X.npy')
        self.y = np.load(data_PATH + 'y.npy')
        self.y_t = np.load(data_PATH + 'y_t.npy')
        self.n_study = self.y_t.shape[0]
        self.Z = np.load(data_PATH + 'Z.npy')
        # load index of voxels outside brain
        outside_brain_filename = os.getcwd() + '/data/' + self.dataset + '/outside_brain.npy'
        self.outside_brain = np.load(outside_brain_filename)
        # load the min/max value of xx/yy/zz coord inside brain mask
        coord_min_max_filename = os.getcwd() + '/data/' + self.dataset + '/coord_min_max.npy'
        x_min, x_max, y_min, y_max, z_min, z_max = np.load(coord_min_max_filename)
        self.begin_voxel = [x_min, y_min, z_min]
        self.image_dim = [x_max-x_min+1, y_max-y_min+1, z_max-z_min+1]

    def intensity_prediction(self, X, beta, gamma):
        # mu^X = exp(X * beta)
        log_mu_X = np.matmul(self.X, beta)
        mu_X = np.exp(log_mu_X)
        # mu^Z = exp(Z * gamma)
        if self.covariates == True:
            log_mu_Z = np.matmul(self.Z, gamma)
            mu_Z = np.exp(log_mu_Z)
            # voxelwise intensity function per group
            # mu = mu^X [sum_i mu^Z_i]
            mu = np.sum(mu_Z) * mu_X
        else:
            mu = self.n_study * mu_X
        
        return mu
    
    def mkdir(self, model):
        if self.covariates == True:
            PATH_1 = os.getcwd() + '/results/with_covariate_results/' + self.dataset + '/'
        else:
            PATH_1 = os.getcwd() + '/results/no_covariate_results/' + self.dataset + '/'
        # make directory if it doesn't exist
        if os.path.isdir(PATH_1) == False:
            os.mkdir(PATH_1)
        PATH_2 = PATH_1 + model + '_model/'
        if os.path.isdir(PATH_2) == False:
            os.mkdir(PATH_2)
        PATH_3 = PATH_2 + self.penalty + '_penalty/'
        if os.path.isdir(PATH_3) == False:
            os.mkdir(PATH_3)
        return PATH_3

    def convert2image(self, model, beta, gamma, mask='brain'):
        # convert the predicted responses into nifti image 
        pred_mu = self.intensity_prediction(self.X, beta, gamma)
        # brain mask
        self.brain_mask = nimare.utils.get_template(space='mni152_2mm', mask=mask).dataobj 
        brain_mask_affine = nimare.utils.get_template(space='mni152_2mm', mask=mask).affine
        self.dim_mask = self.brain_mask.shape # (91, 109, 91)
        ## outside_brain voxel index
        brain_voxel_index = np.setdiff1d(np.arange(np.prod(self.image_dim)), self.outside_brain) # (228453, )
        output_image = np.zeros(shape=self.dim_mask)

        # conver to x/y/z coords
        for i in range(brain_voxel_index.shape[0]):
            index = brain_voxel_index[i]
            x_coord = index // (self.image_dim[1]*self.image_dim[2]) + self.begin_voxel[0]
            remainder = index % (self.image_dim[1]*self.image_dim[2])
            y_coord = remainder // self.image_dim[2] + self.begin_voxel[1]
            remainder = remainder % self.image_dim[2]
            z_coord = remainder + self.begin_voxel[2]
            response = pred_mu[i]
            output_image[x_coord, y_coord, z_coord] = response
        
        image = nib.Nifti1Image(output_image, brain_mask_affine)
        PATH = self.mkdir(model)
        image_name = PATH + self.dataset + '_' + model + '.nii.gz'
        # Save as NiBabel file
        image.to_filename(image_name)  

        return
    
    def pred_image_all_models(self, mask='brain'):
        for model in self.models:
            if self.covariates == True:
                PATH_0 = 'with_covariate_results/'
            else:
                PATH_0 = 'no_covariate_results/'
            PATH = os.getcwd() + '/results/' + PATH_0 + self.dataset + '/' + model + '_model/' + str(self.penalty) + '_penalty/'
            beta = np.load(PATH + 'beta.npy')
            if self.covariates == True:
                gamma = np.load(PATH + 'gamma.npy')
            else:
                gamma = None
            self.convert2image(model=model, beta=beta, gamma=gamma, mask=mask)
        
        return 

    def model_selection_criteria(self):
        # number of data points
        n = self.X.shape[0]
        self.X = torch.tensor(self.X, dtype=torch.float64, device='cuda')
        self.y = torch.tensor(self.y, dtype=torch.float64, device='cuda')
        self.Z = torch.tensor(self.Z, dtype=torch.float64, device='cuda')
        self.y_t = torch.tensor(self.y_t, dtype=torch.float64, device='cuda')
        all_l, all_AIC, all_BIC = [], [], []
        if self.covariates == True:
            PATH_0 = 'with_covariate_results/'
        else:
            PATH_0 = 'no_covariate_results/'
        for model in self.models:
            PATH = os.getcwd() + '/results/' + PATH_0 + self.dataset + '/' + model + '_model/' + str(self.penalty) + '_penalty/'
            beta = np.load(PATH + 'beta.npy')
            beta = torch.tensor(beta, dtype=torch.float64, device='cuda')
            if self.covariates == True:
                gamma = np.load(PATH + 'gamma.npy')
                gamma = torch.tensor(gamma, dtype=torch.float64, device='cuda')
            else:
                gamma = None
            if model == 'Poisson':
                l = GLMPoisson._log_likelihood(beta, gamma, self.X, self.y, self.Z, self.y_t)
                l = l.cpu().detach().numpy()
                if self.covariates == True:
                    k = beta.shape[0] + gamma.shape[0]
                else:
                    k = beta.shape[0]
            elif model == 'Clustered_NB':
                alpha = np.load(PATH + 'alpha.npy')
                alpha = torch.tensor(alpha, dtype=torch.float64, device='cuda')
                l = GLMClustered_NB._log_likelihood(alpha, beta, gamma, self.X, self.y, self.Z, self.y_t)
                l = l.cpu().detach().numpy()
                if self.covariates == True:
                    k = 1 + beta.shape[0] + gamma.shape[0]
                else:
                    k = 1 + beta.shape[0]
            elif model == 'NB':
                sum_alpha = np.load(PATH + 'sum_alpha.npy')
                sum_alpha = torch.tensor(sum_alpha, dtype=torch.float64, device='cuda')
                l = GLMNB._log_likelihood(sum_alpha, beta, gamma, self.X, self.y, self.Z, self.y_t)
                l = l.cpu().detach().numpy()
                if self.covariates == True:
                    k = 1 + beta.shape[0] + gamma.shape[0]
                else:
                    k = 1 + beta.shape[0]
            # log_likelihood
            all_l.append(l)
            # AIC
            AIC = 2*k - 2*l
            all_AIC.append(AIC)
            # BIC
            BIC = k*np.log(n) - 2*l
            all_BIC.append(BIC)
        all_l = np.array(all_l)
        print(all_l)
        all_AIC, all_BIC = np.array(all_AIC), np.array(all_BIC)
        # save to file
        file_folder = os.getcwd() + '/results/' + PATH_0 + self.dataset 
        np.save(file_folder+'/log_likelihood.npy', all_l, allow_pickle=True)
        np.save(file_folder+'/AIC.npy', all_AIC, allow_pickle=True)
        np.save(file_folder+'/BIC.npy', all_BIC, allow_pickle=True)
        
        return

class model_comparison(object):
    def __init__(self, spacing, penalty, covariates, device='cpu'):
        self.likelihood_criteria = ['log_likelihood', 'AIC', 'BIC']
        self.spacing = spacing
        self.penalty = penalty
        self.covariates = covariates
        self.likelihood_models = ['Poisson_model', 'Clustered_NB_model', 'NB_model']
        self.all_models = ['Poisson_model', 'Clustered_NB_model', 'NB_model', 'Quasi_Poisson_model']
        self.dataset = ['1_Social_Processing', '2_PTSD', '3_Substance_Use', '4_Dementia', '5_Cue_Reactivity', '6_Emotion_Regulation', 
                        '7_Decision_Making', '8_Reward', '9_Sleep_Deprivation', '10_Naturalistic', '11_Problem_Solving', '12_Emotion',
                        '13_Cannabis_Use', '14_Nicotine_Use', '15_Frontal_Pole_CBP', '16_Face_Perception', '17_Nicotine_Administration', 
                        '18_Executive_Function', '19_Finger_Tapping', '20_n-Back']
        # Path for files
        if self.covariates == True:
            self.PATH_0 = 'with_covariate_results/'
            self.PATH_fig_0 = 'with_covariate_figures/'
        else:
            self.PATH_0 = 'no_covariate_results/'
            self.PATH_fig_0 = 'no_covariate_figures/'
    
    def likelihood_comparison(self):
        n_datasets = len(self.dataset)
        l_inference, AIC_inference, BIC_inference = [], [], []
        for dataset in self.dataset:
            fold_path = os.getcwd() + '/results/' + self.PATH_0 + dataset + '/'
            l = np.load(fold_path + 'log_likelihood.npy')
            AIC = np.load(fold_path + 'AIC.npy')
            BIC = np.load(fold_path + 'BIC.npy')
            l_inference.append(l)
            AIC_inference.append(AIC)
            BIC_inference.append(BIC)
        l_inference = np.array(l_inference)
        print(l_inference)
        n_datasets = l_inference.shape[0]
        AIC_inference, BIC_inference = np.array(AIC_inference), np.array(BIC_inference)
        model_inference = [l_inference, AIC_inference, BIC_inference]
        n_criteria = len(model_inference)
        # make boxplots for each GoF
        sns.set_context("paper")
        x = np.array([0,1,2])
        models_xticks = ['Poi', 'Clu_NB', 'NB']
        fig = plt.figure(figsize=(30, 20))
        plt.subplots_adjust(wspace=0.35, hspace = 0.2) # adjust the space between subplots
        k = 0
        n_subplots_row = np.int(n_datasets/2)
        for i in range(n_criteria):
            for j in range(n_datasets):
                dataset_inference = model_inference[i][j, :]
                ax = fig.add_subplot(2*n_criteria, n_subplots_row, k+1)
                sns.barplot(x=x, y=dataset_inference, palette='vlag')
                ax.set_title(self.dataset[j])
                ax.get_xaxis().set_visible(False)
                # set y_label only for the first plot on the left
                if j%10==0:
                    ax.set_ylabel(self.likelihood_criteria[i])
                plt.xticks(x, models_xticks, fontsize=10)
                # only display x-axis label for plots on the last row
                if i==2 and j//10==1:
                    ax.get_xaxis().set_visible(True)
                k += 1
        print(os.getcwd() + '/figures/' + self.PATH_fig_0 + 'log_likelihood_plot.pdf')
        fig.savefig(os.getcwd() + '/figures/' + self.PATH_fig_0 + 'log_likelihood_plot.pdf')
        print("done")

    def intensity_prediction(self, X, Z, beta, gamma):
        # mu^X = exp(X * beta)
        log_mu_X = np.matmul(X, beta)
        mu_X = np.exp(log_mu_X)
        # mu^Z = exp(Z * gamma)
        if gamma is None:
            mu = self.n_study * mu_X
            mu_t = np.repeat(a=np.sum(mu_X), repeats=self.n_study)
        else:
            log_mu_Z = np.matmul(Z, gamma)
            mu_Z = np.exp(log_mu_Z)
            # voxelwise intensity function per group
            # mu_t = mu^Z_i * [sum_i mu^X_g]
            mu = np.sum(mu_Z) * mu_X
            mu_t = np.sum(mu_X) * mu_Z
    
        return mu, mu_t

    def foci_per_study(self):
        n_models = len(self.all_models)
        all_diff = np.empty(shape=(0, n_models))
        for dataset in self.dataset:
            # acutual n_foci per study
            Y_PATH = PATH = os.getcwd() + '/data/' + dataset + '/Y.npy'
            Y = np.load(Y_PATH)
            n_foci = np.sum(Y)
            M, N = Y.shape # n_study, n_voxel
            foci_per_study = n_foci / M
            self.n_study = Y.shape[0]
            # estimated n_foci per study
            est_intensity_per_study = []
            for model in self.all_models:
                path = os.getcwd() + '/data/' + dataset 
                X = np.load(path + '/X.npy')
                Z = np.load(path + '/Z.npy')
                PATH = os.getcwd() + '/results/' + self.PATH_0 + dataset + '/' + model + '/' + str(self.penalty) + '_penalty/'
                beta = np.load(PATH + 'beta.npy')
                if self.covariates == True:
                    gamma = np.load(PATH + 'gamma.npy')
                else:
                    gamma = None
                mu_t = self.intensity_prediction(X, Z, beta, gamma)[1]
                sum_mu_per_study = np.mean(mu_t)
                est_intensity_per_study.append(sum_mu_per_study)
            est_intensity_per_study = np.array(est_intensity_per_study)
            
            diff_percent = (est_intensity_per_study - foci_per_study) / foci_per_study
            diff_percent = diff_percent.reshape((-1, n_models))
            all_diff = np.concatenate((all_diff, diff_percent), axis=0)
        print(all_diff)
        np.save(os.getcwd()+'/figures/'+ self.PATH_fig_0 + 'study_foci_diff.npy', all_diff)
        
        return all_diff

    def FociSum_boxplot(self):
        diff = np.load(os.getcwd()+'/figures/'+self.PATH_fig_0+'study_foci_diff.npy')
        diff = [diff[:,i] for i in range(4)]
        # make boxplots for each GoF
        sns.set_style("darkgrid")
        fig, ax = plt.subplots(figsize=(10, 10))
        # x = np.array([[1, 2, 3]])
        # bplot = ax.boxplot(diff, vert=True, patch_artist=True, labels=self.all_models, showfliers=False)
        ax = sns.boxplot(data=diff, palette="Set3", showfliers = False)
        ax.set_xticklabels(['Poisson', 'Clustered NB', 'NB', 'Quasi Poisson'])
        vals = ax.get_yticks()
        ax.set_yticklabels(['{:,.2%}'.format(x) for x in vals])
        ax.set_ylabel('Relative Bias(estimated total sum of intensity (per study))', fontsize=12)
        ax.yaxis.grid(True)
        # fill with colors
        # colors = ['pink', 'lightblue', 'lightgreen']
        # for patch, color in zip(bplot['boxes'], colors):
        #     patch.set_facecolor(color)
        fig.savefig(os.getcwd() + '/figures/'+self.PATH_fig_0+'foci_sum_plot.pdf')
        print(os.getcwd() + '/figures/'+self.PATH_fig_0+'foci_sum_plot.pdf')
        print('done')

        return 

    def convert2coords(self, s, dataset, mask='brain'):
        # brain mask
        brain_mask = nimare.utils.get_template(space='mni152_2mm', mask=mask).dataobj 
        brain_mask_affine = nimare.utils.get_template(space='mni152_2mm', mask=mask).affine
        dim_mask = brain_mask.shape # (91, 109, 91)
        # load index of voxels outside brain
        outside_brain_filename = os.getcwd() + '/data/' + dataset + '/outside_brain.npy'
        outside_brain = np.load(outside_brain_filename)
        # load the min/max value of xx/yy/zz coord inside brain mask
        coord_min_max_filename = os.getcwd() + '/data/' + dataset + '/coord_min_max.npy'
        x_min, x_max, y_min, y_max, z_min, z_max = np.load(coord_min_max_filename)
        begin_voxel = [x_min, y_min, z_min]
        image_dim = [x_max-x_min+1, y_max-y_min+1, z_max-z_min+1] # [72, 90, 76]
        ## outside_brain voxel index
        brain_voxel_index = np.setdiff1d(np.arange(np.prod(image_dim)), outside_brain) # (228453, )
        output_image = np.zeros(shape=dim_mask)
        # conver to x/y/z coords
        for i in range(brain_voxel_index.shape[0]):
            index = brain_voxel_index[i]
            x_coord = index // (image_dim[1]*image_dim[2]) + begin_voxel[0]
            remainder = index % (image_dim[1]*image_dim[2])
            y_coord = remainder // image_dim[2] + begin_voxel[1]
            remainder = remainder % image_dim[2]
            z_coord = remainder + begin_voxel[2]
            response = s[i]
            output_image[x_coord, y_coord, z_coord] = response

        return output_image

    def expectation(self, S):
        a, b, c = S.shape
        xx, yy, zz = np.meshgrid(np.arange(a), np.arange(b), np.arange(c), indexing='ij')
        # shape: (a,b,c,3) each element is the voxel-wise coords
        coords_array = np.stack((xx,yy,zz), axis=-1)
        S = S.reshape((a,b,c,-1))
        # E[(x,y,z)^T] = sum_x sum_y sum_z [x,y,z]^T lambda(x,y,z)
        element_wise_prod = S * coords_array
        expectation = 1/(a*b*c) * np.sum(element_wise_prod, axis=(0,1,2))
        # expectation = expectation.reshape((-1,1))
        
        return expectation

    def covariance(self, S, E):
        a, b, c = S.shape
        xx, yy, zz = np.meshgrid(np.arange(a)-E[0], np.arange(b)-E[1], np.arange(c)-E[2], indexing='ij')
        # shape: (a,b,c,3) each element is the voxel-wise coords
        first_row = np.stack((xx*xx, xx*yy,xx*zz), axis=-1)
        second_row = np.stack((yy*xx, yy*yy,yy*zz), axis=-1)
        third_row = np.stack((zz*xx, zz*yy,zz*zz), axis=-1)
        # Cov[(x,y,z)^T] = E[g] = E[((x,y,z)^T-E[(x,y,z)^T]) * ((x,y,z)^T - E[(x,y,z)^T])^T]
        g = np.stack((first_row, second_row, third_row), axis=-1)
        element_wise_prod = S.reshape((a,b,c,1,1)) * g
        Cov = np.sum(1/(a*b*c)*element_wise_prod, axis=(0,1,2))
        
        return Cov

    def Cov_structure(self):
        n_models = len(self.all_models)
        all_x_Std, all_y_Std, all_z_Std = np.empty(shape=(0, n_models)), np.empty(shape=(0, n_models)), np.empty(shape=(0, n_models))
        for dataset in self.dataset:
            y_t = np.load(os.getcwd()+'/data/'+dataset +'/y_t.npy')
            self.n_study = y_t.shape[0]
            M = y_t.shape[0] # number of studies
            # load foci 
            y_PATH = os.getcwd() + '/data/' + dataset + '/y.npy'
            y = np.load(y_PATH)
            avg_foci = self.convert2coords(s=y/np.sum(y), dataset=dataset, mask='brain') # shape:(91, 109, 91) 
                                                                                         # x_i,y_i,z_i --> average voxel-wise foci intensity (per study)
            # E[(x,y,z)^T] = sum_x sum_y sum_z [x,y,z]^T lambda(x,y,z)
            E_avg_foci = self.expectation(avg_foci) # shape: (,3)
            # Cov[(x,y,z)^T] = E[((x,y,z)^T-E[(x,y,z)^T]) * ((x,y,z)^T - E[(x,y,z)^T])^T]
            Cov_avg_foci = self.covariance(avg_foci, E_avg_foci)
            Std_avg_foci = np.sqrt(np.diag(Cov_avg_foci))
            Std_list = []
            # estimated intensity
            for model in self.all_models:
                path = os.getcwd() + '/data/' + dataset 
                X = np.load(path + '/X.npy')
                Z = np.load(path + '/Z.npy')
                PATH = os.getcwd() + '/results/' + self.PATH_0 + dataset + '/' + model + '/' + str(self.penalty) + '_penalty/'
                beta = np.load(PATH + 'beta.npy')
                if self.covariates == True:
                    gamma = np.load(PATH + 'gamma.npy')
                else:
                    gamma = None
                mu, mu_t = self.intensity_prediction(X, Z, beta, gamma)
                avg_mu = self.convert2coords(s=mu/np.sum(mu), dataset=dataset, mask='brain') # shape:(91, 109, 91)
                # expectation & Covariance
                E_avg_mu = self.expectation(avg_mu) # shape: (,3)
                Cov_avg_mu = self.covariance(avg_mu, E_avg_mu)
                Std_avg_mu = np.sqrt(np.diag(Cov_avg_mu))
                Std_diff = (Std_avg_mu - Std_avg_foci) / Std_avg_foci
                Std_list.append(Std_diff)
            Std_array = np.array(Std_list)
            x_Std, y_Std, z_Std = Std_array[:,0], Std_array[:,1], Std_array[:,2]
            x_Std, y_Std, z_Std = x_Std.reshape((1, n_models)), y_Std.reshape((1, n_models)), z_Std.reshape((1, n_models))
            all_x_Std = np.concatenate((all_x_Std, x_Std), axis=0)
            all_y_Std = np.concatenate((all_y_Std, y_Std), axis=0)
            all_z_Std = np.concatenate((all_z_Std, z_Std), axis=0)
        # save to files
        np.save(os.getcwd()+'/figures/'+self.PATH_fig_0+'x_Std.npy', all_x_Std)
        np.save(os.getcwd()+'/figures/'+self.PATH_fig_0+'y_Std.npy', all_y_Std)
        np.save(os.getcwd()+'/figures/'+self.PATH_fig_0+'z_Std.npy', all_z_Std)
        
        return
    
    def Std_boxplot(self):
        x_Std = np.load(os.getcwd()+'/figures/'+self.PATH_fig_0+'x_Std.npy')
        y_Std = np.load(os.getcwd()+'/figures/'+self.PATH_fig_0+'y_Std.npy')
        z_Std = np.load(os.getcwd()+'/figures/'+self.PATH_fig_0+'z_Std.npy')
        Std_list = [x_Std, y_Std, z_Std]
        n_datasets = len(self.dataset)
        # make boxplots for each GoF
        sns.set_style("darkgrid")
        fig, ax = plt.subplots(3, 1, sharex='col', sharey='row', figsize=(10, 10))
        x = np.array([1, 2, 3, 4])
        y_label = ['Relative bias of Std(x)', 'Relative bias of Std(y)', 'Relative bias of Std(z)']
        k = 0
        for i in range(len(y_label)): 
            for j in range(n_datasets):
                # bplot = ax[i].boxplot(Std_list[i], vert=True, patch_artist=True, labels=['Poisson_model', 'Clustered_NB_model', 'NB_model'], showfliers=False)
                sns.boxplot(data=Std_list[i], ax=ax[k], width=0.5, palette="Set3", showfliers = False)
                # convert y value to percentage
                label_format = '{:,.3%}'
                ticks_loc = ax[i].get_yticks().tolist()
                ax[i].set_xticklabels(['Poisson', 'Clustered NB', 'NB', 'Quasi Poisson'])
                ax[i].yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
                ax[i].set_yticklabels([label_format.format(x) for x in ticks_loc])
            k += 1
            ax[i].set_ylabel(y_label[i], fontsize=15)
            ax[i].yaxis.grid(True)
        # ax = ax.flatten() 
        # ax.set_xticklabels(['Poisson', 'Clustered NB', 'NB', 'Quasi Poisson'])
        fig.savefig(os.getcwd() + '/figures/'+self.PATH_fig_0+'Std_plot.pdf')
        print(os.getcwd() + '/figures/'+self.PATH_fig_0+'Std_plot.pdf')
        return 

    def variance_comparison(self):
        n_models = len(self.all_models)
        all_bias_var = np.empty(shape=(0, n_models))
        for dataset in self.dataset:
            path = os.getcwd() + '/data/' + dataset 
            X = np.load(path + '/X.npy')
            Z = np.load(path + '/Z.npy')
            M = Z.shape[0]
            y = np.load(path + '/y.npy')
            Y = np.load(path + '/Y.npy')
            nonzero_index = np.where(np.sum(Y, axis=0)!=0)[0]
            nonzero_Y = Y[:, nonzero_index]
            sample_var = np.var(nonzero_Y, axis=0) 
            sample_var = sample_var.reshape((-1, 1)) # shape: (n_nonzero_voxel, 1)
            bias_var_list = []
            for model in self.all_models:
                PATH = os.getcwd() + '/results/' + self.PATH_0 + dataset + '/' + model + '/' + str(self.penalty) + '_penalty/'
                beta = np.load(PATH + 'beta.npy')
                if self.covariates == True:
                    gamma = np.load(PATH + 'gamma.npy')
                else:
                    gamma = None
                # mu, mu_t = self.intensity_prediction(X, Z, beta, gamma)
                # mu_per_study = mu[nonzero_index, :] / M   
                mu_per_study = y[nonzero_index, :] / M 
                if model == 'Poisson_model':
                    estimation_var = mu_per_study
                elif model == 'Clustered_NB_model':
                    alpha = np.load(PATH + 'alpha.npy') 
                    estimation_var = mu_per_study + alpha*mu_per_study**2 
                elif model == 'NB_model':
                    alpha = np.load(PATH + 'sum_alpha.npy')
                    estimation_var = mu_per_study + alpha*mu_per_study**2 
                elif model == 'Quasi_Poisson_model':
                    theta = np.load(PATH + 'theta.npy')
                    estimation_var = mu_per_study / theta
                bias_var = (estimation_var - sample_var) / sample_var
                bias_var_list.append(np.mean(bias_var))
            bias_var_array = np.array(bias_var_list)
            bias_var_array = bias_var_array.reshape((1, n_models))
            all_bias_var = np.concatenate((all_bias_var, bias_var_array), axis=0)
        print(all_bias_var)
        # save to files
        np.save(os.getcwd()+'/figures/'+self.PATH_fig_0+'bias_var.npy', all_bias_var)
        
        return 
    
    def var_boxplot(self):
        var_bias = -np.load(os.getcwd()+'/figures/'+self.PATH_fig_0+'bias_var.npy')
        print(np.median(var_bias, axis=0))
        # make boxplots for bias(variance) between sample & estimation data
        sns.set_style("darkgrid")
        fig, ax = plt.subplots(figsize=(10, 10))
        ax = sns.boxplot(data=var_bias, width=0.5, palette="Set3", showfliers = False)
        # bplot = ax.boxplot(var_bias, vert=True, patch_artist=True, labels=self.all_models, showfliers=False)
        ax.set_xticklabels(['Poisson', 'Clustered NB', 'NB', 'Quasi Poisson'])
        vals = ax.get_yticks()
        ax.set_yticklabels(['{:,.2%}'.format(x) for x in vals])
        ax.set_ylabel('Relative bias of voxel-wise variance between intensity estimation and foci counts', fontsize=12)
        ax.yaxis.grid(True)
        # fill with colors
        # colors = ['pink', 'lightblue', 'lightgreen', 'orange']
        # for patch, color in zip(bplot['boxes'], colors):
        #     patch.set_facecolor(color)
        fig.savefig(os.getcwd() + '/figures/'+self.PATH_fig_0+'var_bias_plot.pdf')

        return

    def skip_diag_strided(self, A):
        m = A.shape[0]
        strided = np.lib.stride_tricks.as_strided
        s0,s1 = A.strides
        return strided(A.ravel()[1:], shape=(m-1,m), strides=(s0+s1,s1)).reshape(m,-1)

    def Index2Coords(self, brain_voxel_index, mask='brain'):
        start_time = time.time()
        # brain mask
        brain_mask = nimare.utils.get_template(space='mni152_2mm', mask=mask).dataobj 
        brain_mask_affine = nimare.utils.get_template(space='mni152_2mm', mask=mask).affine
        dim_mask = brain_mask.shape # (91, 109, 91)
        filename = os.getcwd() + '/data/1_Social_Processing/'
        ## files from data/1_Soical_Processing
        x_min, x_max, y_min, y_max, z_min, z_max = np.load(filename+'coord_min_max.npy')
        # load min/max in x/y/z direction
        begin_voxel = [x_min, y_min, z_min]
        image_dim = [x_max-x_min+1, y_max-y_min+1, z_max-z_min+1] # [72, 90, 76]
        # load index of voxels outside brain
        outside_brain = np.load(filename + 'outside_brain.npy') # shape: (264027, )
        ## outside_brain voxel index
        brain_voxel_index = np.setdiff1d(np.arange(np.prod(image_dim)), outside_brain) # (228453, )
        Index2Coords = np.empty(shape=(0,4))

        # conver to x/y/z coords
        for i in range(brain_voxel_index.shape[0]):
            index = brain_voxel_index[i]
            x_coord = index // (image_dim[1]*image_dim[2]) + begin_voxel[0]
            remainder = index % (image_dim[1]*image_dim[2])
            y_coord = remainder // image_dim[2] + begin_voxel[1]
            remainder = remainder % image_dim[2]
            z_coord = remainder + begin_voxel[2]
            index_coord = np.array([i, np.int(x_coord), y_coord, z_coord]).reshape((1,4))
            Index2Coords = np.concatenate((Index2Coords, index_coord), axis=0)
        
        np.save(os.getcwd()+'/figures/Index2Coords.npy', Index2Coords)
        return

    def L_minus_h(self, foci, intensity, h):
        n_foci = foci.shape[0]
        # pairwise distance betwween two random foci
        all_pairwise_distance = scipy.spatial.distance.cdist(foci, foci, metric='cityblock', p=1)
        indicator = all_pairwise_distance <= h
        indicator = indicator.astype(int)
        # ignore the cases of y_i == y_j
        np.fill_diagonal(a=indicator, val=0, wrap=True)       
        # estimated intensity at each foci
        foci_index = foci[:, 0]
        foci_intensity = intensity[foci_index]
        kron_intensity = np.kron(foci_intensity.T, foci_intensity)
        fraction = indicator / kron_intensity
        # K = 1/|B| * sum_y_i sum_y_j(j!=i) 1(|y_i-y_j|<=t) / lambda(y_i)*lambda(y_j)
        B_dim = np.array([20, 20, 20])
        K = 1 / np.prod(B_dim) * np.sum(fraction)
        # E[K(t)] = (2t)^3
        # => (E[K(t)]/8)^(1/3) = t
        L = (K/8)**(1/3) 
        L_minus_h = L - h
       
        return L

    def K_function(self, h):
        n_models = len(self.all_models)
        Index2Coords = np.load(os.getcwd()+'/figures/Index2Coords.npy').astype(int)
        all_dataset_L_val = np.empty(shape=(0, n_models))
        for dataset in self.dataset:
            y = np.load(os.getcwd() + '/data/' + dataset + '/' + 'y.npy')
            n_foci = y.shape[0]
            foci_occurance = np.empty(shape=(0,))
            for i in range(n_foci):
                repeat_time = y[i, :].item()
                if repeat_time > 0:
                    a = np.repeat([i],repeat_time) 
                    foci_occurance = np.concatenate((foci_occurance, a), axis=0)
            foci_occurance = foci_occurance.astype(dtype=int)
            foci = Index2Coords[foci_occurance]
            dataset_L = []
            for model in self.all_models:
                X = np.load(os.getcwd() + '/data/' + dataset  + '/X.npy')
                Z = np.load(os.getcwd() + '/data/' + dataset  + '/Z.npy')
                est_path = os.getcwd() + '/results/' + dataset + '/' + model + '/' + str(self.penalty) + '_penalty/'
                beta = np.load(est_path + 'beta.npy')
                gamma = np.load(est_path + 'gamma.npy')
                # mu^X = exp(X * beta)
                log_mu_X = np.matmul(X, beta)
                mu_X = np.exp(log_mu_X)
                # mu^Z = exp(Z * gamma)
                log_mu_Z = np.matmul(Z, gamma)
                mu_Z = np.exp(log_mu_Z)
                intensity_per_study = np.sum(mu_Z) * mu_X
                L_minus_h = self.L_minus_h(foci, intensity_per_study, h)
                dataset_L.append(L_minus_h)
            dataset_L = np.array(dataset_L).reshape((1, n_models))
            all_dataset_L_val = np.concatenate((all_dataset_L_val, dataset_L), axis=0)
            print(dataset_L)

        np.save(os.getcwd()+'/figures/dataset_L_vals.npy', all_dataset_L_val)

        return 
    
    def K_val_plot(self):
        n_datasets = len(self.dataset)
        # load K vals 
        K_dataset = np.load(os.getcwd()+'/figures/dataset_K_vals.npy')
        # make boxplots for each K_vals
        fig, ax = plt.subplots(figsize=(10, 10))
        bplot = ax.boxplot(K_dataset, vert=True, patch_artist=True, labels=self.all_models, showfliers=False)
        ax.set_ylabel('K function of spatially inhomogeneous point process', fontsize=12)
        ax.yaxis.grid(True)
        # fill with colors
        colors = ['pink', 'lightblue', 'lightgreen', 'orange']
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        fig.savefig(os.getcwd() + '/figures/K_val_plot.jpg')
        

        return
    
class FDR_control(object):
    def __init__(self, spacing, penalty, covariates, device='cpu'):
        self.likelihood_criteria = ['log_likelihood', 'AIC', 'BIC']
        self.spacing = spacing
