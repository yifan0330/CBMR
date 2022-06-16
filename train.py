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


class train_dataset(object):
    def __init__(self, dataset, spacing, device='cpu'):
        self.device = device
        self.dataset = dataset
        self.spacing = spacing
        ## load design matrix X, foci counts y and study-level covariates Z
        PATH = os.getcwd() + '/data/' + self.dataset + '/'
        X, Z = np.load(PATH+'X.npy'), np.load(PATH+'Z.npy')
        y, y_t = np.load(PATH+'y.npy'), np.load(PATH+'y_t.npy')
        # y_p, Y = np.load(PATH+'y_p.npy'), np.load(PATH+'Y.npy')
        self.X = torch.tensor(X, dtype=torch.float64, device=self.device)
        self.y = torch.tensor(y, dtype=torch.float64, device=self.device)
        # self.Y = torch.tensor(Y, dtype=torch.float64, device=self.device)
        self.y_t = torch.tensor(y_t, dtype=torch.float64, device=self.device)
        # self.y_p = torch.tensor(y_p, dtype=torch.float64, device=self.device)
        self.Z = torch.tensor(Z, dtype=torch.float64, device=self.device)
        
    
    def model_structure(self, model, penalty, covariates):
        beta_dim = self.X.shape[1]
        gamma_dim = self.Z.shape[1] # always with study-level covariates
        ## model type
        if model == 'Poisson':
            model = GLMPoisson(beta_dim=beta_dim, gamma_dim=gamma_dim, covariates=self.covariates, penalty=self.penalty)
        elif model == 'NB':
            # # load beta & gamma from optimized Poisson distribution
            # Poisson_path = os.getcwd() + '/results/' + self.dataset + '/Poisson_model/' + self.penalty + '_penalty/'
            # beta = np.load(Poisson_path+'beta.npy')
            # gamma = np.load(Poisson_path+'gamma.npy')
            model = GLMNB(beta_dim=beta_dim, gamma_dim=gamma_dim, covariates=self.covariates, penalty=self.penalty)
        elif model == 'Clustered_NB':
            model = GLMClustered_NB(beta_dim=beta_dim, gamma_dim=gamma_dim, covariates=self.covariates, penalty=self.penalty)
        elif model == 'ZIP':
            model = GLMZIP(beta_dim=beta_dim, gamma_dim=gamma_dim, covariates=self.covariates, penalty=self.penalty)
        elif model == 'Quasi_Poisson':
            model = GLMQPOI(beta_dim=beta_dim, gamma_dim=gamma_dim, covariates=self.covariates, penalty=self.penalty)
        if 'cuda' in self.device:
            model = model.cuda()
        return model
    
    def _optimizer(self, model, y, Z, y_t, penalty, lr, tol, iter):
        # optimization 
        optimizer = torch.optim.LBFGS(model.parameters(), lr)
        prev_loss = torch.tensor(float('inf'))
        loss_diff = torch.tensor(float('inf'))
        step = 0
        while torch.abs(loss_diff) > tol: 
            if step <= iter:
                def closure():
                    optimizer.zero_grad()
                    loss = model(self.X, y, Z, y_t)
                    loss.backward()
                    return loss
                loss = optimizer.step(closure)
                # reset L_BFGS if NAN appears
                if torch.any(torch.isnan(model.beta_linear.weight)):
                    print("Reset lbfgs optimiser ......")
                    model.beta_linear.weight = torch.nn.Parameter(last_state['beta_linear.weight'])
                    model.gamma_linear.weight = torch.nn.Parameter(last_state['gamma_linear.weight'])
                    if self.model == 'NB':
                        model.theta = torch.nn.Parameter(last_state['theta'])
                    if self.model == 'Clustered_NB':
                        model.alpha = torch.nn.Parameter(last_state['alpha'])
                    loss_diff = torch.tensor(float('inf'))
                    optimizer = torch.optim.LBFGS(model.parameters(), lr)
                    continue
                else:
                    last_state = copy.deepcopy(model.state_dict())
                print("step {0}: loss {1}".format(step, loss))
                loss_diff = loss - prev_loss
                prev_loss = loss
                step = step + 1
            else:
                print('it did not converge \n')
                print('The difference of loss in the current and previous iteration is', loss_diff)
                exit()
        return 

    def mkdir(self):
        PATH_1 = os.getcwd() + '/results/' + self.dataset + '/'
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

    def train(self, model, penalty, covariates, iter=1500, lr=0.01, tol=1e-4):
        self.model = model
        self.penalty = penalty
        self.covariates = covariates
        # model & optimization process
        model = self.model_structure(model=self.model, penalty=self.penalty, covariates=self.covariates)
        optimization = self._optimizer(model=model, y=self.y, Z=self.Z, y_t=self.y_t, penalty=self.penalty, lr=lr, tol=tol, iter=iter)
        # beta
        beta = model.beta_linear.weight
        beta = beta.detach().cpu().numpy().T
        # gamma
        gamma = model.gamma_linear.weight
        gamma = gamma.detach().cpu().numpy().T
        # save to file
        PATH = self.mkdir()
        np.save(PATH+'beta.npy', beta)
        np.save(PATH+'gamma.npy', gamma)
        if self.model == 'NB':
            # alpha
            alpha = model.alpha
            sum_alpha = model.estimated_alpha
            alpha, sum_alpha = alpha.detach().cpu().numpy(), sum_alpha.detach().cpu().numpy()
            print(alpha, sum_alpha)
            np.save(PATH+'alpha.npy', alpha)
            np.save(PATH+'sum_alpha.npy', sum_alpha)

        elif self.model == 'Clustered_NB':
            # alpha
            alpha = model.alpha
            alpha = alpha.detach().cpu().numpy()
            np.save(PATH+'alpha.npy', alpha)
        elif self.model == 'ZIP':
            # psi
            psi = model.psi
            psi = psi.detach().cpu().numpy()
            np.save(PATH+'psi.npy', psi)
        elif self.model == 'Quasi_Poisson':
            # theta
            x = model.x
            theta = torch.sigmoid(x)
            print(theta)
            theta = theta.detach().cpu().numpy()
            np.save(PATH+'theta.npy', theta)
        return 

class inference(object):
    def __init__(self, dataset, spacing, penalty, covariates, device='cpu'):
        self.dataset = dataset
        self.spacing = spacing
        self.penalty = penalty
        self.covariates = covariates
        self.device = device
        self.models = ['Poisson', 'NB', 'Clustered_NB']
        ## load design matrix X, foci counts y and study-level covariates Z
        data_PATH = os.getcwd() + '/data/' + self.dataset + '/'
        self.X = np.load(data_PATH + 'X.npy')
        self.y = np.load(data_PATH + 'y.npy')
        self.y_t = np.load(data_PATH + 'y_t.npy')
        self.Z = np.load(data_PATH + 'Z.npy')
        # load index of voxels outside brain
        outside_brain_filename = os.getcwd() + '/data/' + self.dataset + '/outside_brain.npy'
        self.outside_brain = np.load(outside_brain_filename)
        # load the min/max value of xx/yy/zz coord inside brain mask
        coord_min_max_filename = os.getcwd() + '/data/' + self.dataset + '/coord_min_max.npy'
        x_min, x_max, y_min, y_max, z_min, z_max = np.load(coord_min_max_filename)
        self.begin_voxel = [x_min, y_min, z_min]
        self.image_dim = [x_max-x_min+1, y_max-y_min+1, z_max-z_min+1]
        
        # # load optimized parameters
        # # model: Poisson
        # Poisson_PATH = os.getcwd() + '/results/' + self.dataset + '/Poisson_model/' + str(self.penalty) + '_penalty/'
        # self.Poi_beta = np.load(Poisson_PATH + 'beta.npy')
        # self.Poi_gamma = np.load(Poisson_PATH + 'gamma.npy')
        # # model: NB
        # NB_PATH = os.getcwd() + '/results/' + self.dataset + '/NB_model/' + str(self.penalty) + '_penalty/'
        # self.NB_alpha = np.load(NB_PATH + 'alpha.npy')
        # self.NB_beta = np.load(NB_PATH + 'beta.npy')
        # self.NB_gamma = np.load(NB_PATH + 'gamma.npy')
        # # model: Clustered NB
        # CNB_PATH = os.getcwd() + '/results/' + self.dataset + '/Clustered_NB_model/' + str(self.penalty) + '_penalty/'
        # self.CNB_alpha = np.load(CNB_PATH + 'alpha.npy')
        # self.CNB_beta = np.load(CNB_PATH + 'beta.npy')
        # self.CNB_gamma = np.load(CNB_PATH + 'gamma.npy')

    def intensity_prediction(self, X, beta, gamma):
        # mu^X = exp(X * beta)
        log_mu_X = np.matmul(self.X, beta)
        mu_X = np.exp(log_mu_X)
        # mu^Z = exp(Z * gamma)
        log_mu_Z = np.matmul(self.Z, gamma)
        mu_Z = np.exp(log_mu_Z)
        # voxelwise intensity function per group
        # mu_g = mu^X_g [sum_i mu^Z_i]
        mu = np.sum(mu_Z) * mu_X
        
        return mu
    
    def mkdir(self, model):
        PATH_1 = os.getcwd() + '/results/' + self.dataset + '/'
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
            PATH = os.getcwd() + '/results/' + self.dataset + '/' + model + '_model/' + str(self.penalty) + '_penalty/'
            beta = np.load(PATH + 'beta.npy')
            gamma = np.load(PATH + 'gamma.npy')
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
        for model in self.models:
            PATH = os.getcwd() + '/results/' + self.dataset + '/' + model + '_model/' + str(self.penalty) + '_penalty/'
            beta = np.load(PATH + 'beta.npy')
            beta = torch.tensor(beta, dtype=torch.float64, device='cuda')
            gamma = np.load(PATH + 'gamma.npy')
            gamma = torch.tensor(gamma, dtype=torch.float64, device='cuda')
            if model == 'Poisson':
                l = GLMPoisson._log_likelihood(beta, gamma, self.X, self.y, self.Z, self.y_t)
                l = l.cpu().detach().numpy()
                k = beta.shape[0] + gamma.shape[0]
            elif model == 'NB':
                sum_alpha = np.load(PATH + 'sum_alpha.npy')
                sum_alpha = torch.tensor(sum_alpha, dtype=torch.float32, device='cuda')
                l = GLMNB._log_likelihood(sum_alpha, beta, gamma, self.X, self.y, self.Z, self.y_t)
                l = l.cpu().detach().numpy()
                k = 1 + beta.shape[0] + gamma.shape[0]
            elif model == 'Clustered_NB':
                alpha = np.load(PATH + 'alpha.npy')
                alpha = torch.tensor(alpha, dtype=torch.float32, device='cuda')
                l = GLMClustered_NB._log_likelihood(alpha, beta, gamma, self.X, self.y, self.Z, self.y_t)
                l = l.cpu().detach().numpy()
                k = 1 + beta.shape[0] + gamma.shape[0]
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
        file_folder = os.getcwd() + '/results/' + self.dataset 
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
        self.likelihood_models = ['Poisson_model', 'NB_model', 'Clustered_NB_model']
        self.all_models = ['Poisson_model', 'NB_model', 'Clustered_NB_model', 'Quasi_Poisson_model']
        self.dataset = ['1_Social_Processing', '2_PTSD', '3_Substance_Use', '4_Dementia', '5_Cue_Reactivity', '6_Emotion_Regulation', 
                        '7_Decision_Making', '8_Reward', '9_Sleep_Deprivation', '10_Naturalistic', '11_Problem_Solving', '12_Emotion', 
                        '13_Cannabis_Use', '14_Nicotine_Use', '15_Frontal_Pole_CBP', '16_Face_Perception', '17_Nicotine_Administration', 
                        '18_Executive_Function', '19_Finger_Tapping', '20_n-Back']
        return
    
    def likelihood_comparison(self):
        n_datasets = len(self.dataset)
        l_inference, AIC_inference, BIC_inference = [], [], []
        for dataset in self.dataset:
            fold_path = os.getcwd() + '/results/' + dataset + '/'
            l = np.load(fold_path + 'log_likelihood.npy')
            AIC = np.load(fold_path + 'AIC.npy')
            BIC = np.load(fold_path + 'BIC.npy')
            l_inference.append(l)
            AIC_inference.append(AIC)
            BIC_inference.append(BIC)
        l_inference = np.array(l_inference)
        # sort log_likelihood of all models & split it into 4 subplots
        sort_dataset_index = np.argsort(l_inference[:, 0])
        n_subplots_row = 4 # number of subplots per row
        sort_dataset_index = sort_dataset_index.reshape((n_subplots_row, -1))
        n_dataset_per_subplot = sort_dataset_index.shape[1]
        AIC_inference, BIC_inference = np.array(AIC_inference), np.array(BIC_inference)
        model_inference = [l_inference, AIC_inference, BIC_inference]
        n_criteria = len(model_inference)
        # make boxplots for each GoF
        sns.set_context("paper")
        fig = plt.figure(figsize=(30, 20))
        # fig, ax = plt.subplots(n_criteria, n_subplots_row, sharex='col', sharey='row', figsize=(30, 20))
        plt.subplots_adjust(hspace = 0.1) # adjust the space between subplots
        x = np.repeat(np.array([[1, 2, 3]]), n_dataset_per_subplot, axis=0)
        palette = plt.cm.rainbow(np.linspace(0, 1, n_datasets)) # plt.get_cmap(['Set1', 'Set2', 'Set3']) # color palette
        
        models_xticks = ['Poisson', 'NB', 'Clustered_NB']
        k = 0
        for i in range(n_criteria): # model selection criteria: l / AIC / BIC
            color = 0
            for j in range(n_subplots_row):
                dataset_index = sort_dataset_index[j, :]
                dataset_name = [self.dataset[d] for d in dataset_index]
                dataset_inference = model_inference[i][dataset_index]
                ax = fig.add_subplot(n_criteria, n_subplots_row, k+1)
                ax = plt.plot(x.T, dataset_inference.T, linestyle='-', marker='o', linewidth=2, markersize=2.5, alpha=0.9, label=color)
                ax = plt.legend(dataset_name, ncol=1)
                ax = plt.xticks([1,2,3], models_xticks, fontsize=10)
                if k % n_subplots_row == 0: 
                    ax = plt.ylabel(self.likelihood_criteria[i], size=20)
                color += 1
                k += 1
        fig.savefig(os.getcwd() + '/figures/log_likelihood_plot.jpg')
        
        return

    def intensity_prediction(self, X, Z, beta, gamma):
        # mu^X = exp(X * beta)
        log_mu_X = np.matmul(X, beta)
        mu_X = np.exp(log_mu_X)
        # mu^Z = exp(Z * gamma)
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
            # estimated n_foci per study
            est_intensity_per_study = []
            for model in self.all_models:
                path = os.getcwd() + '/data/' + dataset 
                X = np.load(path + '/X.npy')
                Z = np.load(path + '/Z.npy')
                PATH = os.getcwd() + '/results/' + dataset + '/' + model + '/' + str(self.penalty) + '_penalty/'
                beta = np.load(PATH + 'beta.npy')
                gamma = np.load(PATH + 'gamma.npy')
                mu_t = self.intensity_prediction(X, Z, beta, gamma)[1]
                sum_mu_per_study = np.mean(mu_t)
                est_intensity_per_study.append(sum_mu_per_study)
            est_intensity_per_study = np.array(est_intensity_per_study)
            # acutual n_foci per study
            Y_PATH = PATH = os.getcwd() + '/data/' + dataset + '/Y.npy'
            Y = np.load(Y_PATH)
            n_foci = np.sum(Y)
            M, N = Y.shape # n_study, n_voxel
            foci_per_study = n_foci / M
            diff_percent = (est_intensity_per_study - foci_per_study) / foci_per_study
            print(diff_percent)
            diff_percent = diff_percent.reshape((-1, n_models))
            all_diff = np.concatenate((all_diff, diff_percent), axis=0)
        np.save(os.getcwd()+'/figures/study_foci_diff.npy', all_diff)
        
        return all_diff

    def FociSum_boxplot(self):
        diff = np.load(os.getcwd()+'/figures/study_foci_diff.npy')
        # make boxplots for each GoF
        fig, ax = plt.subplots(figsize=(10, 10))
        # x = np.array([[1, 2, 3]])
        bplot = ax.boxplot(diff, vert=True, patch_artist=True, labels=self.all_models, showfliers=False)
        vals = ax.get_yticks()
        ax.set_yticklabels(['{:,.2%}'.format(x) for x in vals])
        ax.set_ylabel('Relative Bias(estimated total sum of intensity (per study))', fontsize=12)
        ax.yaxis.grid(True)
        # fill with colors
        colors = ['pink', 'lightblue', 'lightgreen', 'orange']
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        fig.savefig(os.getcwd() + '/figures/foci_sum_plot.jpg')
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
                PATH = os.getcwd() + '/results/' + dataset + '/' + model + '/' + str(self.penalty) + '_penalty/'
                beta = np.load(PATH + 'beta.npy')
                gamma = np.load(PATH + 'gamma.npy')
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
        np.save(os.getcwd()+'/figures/x_Std.npy', all_x_Std)
        np.save(os.getcwd()+'/figures/y_Std.npy', all_y_Std)
        np.save(os.getcwd()+'/figures/z_Std.npy', all_z_Std)
        
        return
    
    def Std_boxplot(self):
        x_Std = np.load(os.getcwd()+'/figures/x_Std.npy')
        y_Std = np.load(os.getcwd()+'/figures/y_Std.npy')
        z_Std = np.load(os.getcwd()+'/figures/z_Std.npy')
        Std_list = [x_Std, y_Std, z_Std]
        n_datasets = len(self.dataset)
        # make boxplots for each GoF
        fig, ax = plt.subplots(3, 1, sharex='col', sharey='row', figsize=(20, 20))
        x = np.array([1, 2, 3, 4])
        y_label = ['Relative bias of Std(x)', 'Relative bias of Std(y)', 'Relative bias of Std(z)']
        colors = ['bisque', 'lemonchiffon', 'mediumseagreen', 'lightcyan']
        for i in range(len(y_label)): 
            for j in range(n_datasets):
                bplot = ax[i].boxplot(Std_list[i], vert=True, patch_artist=True, labels=['Poisson model', 'NB model', 'Clustered NB model', 'Quasi-Poisson model'], showfliers=False)
                # convert y value to percentage
                label_format = '{:,.2%}'
                ticks_loc = ax[i].get_yticks().tolist()
                ax[i].yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
                ax[i].set_yticklabels([label_format.format(x) for x in ticks_loc])
                for patch, color in zip(bplot['boxes'], colors):
                    patch.set_facecolor(color)
            # ax[i].plot(np.repeat(x, n_datasets, axis=0), model_inference[i], 'r.', alpha=0.5)
            ax[i].set_ylabel(y_label[i], fontsize=20)
            ax[i].yaxis.grid(True)
            # ax[i].legend(self.dataset, ncol=3)
        fig.savefig(os.getcwd() + '/figures/Std_plot.jpg')

        return 

    def variance_comparison(self):
        n_models = len(self.all_models)
        all_bias_var = np.empty(shape=(0, n_models))
        for dataset in self.dataset:
            print(dataset)
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
                PATH = os.getcwd() + '/results/' + dataset + '/' + model + '/' + str(self.penalty) + '_penalty/'
                beta = np.load(PATH + 'beta.npy')
                gamma = np.load(PATH + 'gamma.npy')
                # mu, mu_t = self.intensity_prediction(X, Z, beta, gamma)
                # mu_per_study = mu[nonzero_index, :] / M   
                mu_per_study = y[nonzero_index, :] / M  
                if model == 'Poisson_model':
                    estimation_var = mu_per_study
                elif model == 'NB_model':
                    alpha = np.load(PATH + 'alpha.npy')
                    estimation_var = mu_per_study + alpha*mu_per_study**2 
                elif model == 'Clustered_NB_model':
                    alpha = np.load(PATH + 'alpha.npy')
                    estimation_var = mu_per_study + alpha*mu_per_study**2 
                elif model == 'Quasi_Poisson_model':
                    theta = np.load(PATH + 'theta.npy')
                    estimation_var = mu_per_study + 1/theta * mu_per_study**2 
                bias_var = np.mean((estimation_var - sample_var) / sample_var)
                print(bias_var)
                bias_var_list.append(np.mean(bias_var))
            bias_var_array = np.array(bias_var_list)
            bias_var_array = bias_var_array.reshape((1, n_models))
            all_bias_var = np.concatenate((all_bias_var, bias_var_array), axis=0)
        # save to files
        np.save(os.getcwd()+'/figures/bias_var.npy', all_bias_var)
        
        return 
    
    def var_boxplot(self):
        var_bias = np.load(os.getcwd()+'/figures/bias_var.npy')
        # make boxplots for bias(variance) between sample & estimation data
        fig, ax = plt.subplots(figsize=(10, 10))
        bplot = ax.boxplot(var_bias, vert=True, patch_artist=True, labels=self.all_models, showfliers=False)
        ax.set_ylabel('Average bias of voxel-wise variance between foci and intensity estimation', fontsize=12)
        ax.yaxis.grid(True)
        # fill with colors
        colors = ['pink', 'lightblue', 'lightgreen', 'orange']
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        fig.savefig(os.getcwd() + '/figures/var_bias_plot.jpg')

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
    
