import torch
import time
import numpy as np
import os

class GLMPoisson(torch.nn.Module):
    def __init__(self, beta_dim=None, gamma_dim=None, covariates=False, penalty='No', device='cpu'):
        super().__init__()
        self.covariates = covariates
        self.penalty = penalty
        self.device = device
        # beta
        self.beta_dim = beta_dim
        self.beta_linear = torch.nn.Linear(self.beta_dim, 1, bias=False).double()
        # initialization for beta
        torch.nn.init.uniform_(self.beta_linear.weight, a=-0.01, b=0.01)
        # gamma 
        if self.covariates == True:
            self.gamma_dim = gamma_dim
            self.gamma_linear = torch.nn.Linear(self.gamma_dim, 1, bias=False).double()
            torch.nn.init.uniform_(self.gamma_linear.weight, a=-0.01, b=0.01)
        else:
             self.gamma_dim = None

    def Fisher_information(self, X, mu_X, Z=None, mu_Z=None):
        if self.covariates == False:
            mu_X_sqrt = torch.sqrt(mu_X)
            # print(torch.min(mu_X), torch.max(mu_X))
            X_star = mu_X_sqrt * X # X* = W^(1/2) X
            I = torch.mm(X_star.t(), X_star)       
        else:
            # mu = [sum_i mu_i^Z] * mu_g^X
            mu = torch.sum(mu_Z) * mu_X
            mu_sqrt = torch.sqrt(mu)
            # the total sum of intensity in study i regardless of foci location in the whole dataset
            # mu_t = [sum_j mu_j^X] mu^Z
            mu_t = torch.sum(mu_X) * mu_Z
            mu_t_sqrt = torch.sqrt(mu_t)
            # Fisher Information matrix
            # block matrix at top left: I(beta)
            X_star = mu_sqrt * X # X* = W^(1/2) X
            I_beta = torch.mm(X_star.t(), X_star)
            # block matrix of the cross term: I(beta, gamma)
            I_cross_term = torch.mm(torch.mm(X.t(), mu_X), torch.mm(mu_Z.t(), Z))
            # block matrix at bittom right: I(gamma)
            Z_star = mu_t_sqrt * Z # Z* = V^(1/2) Z
            I_gamma = torch.mm(Z_star.t(), Z_star) # ZVZ = (V^(1/2) Z)^T (V^(1/2) Z)
            # concatenate to the Fisher Information matrix
            I_top = torch.cat((I_beta, I_cross_term), axis=1) # shape: (P, P+R)
            I_bottom = torch.cat((I_cross_term.t(), I_gamma), axis=1) # shape(R, P+R)
            I = torch.cat((I_top, I_bottom), axis=0) # shape: (P+R, P+R)
        
        return I

    def _log_likelihood(beta, gamma, X, y, Z, y_t):
        n_study = y_t.shape[0]
        log_mu_X = X @ beta
        mu_X = torch.exp(log_mu_X)
        if gamma is None:
            log_mu_Z = torch.zeros(n_study, 1, device='cuda')
            mu_Z = torch.ones(n_study, 1, device='cuda')
        else:
            log_mu_Z = Z @ gamma
            mu_Z = torch.exp(log_mu_Z)
        # l = [Y_g]^T * log(mu^X) + [Y^t]^T * log(mu^Z) - [1^T mu_g^X]*[1^T mu_g^Z]
        log_l = torch.sum(torch.mul(y, log_mu_X)) + torch.sum(torch.mul(y_t, log_mu_Z)) - torch.sum(mu_X) * torch.sum(mu_Z) 
        
        return log_l

    def forward(self, X, y, Z=None, y_t=None):
        # mu^X = exp(X * beta)
        log_mu_X = self.beta_linear(X) 
        mu_X = torch.exp(log_mu_X)
        # n_study
        n_study = y_t.shape[0]
        if self.covariates == True:
            # mu^Z = exp(Z * gamma)
            log_mu_Z = self.gamma_linear(Z)
            mu_Z = torch.exp(log_mu_Z)
        else:
            log_mu_Z = torch.zeros(n_study, 1, device=self.device)
            mu_Z = torch.ones(n_study, 1, device=self.device)
        # Under the assumption that Y_ij is either 0 or 1
        # l = [Y_g]^T * log(mu^X) + [Y^t]^T * log(mu^Z) - [1^T mu_g^X]*[1^T mu_g^Z]
        log_l = torch.sum(torch.mul(y, log_mu_X)) + torch.sum(torch.mul(y_t, log_mu_Z)) - torch.sum(mu_X) * torch.sum(mu_Z) 
        if self.penalty == 'No':
            l = log_l
        elif self.penalty == 'Firth':
            I = self.Fisher_information(X, mu_X, Z, mu_Z)
            eig_vals = torch.linalg.eig(I)[0].real
            log_det_I = torch.sum(torch.log(eig_vals))
            l = log_l + 1/2 * log_det_I
            # start_time = time.time()
            # beta = self.beta_linear.weight.T
            # gamma = self.gamma_linear.weight.T
            # params = (beta, gamma)
            # # l = GLMPoisson._log_likelihood(beta, gamma, X, y, Z, y_t)
            # nll = lambda beta, gamma: -GLMPoisson._log_likelihood(beta, gamma, X, y, Z, y_t)
            # h = torch.autograd.functional.hessian(nll, params, create_graph=False)
            # n_params = len(h)
            # # approximate hessian matrix by its diagonal matrix
            # h_beta = h[0][0].view(self.beta_dim, -1)
            # h_gamma = h[1][1].view(self.gamma_dim, -1)
            # h_diagonal_beta, h_diagonal_gamma = torch.diagonal(h_beta, 0), torch.diagonal(h_gamma, 0)
            # # # Firth-type penalty
            # log_det_I = torch.sum(torch.log(h_diagonal_beta)) + torch.sum(torch.log(h_diagonal_gamma))
            # l = log_l + 1/2 * log_det_I
            # print(log_det_I)
        print(-l)
        return -l

class GLMNB(torch.nn.Module):
    def __init__(self, beta_dim=None, gamma_dim=None, covariates=False, penalty='No', n_study=None):
        super().__init__()
        self.covariates = covariates
        self.penalty = penalty
        self.n_study = n_study
        ## beta
        self.beta_dim = beta_dim
        self.beta_linear = torch.nn.Linear(self.beta_dim, 1, bias=False).double()
        ## initialization for beta
        torch.nn.init.uniform_(self.beta_linear.weight, a=-0.01, b=0.01)
        # initialization for alpha: uniform distribution between 0 and 1
        alpha_init = torch.tensor([0.1], dtype=torch.float64, device='cuda')
        theta_init = -torch.log(100*self.n_study/alpha_init - 1)
        self.theta = torch.nn.Parameter(theta_init, requires_grad=True).double()
        if self.covariates == True:
            self.gamma_dim = gamma_dim
            self.gamma_linear = torch.nn.Linear(self.gamma_dim, 1, bias=False).double()
            torch.nn.init.uniform_(self.gamma_linear.weight, a=-0.1, b=0.1)
        else:
             self.gamma_dim = None
    
    def kron(a, b):
        siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(b.shape[-2:]))
        res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
        # siz0 = res.shape[:-4]
        return res.reshape((res.shape[0], res.shape[1], res.shape[2]))

    def first_three_term(y, r):
        max_foci = np.int(torch.max(y).item())
        sum_three_term = 0
        for k in range(max_foci):
            foci_index = (y == k+1).nonzero()[:,0]
            r_j = r[foci_index]
            n_voxel = list(foci_index.shape)[0]
            y_j = torch.tensor([k+1]*n_voxel, dtype=torch.float64, device='cuda')
            y_j = y_j.reshape((n_voxel, 1))
            # y=0 => sum_three_term = 0
            sum_three_term += torch.sum(torch.lgamma(y_j+r_j) - torch.lgamma(y_j+1) - torch.lgamma(r_j))
        
        return sum_three_term


    def _log_likelihood(alpha, beta, gamma, X, y, Z, y_t):
        n_study = y_t.shape[0]
        n_voxel = X.shape[0]
        v = 1 / alpha
        # mu_X
        log_mu_X = X @ beta
        mu_X = torch.exp(log_mu_X)
        # set machine epsilon as the lower bound 
        epsilon = torch.finfo(torch.float64).eps
        mu_X = torch.clamp(mu_X, min=epsilon)
        # mu_Z
        if gamma is None:
            log_mu_Z = torch.zeros(n_study, 1, device='cuda')
            mu_Z = torch.ones(n_study, 1, device='cuda')
        else: 
            log_mu_Z = Z @ gamma
            mu_Z = torch.exp(log_mu_Z)
        ## moment matching NB distribution
        # sum_i mu_{ij}^2 = sum_i (mu^Z_i)^2 * (mu^X_j)^2 = (mu^X_j)^2 * sum_i (mu^Z_i)^2
        sum_of_squre = mu_X**2 * torch.sum(mu_Z**2)
        # (sum_i mu_{ij})^2 = (sum_i mu^Z_i * mu^X_j)^2 = (mu^X_j * sum_i mu^Z_i)^2
        square_of_sum = mu_X**2 * torch.sum(mu_Z)**2
        # sum_i mu_ij = sum_i mu^X_j * mu^Z_i = mu^X_j * (sum_i mu^Z_i)
        sum_mu_ij = mu_X * torch.sum(mu_Z)
        ## moment matching NB distribution
        p = sum_of_squre / (v*sum_mu_ij + sum_of_squre)
        r = v * square_of_sum / sum_of_squre
        l = GLMNB.first_three_term(y,r) + torch.sum(r*torch.log(1-p) + y*torch.log(p))
        
        return l

    def forward(self, X, y, Z=None, y_t=None):
        log_mu_X = self.beta_linear(X)
        mu_X = torch.exp(log_mu_X)
        if self.covariates == True:
            log_mu_Z = self.gamma_linear(Z)
            mu_Z = torch.exp(log_mu_Z)
        else:
            log_mu_Z = torch.zeros(self.n_study, 1, device='cuda')
            mu_Z = torch.ones(self.n_study, 1, device='cuda')
        # Now the sum of NB variates are no long NB distributed (since mu_ij != mu_i'j),
        # Therefore, we use moment matching approach,
        # create a new NB approximation to the mixture of NB distributions: 
        # alpha' = sum_i mu_{ij}^2 / (sum_i mu_{ij})^2 * alpha
        self.alpha = 100*self.n_study * torch.nn.Sigmoid()(self.theta) + 1e-8
        # print(self.alpha.item())
        # sum_i mu_{ij}^2 = sum_i (mu^Z_i)^2 * (mu^X_j)^2 = (mu^X_j)^2 * sum_i (mu^Z_i)^2
        numerator = mu_X**2 * torch.sum(mu_Z**2)
        # (sum_i mu_{ij})^2 = (sum_i mu^Z_i * mu^X_j)^2 = (mu^X_j * sum_i mu^Z_i)^2
        denominator = mu_X**2 * torch.sum(mu_Z)**2
        voxel_sum_alpha = self.alpha * numerator / denominator
        self.estimated_alpha = torch.mean(voxel_sum_alpha)
        v = 1 / self.estimated_alpha
        # sum_i mu_ij = sum_i mu^X_j * mu^Z_i = mu^X_j * (sum_i mu^Z_i)
        sum_mu_ij = mu_X * torch.sum(mu_Z)
        ## moment matching NB distribution
        p = numerator / (v*sum_mu_ij + numerator)
        r = v * denominator / numerator
        l = GLMNB.first_three_term(y,r) + torch.sum(r*torch.log(1-p) + y*torch.log(p))
        
        return -l
        

class GLMClustered_NB(torch.nn.Module):
    def __init__(self, beta_dim=None, gamma_dim=None, covariates=False, penalty='No'):
        super().__init__()
        self.covariates = covariates
        self.penalty = penalty
        ## beta
        self.beta_dim = beta_dim
        self.beta_linear = torch.nn.Linear(self.beta_dim, 1, bias=False).double()
        ## initialization for beta
        torch.nn.init.uniform_(self.beta_linear.weight, a=-0.01, b=0.01)
        # initialization for alpha: uniform distribution between 0 and 1
        alpha_init = torch.tensor(1e-2, dtype=torch.float64, device='cuda')
        self.alpha = torch.nn.Parameter(alpha_init, requires_grad=True)
        # initialization for gamma
        self.gamma_dim = gamma_dim
        self.gamma_linear = torch.nn.Linear(self.gamma_dim, 1, bias=False).double()
        torch.nn.init.uniform_(self.gamma_linear.weight, a=-0.1, b=0.1)

    def _log_likelihood(alpha, beta, gamma, X, y, Z, y_t):
        n_study = y_t.shape[0]
        v = 1 / alpha
        # mu^X = exp(X * beta)
        log_mu_X = X @ beta
        mu_X = torch.exp(log_mu_X)
        # mu^Z = exp(Z * gamma)
        if gamma is None:
            log_mu_Z = torch.zeros(n_study, 1, device='cuda')
            mu_Z = torch.ones(n_study, 1, device='cuda')
        else:
            log_mu_Z = Z @ gamma
            mu_Z = torch.exp(log_mu_Z)
        # mu_t_i = [sum_j mu^X_j] mu^Z_i
        mu_t = torch.sum(mu_X) * mu_Z
        # n_study & n_voxels
        M, N = y_t.shape[0], y.shape[0]
        log_l = M * v * torch.log(v) - M * torch.lgamma(v) + torch.sum(torch.lgamma(y_t + v)) - torch.sum((y_t + v) * torch.log(mu_t + v)) \
            + torch.sum(y * log_mu_X) + torch.sum(y_t * log_mu_Z)  
        
        return log_l

    def forward(self, X, y, Z, y_t):
        n_study = y_t.shape[0]
        v = 1 / self.alpha
        # mu^X = exp(X * beta)
        log_mu_X = self.beta_linear(X) 
        mu_X = torch.exp(log_mu_X)
        # mu^Z = exp(Z * gamma)
        if self.covariates == True:
            log_mu_Z = self.gamma_linear(Z)
            mu_Z = torch.exp(log_mu_Z)
        else:
            log_mu_Z = torch.zeros(n_study, 1, device='cuda')
            mu_Z = torch.ones(n_study, 1, device='cuda')
        # mu_t = [sum_j mu^X_j] mu^Z
        mu_t = torch.sum(mu_X) * mu_Z
        # n_study & n_voxels
        M, N = y_t.shape[0], y.shape[0]
        log_l = M * v * torch.log(v) - M * torch.lgamma(v) + torch.sum(torch.lgamma(y_t + v)) - torch.sum((y_t + v) * torch.log(mu_t + v)) \
            + torch.sum(y * log_mu_X) + torch.sum(y_t * log_mu_Z)              
        print(mu_t)
        print(mu_t.shape)        
        exit()
        if self.penalty == 'No':
            l = log_l
        elif self.penalty == 'Firth':
            alpha = self.alpha
            beta = self.beta_linear.weight.T
            gamma = self.gamma_linear.weight.T
            params = (beta)
            nll = lambda beta: -GLMClustered_NB._log_likelihood(alpha, beta, gamma, X, y, Y, Z, y_t)
            h_beta = torch.autograd.functional.hessian(nll, params, create_graph=False)
            h_beta = h_beta.reshape((self.beta_dim, -1))
            # n_params = len(h)
            # # approximate hessian matrix by its diagonal matrix
            # h_alpha = h[0][0].view(1,1)
            # h_beta = h[1][1].view(self.beta_dim, -1)
            # h_gamma = h[2][2].view(self.gamma_dim, -1)
            # h_diagonal_beta, h_diagonal_gamma = torch.diagonal(h_beta, 0), torch.diagonal(h_gamma, 0)
            # # epsilon = torch.finfo(torch.float64).eps
            # # h_diagonal_alpha, h_diagonal_beta, h_diagonal_gamma = torch.clamp(torch.abs(h_diagonal_alpha), min=epsilon), torch.clamp(torch.abs(h_diagonal_beta), min=epsilon), torch.clamp(torch.abs(h_diagonal_gamma), min=epsilon)
            # # # Firth-type penalty
            # log_det_I = torch.sum(torch.log(h_alpha)) + torch.sum(torch.log(h_diagonal_beta)) + torch.sum(torch.log(h_diagonal_gamma))
            h_diagonal_beta = torch.diagonal(h_beta, 0)
            log_det_I = torch.sum(torch.log(h_diagonal_beta))
            l = log_l + 1/2 * log_det_I
            print(log_det_I)
        
        return -l

class GLMZIP(torch.nn.Module):
    def __init__(self, beta_dim=None, gamma_dim=None, covariates=False, penalty='No'):
        super().__init__()
        self.covariates = covariates
        self.penalty = penalty
        # beta
        self.beta_dim = beta_dim
        self.beta_linear = torch.nn.Linear(self.beta_dim, 1, bias=False)#.double()
        # initialization for beta 
        torch.nn.init.uniform_(self.beta_linear.weight, a=-0.01, b=0.01)
        # gamma 
        self.gamma_dim = gamma_dim
        self.gamma_linear = torch.nn.Linear(self.gamma_dim, 1, bias=False)#.double()
        torch.nn.init.uniform_(self.gamma_linear.weight, a=-0.01, b=0.01)
        # psi
        psi_init = torch.tensor(0.01, dtype=torch.float64, device='cuda')
        self.psi = torch.nn.Parameter(psi_init, requires_grad=True) 

    def setdiff(t1, t2):
        combined = torch.cat((t1, t2))
        uniques, counts = combined.unique(return_counts=True)
        diff = uniques[counts == 1]
        # intersection = uniques[counts > 1]
        return diff


    def forward(self, X, y, Z, y_t, y_p):
        N, P = X.shape
        # mu^X = exp(X * beta)
        log_mu_X = self.beta_linear(X) 
        mu_X = torch.exp(log_mu_X)
        # mu^Z = exp(Z * gamma)
        log_mu_Z = self.gamma_linear(Z)
        mu_Z = torch.exp(log_mu_Z)
        n_study = y_t.shape[0]
        l = 0
        for i in range(n_study):
            nonzero_voxel = y_p[y_p[:,0]==i][:,1]
            nonzero_voxel = nonzero_voxel.to(torch.long)
            nonzero_mu = mu_X[nonzero_voxel, :] * mu_Z[i]
            log_nonzero_mu = log_mu_X[nonzero_voxel, :] + log_mu_Z[i]
            # all nonzero foci counts are 1
            nonzero_term = torch.log(1-self.psi) - nonzero_mu + log_nonzero_mu
            zero_voxel = GLMZIP.setdiff(torch.arange(N,device='cuda'), nonzero_voxel)
            zero_mu = mu_X[zero_voxel, :] * mu_Z[i]
            zero_term = torch.log(self.psi + (1-self.psi)*torch.exp(-zero_mu))
            
            l += torch.sum(zero_term) + torch.sum(nonzero_term)
            
        return -l
    
    def _log_likelihood(psi, beta, gamma, X, y, Y, Z, y_t, y_p):
        N, P = X.shape
        # mu^X = exp(X * beta)
        log_mu_X = torch.matmul(X, beta)
        mu_X = torch.exp(log_mu_X)
        # # mu^Z = exp(Z * gamma)
        # log_mu_Z = self.gamma_linear(Z)
        # mu_Z = torch.exp(log_mu_Z)
        n_study = y_t.shape[0]
        for i in range(n_study):
            nonzero_voxel = y_p[y_p[:,0]==i][:,1].int().detach().cpu().numpy()
            nonzero_mu = mu_X[nonzero_voxel, :]
            nonzero_Y = Y[i, nonzero_voxel]
            nonzero_term = torch.sum(torch.log(1-psi) - nonzero_mu + nonzero_Y*torch.log(nonzero_mu) - torch.lgamma(nonzero_Y+1))
            zero_voxel = np.setdiff1d(np.arange(N), nonzero_voxel)
            zero_mu = mu_X[zero_voxel, :]
            zero_Y = Y[i, zero_voxel]
            zero_term = torch.sum(torch.log(psi + (1-psi)*torch.exp(-zero_mu)))
            # log likelihood
            l = zero_term + nonzero_term

        return l

class GLMQPOI(torch.nn.Module):
    def __init__(self, beta_dim=None, gamma_dim=None, covariates=False, penalty='No'):
        super().__init__()
        self.covariates = covariates
        self.penalty = penalty
        # beta
        self.beta_dim = beta_dim
        self.beta_linear = torch.nn.Linear(self.beta_dim, 1, bias=False)#.double()
        # initialization for beta 
        torch.nn.init.uniform_(self.beta_linear.weight, a=-0.01, b=0.01)
        # gamma 
        self.gamma_dim = gamma_dim
        self.gamma_linear = torch.nn.Linear(self.gamma_dim, 1, bias=False)#.double()
        torch.nn.init.uniform_(self.gamma_linear.weight, a=-0.01, b=0.01)
        # phi
        x_init = torch.tensor(0, dtype=torch.float64, device='cuda')
        self.x = torch.nn.Parameter(x_init, requires_grad=True) 

    def kron(self, a, b):
        siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(b.shape[-2:]))
        res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
        # siz0 = res.shape[:-4]
        return res.reshape((res.shape[0], res.shape[1], res.shape[2])) 

    def forward(self, X, y, Y, Z, y_t):
        theta = torch.sigmoid(self.x)
        M, N = Y.shape
        # mu^X = exp(X * beta)
        log_mu_X = self.beta_linear(X)
        mu_X = torch.exp(log_mu_X)
        # mu^Z = exp(Z * gamma)
        log_mu_Z = self.gamma_linear(Z)
        mu_Z = torch.exp(log_mu_Z)
        # Under the assumption that Y_ij is either 0 or 1
        # l = [Y_g]^T * log(mu^X) + [Y^t]^T * log(mu^Z) - [1^T mu_g^X]*[1^T mu_g^Z]
        l_poi = torch.sum(torch.mul(y, log_mu_X)) + torch.sum(torch.mul(y_t, log_mu_Z)) - torch.sum(mu_X) * torch.sum(mu_Z) 
        l_Y = torch.nansum(Y * torch.log(Y) - Y + torch.lgamma(Y+1))
        l = 1/2 * M  * torch.log(theta) + theta * l_poi + (1-theta)*l_Y
        # print(theta.item(), l_poi.item())
        
        # Y = Y.reshape((M, N, -1))
        # # mu^X = exp(X * beta)
        # log_mu_X = self.beta_linear(X)
        # mu_X = torch.exp(log_mu_X)
        # # mu^Z = exp(Z * gamma)
        # log_mu_Z = self.gamma_linear(Z)
        # mu_Z = torch.exp(log_mu_Z)
        # mu = self.kron(mu_Z, mu_X)
        # n_study = y_t.shape[0]
        # l = 0
        # for i in range(n_study):
        #     nonzero_voxel = y_p[y_p[:,0]==i][:,1].int().detach().cpu().numpy()
        #     nonzero_mu = mu[i, nonzero_voxel, :]
        #     nonzero_Y = Y[i, nonzero_voxel, :]
        #     ## theta*sum_i sum_j [y*log(mu)-mu+log(y!)]
        #     nonzero_term = self.theta*torch.sum(nonzero_Y*torch.log(nonzero_mu) - nonzero_mu) \
        #     + (1-self.theta)*torch.sum(nonzero_Y*torch.log(nonzero_Y) - nonzero_Y) - torch.sum(torch.lgamma(nonzero_Y+1))
        #     zero_voxel = np.setdiff1d(np.arange(N), nonzero_voxel)
        #     zero_mu = mu[i, zero_voxel, :]
        #     zero_Y = Y[i, zero_voxel, :]
        #     zero_term = self.theta*torch.sum(zero_Y*torch.log(zero_mu) - zero_mu - torch.lgamma(zero_Y+1))
        #     # log likelihood
        #     l += 1/2*N*torch.log(self.theta) + zero_term + nonzero_term
        
        return -l
    
    def _log_likelihood(X, y, y_t, y_p, Y, Z, beta, gamma, psi):
        N, P = X.shape
        # mu^X = exp(X * beta)
        log_mu_X = torch.matmul(X, beta)
        mu_X = torch.exp(log_mu_X)
        # # mu^Z = exp(Z * gamma)
        # log_mu_Z = self.gamma_linear(Z)
        # mu_Z = torch.exp(log_mu_Z)
        n_study = y_t.shape[0]
        for i in range(n_study):
            nonzero_voxel = y_p[y_p[:,0]==i][:,1].int().detach().cpu().numpy()
            nonzero_mu = mu_X[nonzero_voxel, :]
            nonzero_Y = Y[i, nonzero_voxel]
            nonzero_term = torch.sum(torch.log(1-psi) - nonzero_mu + nonzero_Y*torch.log(nonzero_mu) - torch.lgamma(nonzero_Y+1))
            zero_voxel = np.setdiff1d(np.arange(N), nonzero_voxel)
            zero_mu = mu_X[zero_voxel, :]
            zero_Y = Y[i, zero_voxel]
            zero_term = torch.sum(torch.log(psi + (1-psi)*torch.exp(-zero_mu)))
            # log likelihood
            l = zero_term + nonzero_term
           
        return l

