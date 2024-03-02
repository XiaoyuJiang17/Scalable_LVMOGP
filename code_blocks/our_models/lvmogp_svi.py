import sys
sys.path.append('/Users/jiangxiaoyu/Desktop/All Projects/Scalable_LVMOGP/')
import torch
from torch import Tensor
import numpy as np
from code_blocks.gplvm.latent_variables import VariationalLatentVariable, VariationalCatLatentVariable, NNEncoderLatentVariable
from code_blocks.gplvm.lvmogp_preparation import BayesianGPLVM_
from code_blocks.variational.cholesky_kronecker_variational_distribution import CholeskyKroneckerVariationalDistribution
from code_blocks.variational.kronecker_variational_strategy import KroneckerVariationalStrategy
from utils_general import helper_specify_kernel_by_name
from gpytorch.priors import NormalPrior
# from gpytorch.models.gplvm.latent_variable import *
from gpytorch.means import ZeroMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.distributions import MultivariateNormal
from linear_operator.operators import KroneckerProductLinearOperator


class LVMOGP_SVI(BayesianGPLVM_):

    def __init__(self, 
                 n_outputs, 
                 n_input,                    # NOTE PAY Strong ATTENTION, not total n_inputs, number of inputs for training!
                 input_dim, 
                 latent_dim,                 # This refers to the total dim: both trainable and non-trainable
                 n_inducing_input, 
                 n_inducing_latent, 
                 trainable_latent_dim = None, # how many dims are trainable (counting from the start), if none, all trainable
                 latent_first_init = None,    # trainable part initialization
                 latent_second_init = None,   # fixed part initialization
                 learn_inducing_locations_latent=True, 
                 learn_inducing_locations_input=True, 
                 latent_kernel_type='Scale_RBF', 
                 input_kernel_type='Scale_RBF',
                 NNEncoder=False,              # whether we want NNencoder ... 
                 layers=None):                 # to specify neural network layers used in NNencoder ...

        self.n_outputs = n_outputs
        self.n_input = n_input
        self.inducing_inputs_latent = torch.randn(n_inducing_latent, latent_dim)
        self.inducing_inputs_input = torch.randn(n_inducing_input, input_dim)
        
        q_u = CholeskyKroneckerVariationalDistribution(n_inducing_input, n_inducing_latent)

        q_f = KroneckerVariationalStrategy(self, self.inducing_inputs_latent, self.inducing_inputs_input, q_u, 
                                           learn_inducing_locations_latent=learn_inducing_locations_latent, 
                                           learn_inducing_locations_input=learn_inducing_locations_input)

        # Define prior for latent
        latent_prior_mean = torch.zeros(n_outputs, latent_dim)  # shape: N x Q
        prior_latent = NormalPrior(latent_prior_mean, torch.ones_like(latent_prior_mean))

        # Specify LatentVariable 
        if NNEncoder == False:
            # Initialization 
            latent_init = torch.randn(n_outputs, latent_dim)

            if latent_first_init != None and trainable_latent_dim != None:
                latent_init[:, :trainable_latent_dim] = latent_first_init

            # second part is fixed during training ... 
            if latent_second_init != None and trainable_latent_dim != None:
                if trainable_latent_dim < latent_dim:
                    latent_init[:, trainable_latent_dim:] = latent_second_init

            if trainable_latent_dim != None :
                latent_variables = VariationalCatLatentVariable(n_outputs, n_input, latent_dim, latent_init, prior_latent, trainable_latent_dim)
            else:
                latent_variables = VariationalLatentVariable(n_outputs, n_input, latent_dim, latent_init, prior_latent, trainable_latent_dim=None)

        elif NNEncoder == True:
            assert trainable_latent_dim == latent_first_init == latent_second_init == None
            if layers == None:
                layers = [4, 8, 4]
                # NotImplementedError('please specify layers in neural network encoder')
            
            latent_variables = NNEncoderLatentVariable(
                    n=n_outputs, 
                    data_dim=n_input, 
                    latent_dim=latent_dim, 
                    prior_x=prior_latent, 
                    latent_info_dim=2,  # NOTE: We currently fix this for spatio-temporal dataset.
                    layers=layers
                )

        super().__init__(latent_variables, q_f)
        self.mean_module = ZeroMean()

        # Kernel (acting on latent dimensions)
        # NOTE: Scale_RBF is the default choice, as prediction via integration of latent variable is possible.
        if latent_kernel_type == 'Scale_RBF':
            self.covar_module_latent = ScaleKernel(RBFKernel(ard_num_dims=latent_dim))

        # Kernel (acting on input dimensions)
        self.covar_module_input = helper_specify_kernel_by_name(input_kernel_type, input_dim)

    def _get_batch_idx(self, batch_size, sample_latent = True):
        if sample_latent == True:
            valid_indices = np.arange(self.n_outputs)
        else:
            valid_indices = np.arange(self.n_input)
        batch_indices = np.random.choice(valid_indices, size=batch_size, replace=False)
        
        return batch_indices


