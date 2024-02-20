import sys
sys.path.append('/Users/jiangxiaoyu/Desktop/All Projects/Scalable_LVMOGP/')
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.kernels import ScaleKernel, RBFKernel, PeriodicKernel
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from code_blocks.likelihoods.gaussian_likelihood import GaussianLikelihood
from utils_general import helper_specify_kernel_by_name
from torch import Tensor

class Variational_GP(ApproximateGP):
    def __init__(self, 
                 inducing_points, 
                 kernel_type='Scale_RBF', 
                 learn_inducing_locations=True,
                 input_dim=1):
        
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=learn_inducing_locations)
        super(Variational_GP, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ZeroMean()

        self.covar_module_input = helper_specify_kernel_by_name(kernel_type, input_dim=input_dim)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module_input(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class Multi_Variational_IGP:
    '''each output corresponds to an independent SVGP model'''
    def __init__(self, 
                 num_models, 
                 inducing_points, 
                 init_likelihood_noise, 
                 kernel_type='Scale_RBF', 
                 learn_inducing_locations=True,
                 input_dim = 1):
        
        self.models = [Variational_GP(inducing_points, kernel_type, learn_inducing_locations, input_dim) for _ in range(num_models)]
        self.likelihoods = [GaussianLikelihood() for _ in range(num_models)]

        for likelihood in self.likelihoods:
            likelihood.noise = init_likelihood_noise

    def get_model(self, model_number):
        if 0 <= model_number <= len(self.models) - 1:
            return self.models[model_number]
        else:
            raise ValueError(f"Model number must be between 1 and {len(self.models)}")
    
    def get_likelihood(self, likelihood_number):
        if 0 <= likelihood_number <= len(self.likelihoods) - 1:
            return self.likelihoods[likelihood_number]
        else:
            raise ValueError(f"Likelihood number must be between 1 and {len(self.likelihoods)}")