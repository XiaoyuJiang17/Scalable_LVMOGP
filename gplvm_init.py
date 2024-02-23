# Train an GPLVM-svi and use trained latent variables as initialization for lvmogp-svi.
import torch
from tqdm import trange
from gpytorch.models import ApproximateGP
from gpytorch.variational import VariationalStrategy
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.priors import MultivariateNormalPrior
from gpytorch.mlls import VariationalELBO
from code_blocks.gplvm.latent_variables import VariationalLatentVariable
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.distributions import MultivariateNormal
import numpy as np

class BayesianGPLVM(ApproximateGP):
    def __init__(self, X, variational_strategy):
        
        """The GPLVM model class for unsupervised learning. The current class supports
        
        (a) Point estimates for latent X when prior_x = None 
        (b) MAP Inference for X when prior_x is not None and inference == 'map'
        (c) Gaussian variational distribution q(X) when prior_x is not None and inference == 'variational'

        :param X (LatentVariable): An instance of a sub-class of the LatentVariable class.
                                    One of,
                                    PointLatentVariable / 
                                    MAPLatentVariable / 
                                    VariationalLatentVariable to
                                    facilitate inference with (a), (b) or (c) respectively.
       
        """
        super(BayesianGPLVM, self).__init__(variational_strategy)
        
        # Assigning Latent Variable 
        self.X = X 
    
    def forward(self):
        raise NotImplementedError
          
    def sample_latent_variable(self, *args, **kwargs):
        sample = self.X(*args, **kwargs)
        return sample

class GPLVM(BayesianGPLVM):

    def __init__(self,
                 n,
                 data_dim,
                 latent_dim,
                 n_inducing):

        self.n = n
        self.batch_shape = torch.Size([data_dim])
        self.inducing_inputs = torch.randn(data_dim, n_inducing, latent_dim)

        q_u = CholeskyVariationalDistribution(n_inducing, batch_shape=self.batch_shape)
        q_f = VariationalStrategy(self, self.inducing_inputs, q_u, learn_inducing_locations=True)

        X_prior_mean = torch.zeros(n, latent_dim)
        X_init = torch.randn(n, latent_dim)

        prior_x = MultivariateNormalPrior(X_prior_mean, torch.eye(X_prior_mean.shape[1]))
        X = VariationalLatentVariable(n, data_dim, latent_dim, X_init, prior_x)

        super(GPLVM, self).__init__(X, q_f)

        self.mean_module = ConstantMean(ard_num_dims=latent_dim)
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=latent_dim))

    def forward(self, X):
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)
        dist = MultivariateNormal(mean_x, covar_x)
        return dist
    
    def _get_batch_idx(self, batch_size):
        valid_indices = np.arange(self.n)
        batch_indices = np.random.choice(valid_indices, size=batch_size, replace=False)
        return np.sort(batch_indices)

def specify_gplvm(config):

    gplvm_model = GPLVM(n = config['n_outputs'],
                        data_dim = config['n_input_train'], 
                        latent_dim = config['trainable_latent_dim'], 
                        n_inducing = config['n_inducing_input'])
    return gplvm_model

def train_gplvm(gplvm_model, 
                gplvm_likelihood,
                data_Y, 
                hyper_parameters=None,
                ):
    '''
    data_Y: tensor of shape (n, data_dim)
    '''
    print('Start Training GPLVM!')
    if hyper_parameters == None:
        batch_size = min(100, data_Y.shape[0])
        hyper_parameters = {'training_steps': 5000,
                            'batch_size': batch_size,
                            'lr': 0.01}
        
    elbo = VariationalELBO(gplvm_likelihood, gplvm_model, num_data=data_Y.shape[0])
    optimizer = torch.optim.Adam(list(gplvm_model.parameters()) + list(gplvm_likelihood.parameters()), lr=hyper_parameters['lr'])

    iterator = trange(hyper_parameters['training_steps'])
    losses = []
    for i in iterator:
        batch_index = gplvm_model._get_batch_idx(hyper_parameters['batch_size'])
        optimizer.zero_grad()
        sample_batch = gplvm_model.sample_latent_variable(batch_idx=batch_index)
        output_batch = gplvm_model(sample_batch)
        loss = -elbo(output_batch, data_Y[batch_index].T).sum()
        losses.append(loss.item())
        iterator.set_description(
            '-elbo: ' + str(np.round(loss.item(), 2)) +\
            ". Step: " + str(i))
        loss.backward()
        optimizer.step()

    return gplvm_model, gplvm_likelihood, losses