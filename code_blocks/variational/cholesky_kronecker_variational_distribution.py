import torch
from abc import ABC
from gpytorch.module import Module
from gpytorch.distributions import MultivariateNormal
from linear_operator.operators import CholLinearOperator, TriangularLinearOperator, KroneckerProductLinearOperator

class CholeskyKroneckerVariationalDistribution(Module, ABC):

    def __init__(
        self,
        n_inducing_input: int,
        n_inducing_latent: int,
    ):
        super().__init__()
        self.n_inducing_input = n_inducing_input
        self.n_inducing_latent = n_inducing_latent

        mean_init = torch.zeros(n_inducing_input * n_inducing_latent)
        covar_init_latent = torch.eye(n_inducing_latent, n_inducing_latent)
        covar_init_input = torch.eye(n_inducing_input, n_inducing_input)

        self.register_parameter(name="variational_mean", parameter=torch.nn.Parameter(mean_init))
        self.register_parameter(name="chol_variational_covar_latent", parameter=torch.nn.Parameter(covar_init_latent))
        self.register_parameter(name="chol_variational_covar_input", parameter=torch.nn.Parameter(covar_init_input))

        _ = self.forward() # get self.variational_covar in this step.

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def shape(self) -> torch.Size:
        r"""
        rtype: torch.Size
        There are n_inducing_input * n_inducing_latent inducing points.
        """
        return torch.Size([(self.n_inducing_input)*(self.n_inducing_latent)])

    def forward(self) -> MultivariateNormal:
        chol_variational_covar_latent = self.chol_variational_covar_latent
        chol_variational_covar_input = self.chol_variational_covar_input
        dtype = chol_variational_covar_latent.dtype
        device = chol_variational_covar_latent.device

        # First make the cholesky factor is upper triangular
        lower_mask_latent = torch.ones(self.chol_variational_covar_latent.shape[-2:], dtype=dtype, device=device).tril(0)
        chol_variational_covar_latent = TriangularLinearOperator(chol_variational_covar_latent.mul(lower_mask_latent))
        
        lower_mask_input = torch.ones(self.chol_variational_covar_input.shape[-2:], dtype=dtype, device=device).tril(0)
        chol_variational_covar_input = TriangularLinearOperator(chol_variational_covar_input.mul(lower_mask_input))

        # Now construct the actual covariance matrix
        variational_covar_latent = CholLinearOperator(chol_variational_covar_latent)
        variational_covar_input = CholLinearOperator(chol_variational_covar_input)
        self.variational_covar = KroneckerProductLinearOperator(variational_covar_latent, variational_covar_input)

        return MultivariateNormal(self.variational_mean, self.variational_covar)

    def initialize_variational_distribution(self, prior_dist: MultivariateNormal) -> None:
        raise NotImplementedError("This function is not implemented yet.")
