from typing import Optional
import torch
from abc import ABC
from gpytorch.module import Module
from gpytorch.models import ApproximateGP
from torch import Tensor
from code_blocks.variational.cholesky_kronecker_variational_distribution import CholeskyKroneckerVariationalDistribution
from gpytorch.utils.errors import CachingError
from gpytorch.settings import _linalg_dtype_cholesky, trace_mode
from gpytorch.distributions import MultivariateNormal, Distribution
from gpytorch.utils.memoize import cached, clear_cache_hook, pop_from_cache_ignore_args
from linear_operator.operators import (
    DiagLinearOperator,
    LinearOperator,
    MatmulLinearOperator,
    SumLinearOperator,
    TriangularLinearOperator,
    KroneckerProductLinearOperator,
)
from linear_operator.utils.cholesky import psd_safe_cholesky
from linear_operator import to_dense
from gpytorch import settings

class KroneckerVariationalStrategy(Module, ABC):

    def __init__(
        self,
        model: ApproximateGP,
        inducing_points_latent: Tensor,
        inducing_points_input: Tensor,
        variational_distribution: CholeskyKroneckerVariationalDistribution,
        learn_inducing_locations_latent: bool = True,
        learn_inducing_locations_input: bool = True,
        jitter_val: Optional[float] = None,
    ):
        super().__init__()
        self._jitter_val = jitter_val

        # Model
        object.__setattr__(self, "model", model)

        # Inducing points latent in space and input space
        inducing_points_latent = inducing_points_latent.clone()
        inducing_points_input = inducing_points_input.clone()

        if inducing_points_latent.dim() == 1:
            inducing_points_latent = inducing_points_latent.unsqueeze(-1)
        if inducing_points_input.dim() == 1:
            inducing_points_input = inducing_points_input.unsqueeze(-1)

        if learn_inducing_locations_latent:
            self.register_parameter(name="inducing_points_latent", parameter=torch.nn.Parameter(inducing_points_latent))
        else:
            self.register_buffer("inducing_points_latent", inducing_points_latent)

        if learn_inducing_locations_input:
            self.register_parameter(name="inducing_points_input", parameter=torch.nn.Parameter(inducing_points_input))
        else:
            self.register_buffer("inducing_points_input", inducing_points_input)
        
        # Variational distribution
        self._variational_distribution = variational_distribution

    @cached(name="cholesky_factor_latent", ignore_args=False)
    def _cholesky_factor_latent(self, induc_induc_covar: LinearOperator) -> TriangularLinearOperator:
        L = psd_safe_cholesky(to_dense(induc_induc_covar).type(_linalg_dtype_cholesky.value()), max_tries=4)
        return TriangularLinearOperator(L)
    
    @cached(name="cholesky_factor_input", ignore_args=False)
    def _cholesky_factor_input(self, induc_induc_covar: LinearOperator) -> TriangularLinearOperator:
        L = psd_safe_cholesky(to_dense(induc_induc_covar).type(_linalg_dtype_cholesky.value()), max_tries=4)
        return TriangularLinearOperator(L)

    @property
    @cached(name="prior_distribution_memo")
    def prior_distribution(self) -> MultivariateNormal:
        zeros = torch.zeros(
            self._variational_distribution.shape(),
            dtype=self._variational_distribution.dtype,
            device=self._variational_distribution.device,
        )
        ones = torch.ones_like(zeros)
        res = MultivariateNormal(zeros, DiagLinearOperator(ones))
        return res

    @property
    @cached(name="variational_distribution_memo")
    def variational_distribution(self) -> Distribution:
        return self._variational_distribution()

    @property
    def jitter_val(self) -> float:
        if self._jitter_val is None:
            return settings.variational_cholesky_jitter.value(dtype=self.inducing_points_latent.dtype)
        return self._jitter_val

    @jitter_val.setter
    def jitter_val(self, jitter_val: float):
        self._jitter_val = jitter_val
    
    def _clear_cache(self) -> None:
        clear_cache_hook(self)

    def kl_divergence(self) -> Tensor:
        r"""
        Compute the KL divergence between the variational inducing distribution :math:`q(\mathbf u)`
        and the prior inducing distribution :math:`p(\mathbf u)`.
        """
        # NOTE: due to whitening, prior is N(0,I) not N(0, K_ZZ).
        with settings.max_preconditioner_size(0):
            kl_divergence = torch.distributions.kl.kl_divergence(self.variational_distribution, self.prior_distribution)
        return kl_divergence
        
    def forward(
        self,
        latents: Tensor,
        inputs: Tensor,
        inducing_points_latent: Tensor,
        inducing_points_input: Tensor,
        inducing_values: Tensor, 
        variational_inducing_covar: Optional[LinearOperator] = None,
        **kwargs,
    ) -> MultivariateNormal:
        # Ensure latents, inputs has the same length, i.e. a (latents[i],inputs[i]) pair jointly determines a prediction value / target value.
        assert latents.shape[-2] == inputs.shape[-2]
        mini_batch_size = latents.shape[-2]
        latents, inputs = latents.to(torch.double), inputs.to(torch.double)
        inducing_points_latent, inducing_points_input = inducing_points_latent.to(torch.double), inducing_points_input.to(torch.double)

        # NOTE: following two tensors might contains repeting elements!
        full_latent = torch.cat([latents, inducing_points_latent], dim=-2)
        full_input = torch.cat([inputs, inducing_points_input], dim=-2)

        full_covar_latent = self.model.covar_module_latent(full_latent)
        full_covar_input = self.model.covar_module_input(full_input)

        # Covariance terms
        induc_latent_covar = full_covar_latent[mini_batch_size:, mini_batch_size:]
        induc_input_covar = full_covar_input[mini_batch_size:, mini_batch_size:]
        data_data_covar = full_covar_latent[:mini_batch_size, :mini_batch_size] * full_covar_input[:mini_batch_size, :mini_batch_size] # elementwise product

        induc_latent_data_latent_covar = full_covar_latent[mini_batch_size:, :mini_batch_size] # (n_induc_latent, mini_batch_size)
        induc_input_data_input_covar = full_covar_input[mini_batch_size:, :mini_batch_size] # (n_induc_input, mini_batch_size)
        n_induc_latent, n_induc_input = inducing_points_latent.shape[-2], inducing_points_input.shape[-2]

        # Some Test Unit
        # assert induc_latent_data_latent_covar.shape[-2] == n_induc_latent
        # assert induc_input_data_input_covar.shape[-2] == n_induc_input
        # assert induc_latent_data_latent_covar.shape[-1] == mini_batch_size
        # assert induc_input_data_input_covar.shape[-1] == mini_batch_size

        # broadcasting
        induc_data_covar = induc_latent_data_latent_covar.to_dense().unsqueeze(1) * induc_input_data_input_covar.to_dense().unsqueeze(0)
        induc_data_covar = induc_data_covar.reshape((n_induc_latent*n_induc_input), mini_batch_size)
        
        # Compute interpolation terms
        # K_uu^{-1/2} K_uf

        L_latent_inv = self._cholesky_factor_latent(induc_latent_covar).solve(torch.eye(induc_latent_covar.size(-1), device=induc_latent_covar.device, dtype=induc_latent_covar.dtype))
        L_input_inv = self._cholesky_factor_input(induc_input_covar).solve(torch.eye(induc_input_covar.size(-1), device=induc_input_covar.device, dtype=induc_input_covar.dtype))
        L_inv = KroneckerProductLinearOperator(L_latent_inv, L_input_inv).to_dense()
        
        if L_inv.shape[0] != induc_data_covar.shape[0]:
            print('nasty shape incompatibilies error happens!')
            # Aggressive caching can cause nasty shape incompatibilies when evaluating with different batch shapes
            # TODO: Use a hook fo this
            try:
                pop_from_cache_ignore_args(self, "cholesky_factor")
            except CachingError:
                pass
            L_latent_inv = self._cholesky_factor_latent(induc_latent_covar).solve(torch.eye(induc_latent_covar.size(-1), device=induc_latent_covar.device, dtype=induc_latent_covar.dtype))
            L_input_inv = self._cholesky_factor_input(induc_input_covar).solve(torch.eye(induc_input_covar.size(-1), device=induc_input_covar.device, dtype=induc_input_covar.dtype))
            L_inv = KroneckerProductLinearOperator(L_latent_inv, L_input_inv).to_dense()

        interp_term = (L_inv @ induc_data_covar.to(L_inv.dtype))

        # Compute the mean of q(f), K_fu K_uu^{-1/2} u
        predictive_mean = (interp_term.transpose(-1, -2) @ (inducing_values.to(interp_term.dtype).unsqueeze(-1)).squeeze(-1))

        # Compute the covariance of q(f)
        middle_term = self.prior_distribution.lazy_covariance_matrix.mul(-1).to(interp_term.dtype)
        if variational_inducing_covar is not None:
            middle_term = SumLinearOperator(variational_inducing_covar, middle_term)

        # TODO: avoid multiplication with big matrices, use kronecker product properties ... 
        if trace_mode.on():
            predictive_covar = (
                data_data_covar.add_jitter(self.jitter_val).to_dense()
                + interp_term.transpose(-1, -2) @ middle_term.to_dense() @ interp_term
            )
        else:
            predictive_covar = SumLinearOperator(
                data_data_covar.add_jitter(self.jitter_val),
                MatmulLinearOperator(interp_term.transpose(-1, -2), middle_term.to(interp_term.dtype) @ interp_term),
            )
        
        # Return the distribution
        return MultivariateNormal(predictive_mean, predictive_covar)
    
    def __call__(self, latents: Tensor, inputs: Tensor, prior: bool = False, **kwargs) -> MultivariateNormal:
        if prior:
            return self.model.forward(latents, inputs, **kwargs)

        if self.training:
            self._clear_cache()
        
        inducing_points_latent = self.inducing_points_latent
        inducing_points_input = self.inducing_points_input

        # Get p(u)/q(u)
        variational_dist_u = self.variational_distribution

        # Get q(f)
        if isinstance(variational_dist_u, MultivariateNormal): 
            return super().__call__(
                latents,
                inputs,
                inducing_points_latent,
                inducing_points_input,
                inducing_values=variational_dist_u.mean,
                variational_inducing_covar=variational_dist_u.lazy_covariance_matrix,
                **kwargs,
            )
        else:
            raise RuntimeError(
                f"Invalid variational distribuition ({type(variational_dist_u)}). "
                "Expected a multivariate normal or a delta distribution (NOT IMPLEMENTED YET)."
            )

        