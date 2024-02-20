import math
import warnings
from copy import deepcopy
from typing import Any, Optional, Tuple, Union
import torch
from linear_operator.operators import LinearOperator, MaskedLinearOperator, ZeroLinearOperator
from torch import Tensor
from torch.distributions import Distribution, Normal
from gpytorch import settings
from gpytorch.constraints import Interval
from gpytorch.distributions import base_distributions, MultivariateNormal
from gpytorch.priors import Prior
from gpytorch.utils.warnings import GPInputWarning
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.likelihoods.noise_models import FixedGaussianNoise, HomoskedasticNoise, Noise


class _GaussianLikelihoodBase(Likelihood):
    """Base class for Gaussian Likelihoods, supporting general heteroskedastic noise models."""

    has_analytic_marginal = True

    def __init__(self, noise_covar: Union[Noise, FixedGaussianNoise], **kwargs: Any) -> None:
        super().__init__()
        param_transform = kwargs.get("param_transform")
        if param_transform is not None:
            warnings.warn(
                "The 'param_transform' argument is now deprecated. If you want to use a different "
                "transformaton, specify a different 'noise_constraint' instead.",
                DeprecationWarning,
            )

        self.noise_covar = noise_covar

    def _shaped_noise_covar(self, base_shape: torch.Size, *params: Any, **kwargs: Any) -> Union[Tensor, LinearOperator]:
        return self.noise_covar(*params, shape=base_shape, **kwargs)

    def expected_log_prob(self, target: Tensor, input: MultivariateNormal, *params: Any, **kwargs: Any) -> Tensor:

        way = 'way2'

        if way == 'way1':
            noise = self._shaped_noise_covar(input.mean.shape, *params, **kwargs).diagonal(dim1=-1, dim2=-2)
            # Potentially reshape the noise to deal with the multitask case
            noise = noise.view(*noise.shape[:-1], *input.event_shape)
            
            mean, variance = input.mean, input.variance.clamp_min(1e-8)
            full_variance = variance + noise 
            # the original implementation is: (# NOTE I find misunderstanding ...(even wrong))
            # res = ((target - mean).square() + variance) / noise + noise.log() + math.log(2 * math.pi)
            res = ((target - mean) / full_variance.sqrt()).square() + full_variance.log() + math.log(2 * math.pi)
            res = res.mul(-0.5)

        elif way == 'way2':
            # NOTE: This is my newly added implementation. way1 and way2 are tested to have same output (with tolerence for numerical differences)
            res = self.log_marginal(target, input, *params, **kwargs)

        return res

    def forward(self, function_samples: Tensor, *params: Any, **kwargs: Any) -> Normal:
        noise = self._shaped_noise_covar(function_samples.shape, *params, **kwargs).diagonal(dim1=-1, dim2=-2)
        return base_distributions.Normal(function_samples, noise.sqrt())

    def log_marginal(
        self, observations: Tensor, function_dist: MultivariateNormal, *params: Any, **kwargs: Any
    ) -> Tensor:
        marginal = self.marginal(function_dist, *params, **kwargs)

        # We're making everything conditionally independent
        indep_dist = base_distributions.Normal(marginal.mean, marginal.variance.clamp_min(1e-8).sqrt())
        res = indep_dist.log_prob(observations)

        return res

    def marginal(self, function_dist: MultivariateNormal, *params: Any, **kwargs: Any) -> MultivariateNormal:
        mean, covar = function_dist.mean, function_dist.lazy_covariance_matrix
        noise_covar = self._shaped_noise_covar(mean.shape, *params, **kwargs)
        full_covar = covar + noise_covar
        return function_dist.__class__(mean, full_covar)


class GaussianLikelihood(_GaussianLikelihoodBase):
    r"""
    The standard likelihood for regression.
    Assumes a standard homoskedastic noise model:

    .. math::
        p(y \mid f) = f + \epsilon, \quad \epsilon \sim \mathcal N (0, \sigma^2)

    where :math:`\sigma^2` is a noise parameter.

    .. note::
        This likelihood can be used for exact or approximate inference.

    .. note::
        GaussianLikelihood has an analytic marginal distribution.

    :param noise_prior: Prior for noise parameter :math:`\sigma^2`.
    :param noise_constraint: Constraint for noise parameter :math:`\sigma^2`.
    :param batch_shape: The batch shape of the learned noise parameter (default: []).
    :param kwargs:

    :ivar torch.Tensor noise: :math:`\sigma^2` parameter (noise)
    """

    def __init__(
        self,
        noise_prior: Optional[Prior] = None,
        noise_constraint: Optional[Interval] = None,
        batch_shape: torch.Size = torch.Size(),
        **kwargs: Any,
    ) -> None:
        noise_covar = HomoskedasticNoise(
            noise_prior=noise_prior, noise_constraint=noise_constraint, batch_shape=batch_shape
        )
        super().__init__(noise_covar=noise_covar)

    @property
    def noise(self) -> Tensor:
        return self.noise_covar.noise

    @noise.setter
    def noise(self, value: Tensor) -> None:
        self.noise_covar.initialize(noise=value)

    @property
    def raw_noise(self) -> Tensor:
        return self.noise_covar.raw_noise

    @raw_noise.setter
    def raw_noise(self, value: Tensor) -> None:
        self.noise_covar.initialize(raw_noise=value)

    def marginal(self, function_dist: MultivariateNormal, *args: Any, **kwargs: Any) -> MultivariateNormal:
        r"""
        :return: Analytic marginal :math:`p(\mathbf y)`.
        """
        return super().marginal(function_dist, *args, **kwargs)

class GaussianLikelihoodWithMissingObs(GaussianLikelihood):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def _get_masked_obs(x):
        missing_idx = x.isnan()
        x_masked = x.masked_fill(missing_idx, -999.)
        return missing_idx, x_masked

    def expected_log_prob(self, target, input, *params, **kwargs):
        missing_idx, target = self._get_masked_obs(target)
        res = super().expected_log_prob(target, input, *params, **kwargs)
        return res * ~missing_idx

    def log_marginal(self, observations, function_dist, *params, **kwargs):
        missing_idx, observations = self._get_masked_obs(observations)
        res = super().log_marginal(observations, function_dist, *params, **kwargs)
        return res * ~missing_idx

