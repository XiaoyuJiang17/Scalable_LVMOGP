from abc import ABC, abstractmethod
import torch
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood


class _ApproximateMarginalLogLikelihood(MarginalLogLikelihood, ABC):
    r"""
    An approximate marginal log likelihood (typically a bound) for approximate GP models.
    We expect that model is a :obj:`gpytorch.models.ApproximateGP`.

    Args:
        likelihood (:obj:`gpytorch.likelihoods.Likelihood`):
            The likelihood for the model
        model (:obj:`gpytorch.models.ApproximateGP`):
            The approximate GP model
        num_data (int):
            The total number of training data points (necessary for SGD)
        beta (float - default 1.):
            A multiplicative factor for the KL divergence term.
            Setting it to 1 (default) recovers true variational inference
            (as derived in `Scalable Variational Gaussian Process Classification`_).
            Setting it to anything less than 1 reduces the regularization effect of the model
            (similarly to what was proposed in `the beta-VAE paper`_).
        combine_terms (bool):
            Whether or not to sum the expected NLL with the KL terms (default True)
    """

    def __init__(self, likelihood, model, num_data, beta=1.0, alpha=1.0, combine_terms=True):
        super().__init__(likelihood, model)
        self.combine_terms = combine_terms
        self.num_data = num_data
        self.beta = beta
        self.alpha = alpha

    @abstractmethod
    def _log_likelihood_term(self, approximate_dist_f, target, **kwargs):
        raise NotImplementedError

    def forward(self, approximate_dist_f, target, **kwargs):
        r"""
        Computes the Variational ELBO given :math:`q(\mathbf f)` and `\mathbf y`.
        Calling this function will call the likelihood's `expected_log_prob` function.

        Args:
            approximate_dist_f (:obj:`gpytorch.distributions.MultivariateNormal`):
                :math:`q(\mathbf f)` the outputs of the latent function (the :obj:`gpytorch.models.ApproximateGP`)
            target (`torch.Tensor`):
                :math:`\mathbf y` The target values

        Keyword Args:
            Additional arguments passed to the likelihood's `expected_log_prob` function.
        """
        # Get likelihood term and KL term
        num_batch = approximate_dist_f.event_shape[0]
        log_likelihood = self._log_likelihood_term(approximate_dist_f, target, **kwargs).div(num_batch)
        kl_divergence = self.model.variational_strategy.kl_divergence().div(self.num_data / self.beta)

        # Add any additional registered loss terms
        added_loss = torch.zeros_like(log_likelihood)
        had_added_losses = False
        for added_loss_term in self.model.added_loss_terms():
            # ONLY one added loss here, which is KL in latent space
            added_loss.add_(self.alpha * added_loss_term.loss())
            had_added_losses = True

        # Log prior term
        log_prior = torch.zeros_like(log_likelihood)
        for name, module, prior, closure, _ in self.named_priors():
            log_prior.add_(prior.log_prob(closure(module)).sum().div(self.num_data))

        if self.combine_terms:
            # print('log_likelihood :' + str(log_likelihood) + 'kl_divergence :' + str(kl_divergence) + 'added_loss :' + str(added_loss))
            return log_likelihood - kl_divergence + log_prior - added_loss
        else:
            if had_added_losses:
                return log_likelihood, kl_divergence, log_prior, added_loss
            else:
                return log_likelihood, kl_divergence, log_prior


class VariationalELBO(_ApproximateMarginalLogLikelihood):
    r"""
    The variational evidence lower bound (ELBO). This is used to optimize
    variational Gaussian processes (with or without stochastic optimization).

    .. math::

       \begin{align*}
          \mathcal{L}_\text{ELBO} &=
          \mathbb{E}_{p_\text{data}( y, \mathbf x )} \left[
            \mathbb{E}_{p(f \mid \mathbf u, \mathbf x) q(\mathbf u)} \left[  \log p( y \! \mid \! f) \right]
          \right] - \beta \: \text{KL} \left[ q( \mathbf u) \Vert p( \mathbf u) \right]
          \\
          &\approx \sum_{i=1}^N \mathbb{E}_{q( f_i)} \left[
            \log p( y_i \! \mid \! f_i) \right] - \beta \: \text{KL} \left[ q( \mathbf u) \Vert p( \mathbf u) \right]
       \end{align*}

    where :math:`N` is the number of datapoints, :math:`q(\mathbf u)` is the variational distribution for
    the inducing function values, :math:`q(f_i)` is the marginal of
    :math:`p(f_i \mid \mathbf u, \mathbf x_i) q(\mathbf u)`,
    and :math:`p(\mathbf u)` is the prior distribution for the inducing function values.

    :math:`\beta` is a scaling constant that reduces the regularization effect of the KL
    divergence. Setting :math:`\beta=1` (default) results in the true variational ELBO.

    For more information on this derivation, see `Scalable Variational Gaussian Process Classification`_
    (Hensman et al., 2015).

    :param ~gpytorch.likelihoods.Likelihood likelihood: The likelihood for the model
    :param ~gpytorch.models.ApproximateGP model: The approximate GP model
    :param int num_data: The total number of training data points (necessary for SGD)
    :param float beta: (optional, default=1.) A multiplicative factor for the KL divergence term.
        Setting it to 1 (default) recovers true variational inference
        (as derived in `Scalable Variational Gaussian Process Classification`_).
        Setting it to anything less than 1 reduces the regularization effect of the model
        (similarly to what was proposed in `the beta-VAE paper`_).
    :param bool combine_terms: (default=True): Whether or not to sum the
        expected NLL with the KL terms (default True)

    Example:
        >>> # model is a gpytorch.models.ApproximateGP
        >>> # likelihood is a gpytorch.likelihoods.Likelihood
        >>> mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=100, beta=0.5)
        >>>
        >>> output = model(train_x)
        >>> loss = -mll(output, train_y)
        >>> loss.backward()

    .. _Scalable Variational Gaussian Process Classification:
        http://proceedings.mlr.press/v38/hensman15.pdf
    .. _the beta-VAE paper:
        https://openreview.net/pdf?id=Sy2fzU9gl
    """
    def __init__(self, likelihood, model, num_data, beta=1.0, alpha=1.0, combine_terms=True):
        super().__init__(likelihood, model, num_data, beta, alpha, combine_terms)

    def _log_likelihood_term(self, variational_dist_f, target, **kwargs):
        return self.likelihood.expected_log_prob(target, variational_dist_f, **kwargs).sum(-1)

    def forward(self, variational_dist_f, target, **kwargs):
        r"""
        Computes the Variational ELBO given :math:`q(\mathbf f)` and :math:`\mathbf y`.
        Calling this function will call the likelihood's :meth:`~gpytorch.likelihoods.Likelihood.expected_log_prob`
        function.

        :param ~gpytorch.distributions.MultivariateNormal variational_dist_f: :math:`q(\mathbf f)`
            the outputs of the latent function (the :obj:`gpytorch.models.ApproximateGP`)
        :param torch.Tensor target: :math:`\mathbf y` The target values
        :param kwargs: Additional arguments passed to the
            likelihood's :meth:`~gpytorch.likelihoods.Likelihood.expected_log_prob` function.
        :rtype: torch.Tensor
        :return: Variational ELBO. Output shape corresponds to batch shape of the model/input data.
        """
        return super().forward(variational_dist_f, target, **kwargs)

class ClfVariationalELBO(MarginalLogLikelihood, ABC):
    '''
    This class is largely the same as _ApproximateMarginalLogLikelihood and VariationalELBO.
    The reason to have this class is to better suit multi-output multi-class classfication with LVMOGP-SVI model.
    '''
    def __init__(self, likelihood, model, num_data, beta=1.0, alpha=1.0, combine_terms=True):
        super().__init__(likelihood, model)
        self.combine_terms = combine_terms
        self.num_data = num_data
        self.beta = beta
        self.alpha = alpha
    
    def _log_likelihood_term(self, variational_dist_f, ref, **kwargs):
        '''
        different from VariationalELBO, no 'target' is needed for multi-output multi-class classfication with LVMOGP-SVI model,
        no .sum(-1) at the end as well.
        NOTE self.likelihood.expected_log_prob here MUST be Multi_Output_Multi_Class_AR.
        '''
        return self.likelihood.expected_log_prob(variational_dist_f, ref, **kwargs)
    
    def forward(self, approximate_dist_f, ref, **kwargs):
        r"""
        Computes the Variational ELBO given :math:`q(\mathbf f)` and `\mathbf y`.
        Calling this function will call the likelihood's `expected_log_prob` function.

        Args:
            approximate_dist_f (:obj:`gpytorch.distributions.MultivariateNormal`):
                :math:`q(\mathbf f)` the outputs of the latent function (the :obj:`gpytorch.models.ApproximateGP`)
            ref (Tensor):

        Keyword Args:
            Additional arguments passed to the likelihood's `expected_log_prob` function.
        """
        # Get likelihood term and KL term
        num_batch = approximate_dist_f.event_shape[0]
        log_likelihood = self._log_likelihood_term(approximate_dist_f, ref, **kwargs).div(num_batch)
        kl_divergence = self.model.variational_strategy.kl_divergence().div(self.num_data / self.beta)

        # Add any additional registered loss terms
        added_loss = torch.zeros_like(log_likelihood)
        had_added_losses = False
        for added_loss_term in self.model.added_loss_terms():
            # ONLY one added loss here, which is KL in latent space
            added_loss.add_(self.alpha * added_loss_term.loss())
            had_added_losses = True

        # Log prior term
        log_prior = torch.zeros_like(log_likelihood)
        for name, module, prior, closure, _ in self.named_priors():
            log_prior.add_(prior.log_prob(closure(module)).sum().div(self.num_data))

        if self.combine_terms:
            return log_likelihood - kl_divergence + log_prior - added_loss
        else:
            if had_added_losses:
                return log_likelihood, kl_divergence, log_prior, added_loss
            else:
                return log_likelihood, kl_divergence, log_prior


