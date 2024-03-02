from gpytorch.mlls import VariationalELBO, MarginalLogLikelihood
from gpytorch.utils.generic import length_safe_zip
from torch.nn import ModuleList

class SumVariationalELBO(MarginalLogLikelihood):
    """Sum of variational elbo, to be used with Multi-Output models.

    Args:
        model: A MultiOutputModel
        num_data_list: A List of number of data
        mll_cls: Variational ELBO class

    In case the model outputs are independent, this provives the MLL of the multi-output model.

    """

    def __init__(self, model, num_data_list, mll_cls=VariationalELBO):
        super().__init__(model.likelihood, model)
        self.mlls = ModuleList([mll_cls(mdl.likelihood, mdl, num_data_list[i]) for i, mdl in enumerate(model.models)])

    def forward(self, outputs, targets, *params):
        """
        Args:
            outputs: (Iterable[MultivariateNormal]) - the outputs of the latent function
            targets: (Iterable[Tensor]) - the target values
            params: (Iterable[Iterable[Tensor]]) - the arguments to be passed through
                (e.g. parameters in case of heteroskedastic likelihoods)
        """
        if len(params) == 0:
            sum_mll = sum(mll(output, target) for mll, output, target in length_safe_zip(self.mlls, outputs, targets))
        else:
            sum_mll = sum(
                mll(output, target, *iparams)
                for mll, output, target, iparams in length_safe_zip(self.mlls, outputs, targets, params)
            )
        return sum_mll.div_(len(self.mlls))