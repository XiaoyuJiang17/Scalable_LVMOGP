from gpytorch.models.gp import GP
from gpytorch.models.pyro import _PyroMixin  # This will only contain functions if Pyro is installed


class ApproximateGP_(GP, _PyroMixin):
    r"""
    difference to ApproximateGP happens at __call__ function, 
    which allows two sets of inputs.
    """

    def __init__(self, variational_strategy):
        super().__init__()
        self.variational_strategy = variational_strategy

    def forward(self, x):
        raise NotImplementedError

    def pyro_guide(self, input, beta=1.0, name_prefix=""):
        
        raise NotImplementedError
        # return super().pyro_guide(input, beta=beta, name_prefix=name_prefix)

    def pyro_model(self, input, beta=1.0, name_prefix=""):
        
        raise NotImplementedError
        # return super().pyro_model(input, beta=beta, name_prefix=name_prefix)

    def get_fantasy_model(self, inputs, targets, **kwargs):
        
        raise NotImplementedError
        # return self.variational_strategy.get_fantasy_model(inputs=inputs, targets=targets, **kwargs)

    def __call__(self, inputs_X, inputs_C, prior=False, **kwargs):
        if inputs_X.dim() == 1:
            inputs_X = inputs_X.unsqueeze(-1)

        if inputs_C.dim() == 1:
            inputs_C = inputs_C.unsqueeze(-1)

        return self.variational_strategy(inputs_X, inputs_C, prior=prior, **kwargs)

class BayesianGPLVM_(ApproximateGP_):
    """
    Same as BayesianGPLVM except inherited from ApproximateGP_ now.
    """

    def __init__(self, X, variational_strategy):
        super().__init__(variational_strategy)

        # Assigning Latent Variable
        self.X = X

    def forward(self):
        raise NotImplementedError

    def sample_latent_variable(self, *args, **kwargs):
        sample = self.X(*args, **kwargs)
        return sample
