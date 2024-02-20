from gpytorch.kernels import Kernel
from gpytorch.constraints import Interval, Positive
import torch
from typing import Optional

def postprocess_rbf(dist_mat):
    return dist_mat.div_(-2).exp_()

class PeriodicInputsRBFKernel(Kernel):

    has_lengthscale = True

    def __init__(self, 
                 period_length_constraint: Optional[Interval] = None,
                 **kwargs):
        super(PeriodicInputsRBFKernel, self).__init__(**kwargs)

        if period_length_constraint is None:
            period_length_constraint = Positive()

        ard_num_dims = kwargs.get("ard_num_dims", 1)
        self.register_parameter(
            name="raw_period_length", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, ard_num_dims))
        )

        self.register_constraint("raw_period_length", period_length_constraint)

    @property
    def period_length(self):
        return self.raw_period_length_constraint.transform(self.raw_period_length)
    
    @period_length.setter
    def period_length(self, value):
        self._set_period_length(value)

    def _set_period_length(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_period_length)
        self.initialize(raw_period_length=self.raw_period_length_constraint.inverse_transform(value))
        
    def forward(self, x1, x2, diag=False, **params):

        _x1 = 2 * x1 * torch.pi / self.period_length
        _x2 = 2 * x2 * torch.pi / self.period_length

        x1_ = torch.cat((torch.sin(_x1).reshape(-1, 1), torch.cos(_x1).reshape(-1, 1)), axis=1) / self.lengthscale
        x2_ = torch.cat((torch.sin(_x2).reshape(-1, 1), torch.cos(_x2).reshape(-1, 1)), axis=1) / self.lengthscale

        return postprocess_rbf(self.covar_dist(x1_, x2_, square_dist=True, diag=diag, **params))
       