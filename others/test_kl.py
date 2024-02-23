import torch 
from torch import Tensor
from linear_operator.operators import CholLinearOperator, TriangularLinearOperator, KroneckerProductLinearOperator, DiagLinearOperator
import time
from gpytorch.distributions import MultivariateNormal
from torch.distributions import MultivariateNormal as TMultivariateNormal

M_H = 10
M_X = 30
L_H_1 = torch.randn(M_H, M_H).to(torch.double)
L_X_1 = torch.randn(M_X, M_X).to(torch.double)
# _L_H = torch.eye(M_H).to(torch.double)
# _L_X = torch.eye(M_X).to(torch.double)
variational_mean = torch.randn(M_X * M_H).to(torch.double)
# variational_mean[:10] = torch.pi

lower_mask_latent = torch.ones(L_H_1.shape[-2:]).tril(0)
lower_mask_input = torch.ones(L_X_1.shape[-2:]).tril(0)
L_H = L_H_1.mul(lower_mask_latent)
L_X = L_X_1.mul(lower_mask_input)

for i in range(L_H.size(0)): 
    L_H[i, i] = L_H[i, i] ** 2 + 1.


for i in range(L_X.size(0)): 
    L_X[i, i] = L_X[i, i] ** 2 + 1.

varcov = KroneckerProductLinearOperator(CholLinearOperator(TriangularLinearOperator(L_H)), CholLinearOperator(TriangularLinearOperator(L_X)))
# varcov_ = torch.kron(CholLinearOperator(TriangularLinearOperator(L_H)).to_dense(), CholLinearOperator(TriangularLinearOperator(L_X)).to_dense())
q_u = MultivariateNormal(variational_mean, varcov)
zeros = torch.zeros(variational_mean.shape[0])
p_u = MultivariateNormal(zeros, DiagLinearOperator(torch.ones_like(zeros)))

q_u_2 = TMultivariateNormal(variational_mean, varcov.to_dense())
p_u_2 = TMultivariateNormal(zeros, torch.eye(zeros.shape[0]))

# make sure diag all positive ... 
start_time = time.time()
L_H_ = torch.linalg.cholesky(CholLinearOperator(TriangularLinearOperator(L_H)).to_dense())
L_X_ = torch.linalg.cholesky(CholLinearOperator(TriangularLinearOperator(L_X)).to_dense())
end_time = time.time()


start_time = time.time()
res_2 = torch.distributions.kl.kl_divergence(q_u, p_u)
end_time = time.time()
print('GpyTorch method', res_2)
print('total time:', end_time - start_time)