import sys
sys.path.append('/Users/jiangxiaoyu/Desktop/All Projects/Scalable_LVMOGP/')
from code_blocks.kernels.periodic_inputs_rbfkernel import PeriodicInputsRBFKernel
from code_blocks.kernels.periodic_inputs_maternkernel import PeriodicInputsMaternKernel
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel, PeriodicKernel
import torch
from torch import Tensor
from linear_operator.operators import KroneckerProductLinearOperator, TriangularLinearOperator, LinearOperator, CholLinearOperator
from linear_operator.utils.cholesky import psd_safe_cholesky
import numpy as np
from tqdm import trange
from linear_operator import to_dense
from gpytorch.settings import _linalg_dtype_cholesky
import matplotlib.pyplot as plt

################################################   Specify Kernels : Helper Function  ################################################


def helper_specify_kernel_by_name(kernel_name, input_dim=None):
    # Kernel (acting on index dimensions)
    '''
    input_dim is a MUST for RBFKernel (RAD), otherwise common lengthscale.
    '''
    if kernel_name == 'Scale_RBF':
        return ScaleKernel(RBFKernel(ard_num_dims=input_dim))
    
    elif kernel_name == 'Scale_Matern32':
        return ScaleKernel(MaternKernel(nu=1.5))
    
    elif kernel_name == 'Scale_Matern52':
        return ScaleKernel(MaternKernel(nu=2.5))
    
    elif kernel_name == 'Periodic_times_Scale_RBF':
        return PeriodicKernel() * ScaleKernel(RBFKernel())
    
    elif kernel_name == 'PeriodicInputsRBFKernel_times_Scale_RBF':
        return PeriodicInputsRBFKernel() * ScaleKernel(RBFKernel())
    
    elif kernel_name == 'Scale_Matern52_plus_Scale_PeriodicInputsMatern52':
        return ScaleKernel(MaternKernel(nu=2.5)) + ScaleKernel(PeriodicInputsMaternKernel(nu=2.5))
    
    
################################################   Inference  ################################################

# Remeber This is an implementation based on whitening .... 
# TODO: better implementation support both whitening and non-whitening ... 

def prepare_common_background_info(my_model, config):
    '''Prepare all values of a dict called common_background_information, which being used in integration_prediction_func'''
    
    def _cholesky_factor(induc_induc_covar: LinearOperator) -> TriangularLinearOperator:
        L = psd_safe_cholesky(to_dense(induc_induc_covar).type(_linalg_dtype_cholesky.value()), max_tries=4)
        return TriangularLinearOperator(L)
    
    K_uu_latent = my_model.covar_module_latent(my_model.variational_strategy.inducing_points_latent.data).to_dense().data
    # K_uu_latent_inv = torch.linalg.solve(K_uu_latent, torch.eye(K_uu_latent.size(-1))
    K_uu_input = my_model.covar_module_input(my_model.variational_strategy.inducing_points_input.data).to_dense().data
    # K_uu_input_inv = torch.linalg.solve(K_uu_input, torch.eye(K_uu_input.size(-1))

    K_uu = KroneckerProductLinearOperator(K_uu_latent, K_uu_input) #.to_dense().data
    # chol_K_uu_inv_t = _cholesky_factor_latent(KroneckerProductLinearOperator(K_uu_latent_inv, K_uu_input_inv)).to_dense().data.t()
    chol_K_uu_inv_t = KroneckerProductLinearOperator(
            torch.linalg.solve( _cholesky_factor(K_uu_latent).to_dense().data, torch.eye(K_uu_latent.size(-1))),
            torch.linalg.solve( _cholesky_factor(K_uu_input).to_dense().data, torch.eye(K_uu_input.size(-1))),
        )._transpose_nonbatch() # .to_dense().data.t()
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------

    chol_covar_latent_u = my_model.variational_strategy._variational_distribution.chol_variational_covar_latent.data
    covar_latent_u = CholLinearOperator(chol_covar_latent_u)
    chol_covar_input_u = my_model.variational_strategy._variational_distribution.chol_variational_covar_input.data
    covar_input_u = CholLinearOperator(chol_covar_input_u)

    covar_u = KroneckerProductLinearOperator(covar_latent_u, covar_input_u) # .to_dense().data

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------

    common_background_information = {
                        'K_uu': K_uu,
                        'chol_K_uu_inv_t': chol_K_uu_inv_t, 
                        'm_u': my_model.variational_strategy._variational_distribution.variational_mean.data,
                        'Sigma_u': covar_u,
                        'A': chol_K_uu_inv_t @ (covar_u - torch.eye(covar_u.shape[0])) @ chol_K_uu_inv_t._transpose_nonbatch(),
                        'var_H': my_model.covar_module_latent.outputscale.data, # based on the use of RBF kernel
                        'var_X': my_model.covar_module_input(Tensor([0.])).to_dense().item(), # This implementation works for any kind of kernel.
                        #'var_X': my_model.covar_module_input.outputscale.data,  # 
                        'W': my_model.covar_module_latent.base_kernel.lengthscale.data.reshape(-1)**2
                        }
    '''
    chol_K_uu_inv_t: inverse of K_uu matrix, of shape (M_H * M_X, M_H * M_X)
    m_u: mean of the variational distribution
    Sigma_u: covariance matrix of the variational distribution
    A: chol_K_uu_inv_t (Sigma_u - K_uu) chol_K_uu_inv_t.T NOTE: whitening ... 
    var_H: 
    var_X: 
    W: vector; containing all lengthscales in the RAD kernel
    c: constant
    '''
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------
    c = (2 * torch.pi)**(config['latent_dim'] / 2) * common_background_information['var_H'] * common_background_information['W'].sqrt().prod()
    common_background_information['constant_c'] = c

    return common_background_information

def prepare_pre_computation(my_model,
                            test_input):
    '''note for each input x*, we compute a corresponding K_u_f_K_f_u, use this function only once to save computation'''
    input_K_f_u = my_model.covar_module_input(test_input, my_model.variational_strategy.inducing_points_input.data).to_dense().data
    input_K_u_fi_K_fi_u_list = [torch.outer(input_K_f_u[i, :], input_K_f_u[i, :]) for i in range(test_input.shape[0])]
    
    pre_compute_dict = {}
    pre_compute_dict['input_K_f_u'] = input_K_f_u
    pre_compute_dict['input_K_u_fi_K_fi_u_list'] = input_K_u_fi_K_fi_u_list

    return pre_compute_dict

def integration_prediction_func(test_input,     # tensor
                                output_index, 
                                my_model, 
                                common_background_information, 
                                pre_compute_dict,
                                config, 
                                latent_type=None,
                                latent_info=None):
    '''
    Current implementation support doing inference for multiple inputs of same output (latent) simutanously ... 
    NOTE pre_compute_list contains some variables which are pre computed (only once, outside this func) for efficiency ... 
    '''

    if latent_type == None:
        # Mean and covariance of latent distribution.
        m_plus_ = my_model.X.q_mu.data[output_index]
        Sigma_plus_ = 1.0 * my_model.X.q_log_sigma.exp().square().data[output_index]

    elif latent_type == 'NNEncoder':
        # In this case, latent_info is the MUST.
        assert latent_info != None
        m_plus_ = my_model.X(latent_info, eval_mode=True)[0].data[output_index]
        Sigma_plus_ = my_model.X(latent_info, eval_mode=True)[-1].square().data[output_index] # already .exp().

    data_specific_background_information = {
        'm_plus': m_plus_,
        'Sigma_plus': Sigma_plus_,
        'expectation_K_uu_dict': {}
    }
    
    # helper functions -----------------------------------------------------------------------------------------------------------------------
    def multivariate_gaussian_pdf(x, mu, cov):
        '''cov is a vector, representing all elements in the diagonal matrix'''
        k = mu.size(0)
        cov_det = cov.prod()
        cov_inv = torch.diag(1.0 / cov)
        norm_factor = torch.sqrt((2 * torch.pi) ** k * cov_det)

        x_mu = x - mu
        result = torch.exp(-0.5 * x_mu @ cov_inv @ x_mu.t()) / norm_factor
        return result.item()

    def G(h:Tensor, common_background_information=common_background_information, data_specific_background_information=data_specific_background_information):

        mu = data_specific_background_information['m_plus']
        cov_diag = data_specific_background_information['Sigma_plus'] + common_background_information['W']
        result = multivariate_gaussian_pdf(h, mu, cov_diag)
        return common_background_information['constant_c'] * result

    def R(h_1:Tensor, h_2:Tensor, common_background_information=common_background_information, data_specific_background_information=data_specific_background_information):
        mu_1 = h_2
        cov_diag_1 = 2 * common_background_information['W']
        mu_2 = (h_1 + h_2) / 2
        cov_diag_2 = 0.5 * common_background_information['W'] + data_specific_background_information['Sigma_plus']
        result1 = multivariate_gaussian_pdf(h_1, mu_1, cov_diag_1)
        result2 = multivariate_gaussian_pdf(data_specific_background_information['m_plus'], mu_2, cov_diag_2)
        return (common_background_information['constant_c'] ** 2 ) * result1 * result2
    
    def expectation_lambda(common_background_information=common_background_information, data_specific_background_information=data_specific_background_information):
        result_ = KroneckerProductLinearOperator(data_specific_background_information['expectation_latent_K_f_u'].reshape(1, -1), pre_compute_dict['input_K_f_u'])
        result_ = result_ @ common_background_information['chol_K_uu_inv_t'] @ common_background_information['m_u']
        return result_
        
    def expectation_lambda_square(common_background_information=common_background_information, data_specific_background_information=data_specific_background_information):
        result_ = common_background_information['m_u']
        _result = result_ @ common_background_information['chol_K_uu_inv_t']._transpose_nonbatch()

        final_result = torch.zeros(test_input.shape[0])

        for i in range(test_input.shape[0]):
            interm_term = KroneckerProductLinearOperator(data_specific_background_information['expectation_latent_K_u_f_K_f_u'], pre_compute_dict['input_K_u_fi_K_fi_u_list'][i])
            result_ = _result @ interm_term @ _result.t()
            final_result[i] = result_.item()
            # Store these result for next time use ... 
            if i not in data_specific_background_information['expectation_K_uu_dict']:
                data_specific_background_information['expectation_K_uu_dict'][i] = interm_term

        return final_result
        
    def expectation_gamma(common_background_information=common_background_information, data_specific_background_information=data_specific_background_information):

        result_ = common_background_information['var_H'] * common_background_information['var_X']

        final_result = torch.zeros(test_input.shape[0])

        for i in range(test_input.shape[0]):
            if i not in data_specific_background_information['expectation_K_uu_dict']:
                data_specific_background_information['expectation_K_uu_dict'][i] = KroneckerProductLinearOperator(data_specific_background_information['expectation_latent_K_u_f_K_f_u'], \
                                                                                                        pre_compute_dict['input_K_u_fi_K_fi_u_list'][i])
            final_result[i] = (result_ + (common_background_information['A'].to_dense() * data_specific_background_information['expectation_K_uu_dict'][i].to_dense()).sum()).item()

        return final_result
    
    def integration_predictive_mean(common_background_information=common_background_information, data_specific_background_information=data_specific_background_information):
        return expectation_lambda(common_background_information=common_background_information, data_specific_background_information=data_specific_background_information)


    def integration_predictive_var(common_background_information=common_background_information, data_specific_background_information=data_specific_background_information):

        term1 = expectation_lambda_square(common_background_information=common_background_information, data_specific_background_information=data_specific_background_information)
        term2 = expectation_gamma(common_background_information=common_background_information, data_specific_background_information=data_specific_background_information)
        term3 = expectation_lambda(common_background_information=common_background_information, data_specific_background_information=data_specific_background_information).pow(2)
        # TODO: something goes wrong from scalar prediction to vector prediction ... 
        result = ( term1 + term2 - term3 )
        assert torch.all(result > 0.)
        return result
    
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    expectation_latent_K_f_u = Tensor([G(my_model.variational_strategy.inducing_points_latent.data[i]).item() for i in range(config['n_inducing_latent'])])
    expectation_latent_K_u_f_K_f_u = Tensor([R(my_model.variational_strategy.inducing_points_latent.data[i], my_model.variational_strategy.inducing_points_latent.data[j]).item() \
                                            for j in range(config['n_inducing_latent']) for i in range(config['n_inducing_latent'])]).reshape(config['n_inducing_latent'], config['n_inducing_latent'])

    data_specific_background_information['expectation_latent_K_f_u'] = expectation_latent_K_f_u
    data_specific_background_information['expectation_latent_K_u_f_K_f_u'] = expectation_latent_K_u_f_K_f_u

    return integration_predictive_mean(data_specific_background_information=data_specific_background_information), \
           integration_predictive_var(data_specific_background_information=data_specific_background_information)


def pred4all_outputs_inputs(my_model, 
                            my_likelihood, 
                            data_inputs, 
                            config, 
                            common_background_information=None, 
                            latent_type=None, # or 'NNEncoder'
                            latent_info=None,
                            approach='mean', 
                            not4visual=True, 
                            n_data4visual=0):
    '''
    Perform inference on all inputs and outputs pairs, two possible approaches: mean and integration. 
    my_model: LVMOGP based model (with VariationalCatLatentVariable or VariationalLatentVariable).
    NOTE: 
    latent_type=NNEncoder : NNEncoderLatentVariable. 
                            latent_info is a MUST in this case.
    latent_type=None : VariationalLatentVariable, VariationalCatLatentVariable.
    '''
    my_model.eval()
    my_likelihood.eval()

    # check
    if latent_type == 'NNEncoder' and latent_info == None:
        ValueError('latent info mush be given if NNEncoder is in use.')

    if not4visual:
        all_index_latent = np.array([[i]*config['n_input'] for i in range(config['n_outputs'])]).reshape(-1).tolist() 
        all_index_input = [i for i in range(config['n_input'])] * config['n_outputs'] 
        len_outputs = len(all_index_latent)

    else:
        assert n_data4visual > 0
        all_index_latent = np.array([[i]*n_data4visual for i in range(config['n_outputs'])]).reshape(-1).tolist() 
        all_index_input = [i for i in range(n_data4visual)] * config['n_outputs'] 
        len_outputs = len(all_index_latent)

    # used to store all predictions
    all_pred_mean = torch.zeros(len_outputs)
    all_pred_var = torch.zeros(len_outputs)

    # --------------- --------------- --------------- --------------- --------------- --------------- --------------- --------------- ---------------

    if approach == 'mean':
        # access the latent variables for all outputs
        if latent_type == None:
            all_mean_outputs = my_model.X.q_mu.data
        elif latent_type == 'NNEncoder':
            all_mean_outputs = my_model.X(latent_info, eval_mode=True)[0].data

        test_mini_batch_size = 1000
        test_continue = True
        test_start_idx = 0
        test_end_idx = test_mini_batch_size

        # iteratively inference
        while test_continue:
            batch_latent = all_mean_outputs[all_index_latent[test_start_idx:test_end_idx]]
            batch_input = data_inputs[all_index_input[test_start_idx:test_end_idx]]
            batch_output = my_likelihood(my_model(batch_latent, batch_input))
            # store predictions (mean and var) for current batch
            all_pred_mean[test_start_idx:test_end_idx] = batch_output.loc.detach().data
            all_pred_var[test_start_idx:test_end_idx] = batch_output.variance.detach().data

            if test_end_idx < len_outputs:
                test_start_idx += test_mini_batch_size
                test_end_idx += test_mini_batch_size
                test_end_idx = min(test_end_idx, len_outputs)
            else:
                test_continue = False

    elif approach == 'integration':
        # iteratively inference
        pre_compute_dict = prepare_pre_computation(my_model, data_inputs)
        for output_idx in trange(config['n_outputs'], leave=True):
            curr_pred_mean, curr_pred_var = integration_prediction_func(test_input=data_inputs,  # curr_input,
                                                                        output_index=output_idx, # curr_latent_index,
                                                                        my_model=my_model,
                                                                        common_background_information=common_background_information,
                                                                        pre_compute_dict=pre_compute_dict,
                                                                        config=config,
                                                                        latent_type=latent_type, # or 'NNEncoder'
                                                                        latent_info=latent_info)
            
            start_id, end_id = output_idx * data_inputs.shape[0], (output_idx + 1) * data_inputs.shape[0]
            all_pred_mean[start_id:end_id] = curr_pred_mean
            all_pred_var[start_id:end_id]  = curr_pred_var + my_likelihood.noise.data
            
    return all_pred_mean, all_pred_var

def evaluate_on_single_output(
        function_index,
        data_inputs,
        data_Y_squeezed,
        ls_of_ls_train_input,
        ls_of_ls_test_input,
        train_sample_idx_ls,
        test_sample_idx_ls,
        all_pred_mean,
        all_pred_var,
        n_data4visual,
        all_pred_mean4visual,
        all_pred_var4visual
    ):
    # Pick the index of the funtion to show
    # function_index = 982

    performance_dirct = {}
    train_input = data_inputs[ls_of_ls_train_input[function_index]]
    train_start = 0
    for i in range(function_index):
        train_start += len(ls_of_ls_train_input[i]) # don't assume every output has the same length of inputs
    train_end = train_start + len(ls_of_ls_train_input[function_index])
    train_target = data_Y_squeezed[train_sample_idx_ls][train_start:train_end]
    train_predict = all_pred_mean[train_sample_idx_ls][train_start:train_end]
    train_rmse_ = (train_target - train_predict).square().mean().sqrt()
    train_nll_ = neg_log_likelihood(train_target, all_pred_mean[train_sample_idx_ls][train_start:train_end], all_pred_var[train_sample_idx_ls][train_start:train_end])
    performance_dirct['train_rmse'] = train_rmse_
    performance_dirct['train_nll'] = train_nll_

    test_input = data_inputs[ls_of_ls_test_input[function_index]]
    test_start = 0
    for j in range(function_index):
        test_start += len(ls_of_ls_test_input[i])
    test_end = test_start + len(ls_of_ls_test_input[function_index])
    test_target = data_Y_squeezed[test_sample_idx_ls][test_start:test_end]
    test_predict = all_pred_mean[test_sample_idx_ls][test_start:test_end]
    test_rmse_ = (test_predict - test_target).square().mean().sqrt()
    test_nll_ = neg_log_likelihood(test_target, all_pred_mean[test_sample_idx_ls][test_start:test_end], all_pred_var[test_sample_idx_ls][test_start:test_end])
    performance_dirct['test_rmse'] = test_rmse_
    performance_dirct['test_nll'] = test_nll_

    gp4visual_start = n_data4visual * function_index
    gp4visual_end = n_data4visual * (function_index + 1)
    gp_pred_mean = all_pred_mean4visual[gp4visual_start:gp4visual_end]
    gp_pred_std = all_pred_var4visual.sqrt()[gp4visual_start:gp4visual_end]

    return train_input, train_target, test_input, test_target, gp_pred_mean, gp_pred_std, performance_dirct

################################################   Metrices  ################################################

def neg_log_likelihood(Target:Tensor, GaussianMean:Tensor, GaussianVar:Tensor):
    '''
    Evaluate negative log likelihood on given i.i.d. targets, where likelihood function is 
    gaussian with mean GaussianMean variance GaussianVar.
    Another name for this metric: negative log predictive density (NLPD). 

    Args:
        Target (torch.Tensor): The target values.
        GaussianMean (torch.Tensor): The mean values of the Gaussian distribution.
        GaussianVar (torch.Tensor): The variance values of the Gaussian distribution.

        
    Return:
        nll: (torch.Tensor): The scalar value of the negative log likelihood
    '''
    assert Target.shape == GaussianMean.shape == GaussianVar.shape

    # numerical stability 
    epsilon = 1e-8
    GaussianVar += epsilon
    
    nll = 0.5 * torch.mean(torch.log(2 * torch.pi * GaussianVar) + (Target - GaussianMean)**2 / GaussianVar)
    return nll

def root_mean_square_error(Target: Tensor, pred: Tensor):
    '''
    Evaluate rmse given target and predictions ... 
    '''
    assert Target.shape == pred.shape
    return (Target - pred).square().mean().sqrt()

def normalised_mean_square_error(Target: Tensor, pred: Tensor):
    '''
    NMSE, follows definition given by Chunchao Ma
    '''
    assert Target.shape == pred.shape
    term1 = (Target - pred).square().mean()
    term2 = (Target - pred.mean()).square().mean()
    return term1 / term2

################################################   Dimensionality reduction  ################################################
from sklearn.decomposition import PCA

def pca_reduction(originalTensor, n_components=2):
    # originalTensor: of shape (# data, # features)

    pca = PCA(n_components=n_components)
    reducedTensor = pca.fit_transform(originalTensor)

    # reducedTensor of shape (# data, n_components)
    return Tensor(reducedTensor)

################################################   Helper function:  plot ################################################

def  plot_traindata_testdata_fittedgp(train_X: Tensor, train_Y: Tensor, test_X: Tensor, test_Y: Tensor, gp_X: Tensor, gp_pred_mean: Tensor, gp_pred_std: Tensor, inducing_points_X: Tensor, n_inducing_C:int=15, title=None, picture_save_path:str='', title_fontsize=20):
    '''
    This is a 1 dim plot: train (corss) and test (dot) data, fitted gp all in the same figure.
    The shadowed area is mean +/- 1.96 gp_pred_std.
    Args:
        train_X: training input locations.
        train_Y: target values for corresponding train_X.
        test_X: testing input locations.
        test_Y: target values for corresponding test_X.
        gp_pred_mean: mean of predictive funtion.
        gp_pred_std: std of predictive function. 

    Return:    
    NOTE: gp_pred_std**2 is formed by two parts, variance from q(f_test) and likelihood variance.
    '''
    # Get numpy 
    train_X_np = train_X.numpy().squeeze()
    train_Y_np = train_Y.numpy().squeeze()
    test_X_np = test_X.numpy().squeeze()
    test_Y_np = test_Y.numpy().squeeze()
    gp_pred_mean_np = gp_pred_mean.numpy().squeeze()
    gp_pred_std_np = gp_pred_std.numpy().squeeze()
    gp_X = gp_X.numpy().squeeze()
    inducing_points_X = inducing_points_X.numpy().squeeze()

    plt.figure(figsize=(8, 6))
    # Plot training data as crosses
    plt.scatter(train_X_np, train_Y_np, c='r', marker='x', label='Training Data', s=110)

    # Plot test data as dots
    plt.scatter(test_X_np, test_Y_np, c='k', marker='o', label='Test Data', alpha=0.5, s=100)

    # Plot inducing points on x axis
    plt.scatter(inducing_points_X, [plt.gca().get_ylim()[0] - 1] * n_inducing_C, color='black', marker='^', label='Inducing Locations')

    # Plot GP predictions as a line
    plt.plot(gp_X, gp_pred_mean_np, 'b', lw=2.5, zorder=9)

    def fill_between_layers(x, mean, std, color, n_layers=2):
        for i in range(1, n_layers + 1):
            alpha = (0.13 / n_layers) * (n_layers - i + 1)  # decrease alpha
            plt.fill_between(x, mean - i * std, mean + i * std, alpha=alpha, color=color)

    n_layers = 2
    fill_between_layers(gp_X, gp_pred_mean_np, 2 * gp_pred_std_np / n_layers, 'blue', n_layers=n_layers)
    # plt.fill_between(gp_X, gp_pred_mean_np - 1.96 * gp_pred_std_np, gp_pred_mean_np + 1.96 * gp_pred_std_np, alpha=0.2, color='k')

    plt.legend()
    title_ = title if title != None else "Train/Test Data and Fitted GP"
    plt.title(title_, fontsize=title_fontsize, fontweight='bold')
    plt.tight_layout()
    # plt.savefig(picture_save_path)
    plt.show()

    return None
