import sys
sys.path.append('/Users/jiangxiaoyu/Desktop/All Projects/Scalable_LVMOGP/')
from code_blocks.mlls.variational_elbo import VariationalELBO
from utils_general import pred4all_outputs_inputs, neg_log_likelihood, prepare_common_background_info
import torch
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, CyclicLR
from tqdm import trange
import matplotlib.pyplot as plt
import time
import random
import numpy as np
from torch import Tensor
import copy
from linear_operator.utils.errors import NotPSDError

################################################   Train and Evaluate LVMOGP Model  ################################################

def sample_ids_of_latent_and_input(ls_of_ls_inputs, batch_size_latent, batch_size_input):
    # TODO: more efficient implementation 
    '''
    Given ls_of_ls_inputs (list of list) containing available inputs for each latent (output).
        * len(ls_of_ls_inputs) == the number of all latents.
        * ls_of_ls_inputs[i] refers to the list of available indices for (i+1) th latent X.
    Args:
        batch_size_latent: the number of elements sampled from all latents.
        batch_size_input: the number of elements sampled from all available inputs.
    Return:
        latent_id_list: selected_latent_ids
        input_id_list: selected_input_ids
    '''

    assert batch_size_latent <= len(ls_of_ls_inputs)
    assert batch_size_input <= len(ls_of_ls_inputs[0]) # here assume ls_of_ls_inputs[0] is a list
    latent_ids = random.sample(range(len(ls_of_ls_inputs)), batch_size_latent)
    latent_id_list = [x for x in latent_ids for _ in range(batch_size_input)]
    input_id_list = []
    for i in latent_ids:
        input_ids = random.sample(ls_of_ls_inputs[i], batch_size_input)
        for j in input_ids:
            input_id_list.append(j)
    assert len(latent_id_list) == len(input_id_list) == batch_size_latent*batch_size_input

    return latent_id_list, input_id_list

def inhomogeneous_index_of_batch_Y(batch_index_latent, batch_index_input, n_latent, n_input):
    """
    Inhomogeneously get set of indices for Y given latent and input indicies. 
    Args:
        batch_index_latent and batch_index_input jointly determine the position of the corresponding element in Y.
        n_latent: number of elements in X (not in use)
        n_input: number of elements in C
    Return:
        List of indices of elements in Y, which of length len(batch_index_latent), which also equal to len(batch_index_input).
    """
    assert len(batch_index_latent) == len(batch_index_input)
    batch_index_Y = []
    for (index_X, index_C) in zip(batch_index_latent, batch_index_input):
        batch_index_Y.append(index_X * n_input + index_C)
    
    return batch_index_Y


def train_and_eval_lvmogp_model(
        data_inputs,
        data_Y_squeezed,
        ls_of_ls_train_input,
        ls_of_ls_test_input,
        my_model,
        my_likelihood,
        config,
        latent_info,       # Must for NNEncoder
        results_folder_path, 
        train_sample_idx_ls, 
        test_sample_idx_ls,
        means=None,
        stds=None):
    
    number_all = config['n_outputs'] * config['n_input_train']
    results_txt = f'{results_folder_path}/results.txt'
    with open(results_txt, 'w') as file:
        file.write(f'Random seed: {config["random_seed"]}\n')

    ''' -------------------------------------- Training --------------------------------------'''

    # optimizer and scheduler
    optimizer = torch.optim.Adam([
            {'params': my_model.parameters()},
            {'params': my_likelihood.parameters()}
        ], lr=config['lr'])

    # TODO Try different types of schedulers ... 
    if 'scheduler' not in config or config['scheduler'] == CyclicLR: # Default choice
        step_size_up = config['step_size_up'] if 'step_size_up' in config else 30
        scheduler = CyclicLR(optimizer, base_lr=config['lr'], max_lr=0.2*config['lr'], step_size_up=step_size_up, mode='triangular', cycle_momentum=False)

    elif config['scheduler'] == StepLR:
        step_size = config['step_size'] if 'step_size' in config else 20
        gamma = config['gamma'] if 'gamma' in config else 0.95
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma) 
        
    # scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=0.2*config['lr'])

    loss_list = []
    iterator = trange(config['n_iterations'], leave=True)

    my_model.train()
    my_likelihood.train()
    start_time = time.time()
    min_loss_value = 1e+10
    for i in iterator: 

        batch_index_latent, batch_index_input = sample_ids_of_latent_and_input(ls_of_ls_train_input, 
                                                                               batch_size_latent = config['batch_size_latent'], 
                                                                               batch_size_input = config['batch_size_input'])
        optimizer.zero_grad()
        
        ### computing loss = negative variational elbo = - (log_likelihood - kl_divergence - added_loss)
        loss = 0.0
        for _ in range(config['num_latent_MC']):
            sample_batch_latent = my_model.sample_latent_variable(batch_idx=batch_index_latent, latent_info=latent_info)
            sample_batch_input = data_inputs[batch_index_input]
            output_batch = my_model(sample_batch_latent, sample_batch_input) # q(f)
            batch_index_Y = inhomogeneous_index_of_batch_Y(batch_index_latent, batch_index_input, config['n_outputs'], config['n_input'])
        
            ## log-likelihood term
            log_likelihood_batch = my_likelihood.expected_log_prob(input=output_batch, target=data_Y_squeezed[batch_index_Y]).sum(-1).div(output_batch.event_shape[0])
            loss += -log_likelihood_batch

            ## x_kl term
            added_loss = torch.zeros_like(log_likelihood_batch)
            for added_loss_term in my_model.added_loss_terms():
                # ONLY one added loss here, which is KL in latent space
                added_loss.add_(config['alpha'] * added_loss_term.loss())
            loss += added_loss

        ## KL divergence term
        kl_divergence = my_model.variational_strategy.kl_divergence().div(number_all / config['beta'])
        loss = loss / config['num_latent_MC'] + kl_divergence
        loss.backward()

        loss_value = loss.item()

        # store model every 50 iterations
        if i > 100 and i % 50 == 0 and loss_value < min_loss_value:
            print(f'A new model is stored, with current loss value {loss_value}.')
            torch.save(my_model.state_dict(), f'{results_folder_path}/min_model.pth')
            torch.save(my_likelihood.state_dict(), f'{results_folder_path}/min_likelihood.pth')  

            min_loss_value = loss_value

        loss_list.append(loss_value)
        iterator.set_description('Loss: ' + str(float(np.round(loss_value, 3))) + ", iter no: " + str(i))

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(my_model.parameters(), config['model_max_grad_norm'])
        torch.nn.utils.clip_grad_norm_(my_likelihood.parameters(), config['likeli_max_grad_norm'])

        optimizer.step()
        scheduler.step()

    end_time = time.time()
    total_training_time = end_time - start_time

    with open(results_txt, 'a') as file:
        file.write(f'Training time: {total_training_time:.2f}\n')

    # plot training losses
    _loss_list = list(np.array(loss_list)[np.array(loss_list) < 1000]) # remove too large losses, i.e. above 3
    plt.plot(_loss_list)
    plt.savefig(f'{results_folder_path}/filtered_training_loss.png')

    # save model
    torch.save(my_model.state_dict(), f'{results_folder_path}/model.pth')
    torch.save(my_likelihood.state_dict(), f'{results_folder_path}/likelihood.pth') 
    
    print('Finish Training! Start Testing!')

    ''' -------------------------------------- Testing --------------------------------------'''

    my_model.eval()
    my_likelihood.eval() 

    for model_type in ['final', 'min']:
        for approach in ['mean', 'integration']: 
            print(f'We are on {model_type} model, with {approach} approach ... ')
            
            if model_type == 'final': 
                curr_model, curr_likelihood = my_model, my_likelihood
            elif model_type == 'min':
                my_model.load_state_dict(torch.load(f'{results_folder_path}/min_model.pth'))
                my_likelihood.load_state_dict(torch.load(f'{results_folder_path}/min_likelihood.pth'))
                curr_model, curr_likelihood = my_model, my_likelihood

            common_background_information = None
            if approach == 'integration': common_background_information = prepare_common_background_info(curr_model, config)

            all_pred_mean_, all_pred_var_ = pred4all_outputs_inputs( my_model=curr_model,
                                                                     my_likelihood=curr_likelihood,
                                                                     data_inputs=data_inputs,
                                                                     config=config,
                                                                     approach=approach,
                                                                     common_background_information=common_background_information,
                                                                     latent_type='NNEncoder' if config['NNEncoder'] else None,
                                                                     latent_info=latent_info)

            # Evaluate on normalized test points
            train_data_predict_ = all_pred_mean_[train_sample_idx_ls]
            train_rmse_ = (train_data_predict_ - data_Y_squeezed[train_sample_idx_ls]).square().mean().sqrt()

            test_data_predict_ = all_pred_mean_[test_sample_idx_ls]
            test_rmse_ = (test_data_predict_ - data_Y_squeezed[test_sample_idx_ls]).square().mean().sqrt()
    
            train_nll_ = neg_log_likelihood(Target=data_Y_squeezed[train_sample_idx_ls], GaussianMean=all_pred_mean_[train_sample_idx_ls], GaussianVar=all_pred_var_[train_sample_idx_ls])
            test_nll_ = neg_log_likelihood(Target=data_Y_squeezed[test_sample_idx_ls], GaussianMean=all_pred_mean_[test_sample_idx_ls], GaussianVar=all_pred_var_[test_sample_idx_ls])

            with open(results_txt, 'a') as file:
                file.write(f'Evaluation results for {model_type} model with {approach} approach are: \n')
                file.write(f'train rmse is {train_rmse_:.4g} \ntest rmse is {test_rmse_:.4g} \ntrain nll is {train_nll_:.4g} \ntest nll is {test_nll_:.4g} \n')

            # Evaluate on original test points
            if config['dataset_type'] != 'synthetic_regression':
                orig_data_Y_squeezed = (data_Y_squeezed.reshape(config['n_outputs'] , config['n_input']) * stds.unsqueeze(1) + means.unsqueeze(1)).reshape(-1)    
                orig_all_pred_mean = (all_pred_mean_.reshape(config['n_outputs'] , config['n_input']) * stds.unsqueeze(1) + means.unsqueeze(1)).reshape(-1)
                orig_all_pred_var = (all_pred_var_.reshape(config['n_outputs'] , config['n_input']) * (stds**2).unsqueeze(1)).reshape(-1)
                
                train_data_predict = orig_all_pred_mean[train_sample_idx_ls]
                train_rmse = (train_data_predict - orig_data_Y_squeezed[train_sample_idx_ls]).square().mean().sqrt()

                test_data_predict = orig_all_pred_mean[test_sample_idx_ls]
                test_rmse = (test_data_predict - orig_data_Y_squeezed[test_sample_idx_ls]).square().mean().sqrt()

                train_nll = neg_log_likelihood(Target=orig_data_Y_squeezed[train_sample_idx_ls], 
                                                GaussianMean=orig_all_pred_mean[train_sample_idx_ls], 
                                                GaussianVar=orig_all_pred_var[train_sample_idx_ls])
                
                test_nll = neg_log_likelihood(Target=orig_data_Y_squeezed[test_sample_idx_ls], 
                                            GaussianMean=orig_all_pred_mean[test_sample_idx_ls], 
                                            GaussianVar=orig_all_pred_var[test_sample_idx_ls])
                
                # TODO: avoid following redundant code.
                with open(results_txt, 'a') as file:
                    file.write('On original dataset:\n')
                    file.write(f'Evaluation results for {model_type} model with {approach} approach are: \n')
                    file.write(f'train rmse is {train_rmse:.4g} \ntest rmse is {test_rmse:.4g} \ntrain nll is {train_nll:.4g} \ntest nll is {test_nll:.4g} \n')
            else: 
                pass
    
################################################   Train and Evaluate Multi-IndepSVGP Model  ################################################

def mini_batching_sampling_func(num_inputs, batch_size):
    assert batch_size <= num_inputs
    idx_list = random.sample(range(num_inputs), batch_size)
    return idx_list

def save_model_and_likelihoods(multi_variational_igp, filename):
    state_dicts = {
        'models': [model.state_dict() for model in multi_variational_igp.models],
        'likelihoods': [likelihood.state_dict() for likelihood in multi_variational_igp.likelihoods]
    }
    torch.save(state_dicts, filename)
                
def train_and_eval_multiIndepSVGP_model(
        data_inputs,
        data_Y_squeezed,
        ls_of_ls_train_input,
        ls_of_ls_test_input,
        train_sample_idx_ls,
        test_sample_idx_ls,
        my_model,
        config,
        results_folder_path,
        means,
        stds):
    
    '''
    First 6 arguments are returned from functions in prepare_data.py, these data are naturally designed for MOGP.
    To train multiple IGPs, we need to reconstruct datasets based on them.
    '''
    results_txt = f'{results_folder_path}/results.txt'
    with open(results_txt, 'w') as file:
        file.write(f'Results for random seed: {config["random_seed"]}\n')

    # The following lists consist of datasets for all outputs.
    list_train_X, list_train_Y = [], [] 
    list_test_X, list_test_Y = [], []

    # split data_Y_squeezed into train/test part. NOTE: that's train/test target data for all outputs.
    data_Y_train_squeezed = data_Y_squeezed[train_sample_idx_ls]
    data_Y_test_squeezed = data_Y_squeezed[test_sample_idx_ls]

    n_input_test = config['n_input'] - config['n_input_train']
    ##### ------------------------------------------------------------------------
    for i in range(config['n_outputs']):
        # start and end for current output, idx used to pick data for only current output
        idgp_train_start = i * config['n_input_train']
        idgp_train_end = idgp_train_start + config['n_input_train']

        idgp_test_start = i * n_input_test
        idgp_test_end = idgp_test_start + n_input_test

        # training data for current output
        train_X = data_inputs[ls_of_ls_train_input[i]]
        train_Y = data_Y_train_squeezed[idgp_train_start:idgp_train_end]
        assert train_X.shape ==  train_Y.shape == torch.Size([config['n_input_train']])
        list_train_X.append(train_X)
        list_train_Y.append(train_Y)

        # testing data for current output
        test_X = data_inputs[ls_of_ls_test_input[i]]
        test_Y = data_Y_test_squeezed[idgp_test_start:idgp_test_end]
        assert test_X.shape ==  test_Y.shape == torch.Size([n_input_test])
        list_test_X.append(test_X)
        list_test_Y.append(test_Y)
    
    #### ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 
    ls_training_time = []

    ##### Train and Evaluate IGPs one by one
    
    '''########  Training  ########'''

    for j in range(config['n_outputs']):

        # current train_X and train_Y
        train_X = list_train_X[j]
        train_Y = list_train_Y[j]

        curr_model = my_model.get_model(j)
        curr_likelihood = my_model.get_likelihood(j)

        curr_model.train()
        curr_likelihood.train()
        
        curr_optimizer = torch.optim.Adam([
            {'params': curr_model.parameters()},
            {'params': curr_likelihood.parameters()},
        ], lr=config['lr'])

        # TODO Try different types of schedulers ... 
        if 'scheduler' not in config or config['scheduler'] == CyclicLR: # Default choice
            step_size_up = config['step_size_up'] if 'step_size_up' in config else 30
            curr_scheduler = CyclicLR(curr_optimizer, base_lr=config['lr'], max_lr=0.2*config['lr'], step_size_up=step_size_up, mode='triangular', cycle_momentum=False)

        elif config['scheduler'] == StepLR:
            step_size = config['step_size'] if 'step_size' in config else 20
            gamma = config['gamma'] if 'gamma' in config else 0.95
            curr_scheduler = StepLR(curr_optimizer, step_size=step_size, gamma=gamma) 
        # curr_scheduler = CyclicLR(curr_optimizer, base_lr=config['lr'], max_lr=0.2*config['lr'], step_size_up=config['step_size_up'], mode='triangular', cycle_momentum=False)

        mll = VariationalELBO(curr_likelihood, curr_model, num_data=train_Y.size(0))

        # start training!
        ls_train_loss = []
        iterator = trange(config['n_iterations'], leave=True)

        curr_start_time = time.time()
        for i in iterator:
            curr_optimizer.zero_grad()
            mini_batch_idx = mini_batching_sampling_func(num_inputs=train_X.shape[0], batch_size=config['batch_size_input'])
            
            try:
                output_pred = curr_model(train_X[mini_batch_idx])
                loss = -mll(output_pred, train_Y[mini_batch_idx])
                ls_train_loss.append(loss.item())
                iterator.set_description( 'Training '+ str(j) + 'th Model; '+ 'Loss: ' + str(float(np.round(loss.item(),3))) + ", iter no: " + str(i))
                loss.backward()

                # clip gradients
                torch.nn.utils.clip_grad_norm_(curr_model.parameters(), config['model_max_grad_norm'])
                torch.nn.utils.clip_grad_norm_(curr_likelihood.parameters(), config['likeli_max_grad_norm'])

                curr_optimizer.step()
                curr_scheduler.step()
            
            except NotPSDError: 
                print('Encounting NotPSDError, stop training current indepSVGP!')
                with open(results_txt, 'a') as file:
                    file.write(f'{j}th output encounted NotPSDError! \n')
                break

        curr_end_time = time.time()
        curr_total_training_time = curr_end_time - curr_start_time
        ls_training_time.append(curr_total_training_time)

    total_time = np.array(ls_training_time).sum()
    with open(results_txt, 'a') as file:
        file.write(f'Total time: {total_time}\n')

    save_model_and_likelihoods(my_model, f'{results_folder_path}/MultiIGPs_models_and_likelihoods.pth')

    '''########  Testing  ########'''

    train_error_square_sum, test_error_square_sum, train_nll_sum, test_nll_sum, train_error_length, test_error_length = 0., 0., 0., 0., 0., 0.
    original_train_error_square_sum, original_test_error_square_sum, original_train_nll_sum, original_test_nll_sum = 0., 0., 0., 0.

    for j in range(config['n_outputs']):

        curr_model = my_model.get_model(j)
        curr_likelihood = my_model.get_likelihood(j)

        # Inference for train and test data
        curr_train_output_dist = curr_likelihood(curr_model(list_train_X[j]))
        curr_test_output_dist  = curr_likelihood(curr_model(list_test_X[j]))

        # RMSE on normalized data
        curr_train_suqare_errors = (curr_train_output_dist.loc.detach() - list_train_Y[j]).square()
        curr_test_square_errors = (curr_test_output_dist.loc.detach() - list_test_Y[j]).square()
        train_error_square_sum += curr_train_suqare_errors.sum()
        test_error_square_sum  += curr_test_square_errors.sum()

        # NLL on normalized data
        train_nll_ = neg_log_likelihood(list_train_Y[j], 
                                        curr_train_output_dist.loc.detach(), 
                                        curr_train_output_dist.variance.detach())
        
        test_nll_ = neg_log_likelihood(list_test_Y[j], 
                                       curr_test_output_dist.loc.detach(), 
                                       curr_test_output_dist.variance.detach())
        
        train_nll_sum += train_nll_ * len(list_train_Y[j])
        test_nll_sum  += test_nll_ *  len(list_test_Y[j])

        ''' ------------ ------------ ------------ ------------ ------------ ------------ ------------'''
        if means != None:
            original_train_Y_j = (list_train_Y[j] * stds[j].item() )+ means[j].item()
            original_test_Y_j = (list_test_Y[j] * stds[j].item() )+ means[j].item()

            original_curr_train_pred_loc = curr_train_output_dist.loc.detach() * stds[j].item() + means[j].item()
            original_curr_test_pred_loc = curr_test_output_dist.loc.detach() * stds[j].item() + means[j].item()
            original_curr_train_pred_var = curr_train_output_dist.variance.detach() * (stds[j].item()**2)
            original_curr_test_pred_var = curr_test_output_dist.variance.detach() * (stds[j].item()**2)

            # RMSE on original data

            original_curr_train_suqare_errors = (original_curr_train_pred_loc - original_train_Y_j).square()
            original_curr_test_square_errors = (original_curr_test_pred_loc - original_test_Y_j).square()
            original_train_error_square_sum += original_curr_train_suqare_errors.sum()
            original_test_error_square_sum  += original_curr_test_square_errors.sum()

            # NLL on original data
            original_train_nll_ = neg_log_likelihood(original_train_Y_j, 
                                            original_curr_train_pred_loc, 
                                            original_curr_train_pred_var)
            
            original_test_nll_ = neg_log_likelihood(original_test_Y_j, 
                                        original_curr_test_pred_loc, 
                                        original_curr_test_pred_var)
            
            original_train_nll_sum += original_train_nll_ * len(list_train_Y[j])
            original_test_nll_sum  += original_test_nll_ *  len(list_test_Y[j])

        ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 
        train_error_length += len(list_train_Y[j])
        test_error_length += len(list_test_Y[j])

    # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
    with open(results_txt, 'a') as file:
        file.write('Evaluation on normalized data:\n')
        file.write(f'global train rmse is {((train_error_square_sum / train_error_length).sqrt()):.4g}\n')
        file.write(f'global test rmse is {((test_error_square_sum / test_error_length).sqrt()):.4g}\n')
        file.write(f'global train nll is {(train_nll_sum / train_error_length):.4g}\n')
        file.write(f'global test nll is {(test_nll_sum / test_error_length):.4g}\n')
        if means != None:
            file.write('Evaluation on original data:\n')
            file.write(f'global train rmse is {((original_train_error_square_sum / train_error_length).sqrt()):.4g}\n')
            file.write(f'global test rmse is {((original_test_error_square_sum / test_error_length).sqrt()):.4g}\n')
            file.write(f'global train nll is {(original_train_nll_sum / train_error_length):.4g}\n')
            file.write(f'global test nll is {(original_test_nll_sum / test_error_length):.4g}\n')

################################################   Init Model and Likelihood : Helper Function  ################################################
def helper_init_latent_kernel(my_model, gplvm_model):

    '''
    Use kernel hyper-parameters trained from gplvm to initialize lvmogp model ... 
    The default choice for latent kernel is scaled RBF Kernel
    '''

    my_model.covar_module_latent.outputscale = gplvm_model.covar_module.outputscale.data
    my_model.covar_module_latent.base_kernel.lengthscale = gplvm_model.covar_module.base_kernel.lengthscale.data
    return my_model

def helper_init_model_and_likeli(my_model, config, my_likelihood=None, only_init_model=False):
        
    if config['input_kernel_type'] == 'Periodic_times_Scale_RBF':
        my_model.covar_module_input.kernels[0].period_length = config['1stKernel_peirod_init']
        my_model.covar_module_input.kernels[0].lengthscale = config['1stKernel_lengthscale_init']
        my_model.covar_module_input.kernels[1].base_kernel.lengthscale = config['2ndKernel_lengthscale_init']
        my_model.covar_module_input.kernels[1].outputscale = config['2ndKernel_outputscale_init']
    
    if config['input_kernel_type'] == 'Scale_Matern52_plus_Scale_PeriodicInputsMatern52':
        my_model.covar_module_input.kernels[0].outputscale = config['1stKernel_outputscale_init']
        my_model.covar_module_input.kernels[0].base_kernel.lengthscale = config['1stKernel_lengthscale_init']
        my_model.covar_module_input.kernels[1].outputscale = config['2ndKernel_outputscale_init']
        my_model.covar_module_input.kernels[1].base_kernel.lengthscale = config['2ndKernel_lengthscale_init']
        my_model.covar_module_input.kernels[1].base_kernel.period_length = config['2ndKernel_period_init']
        
    if config['input_kernel_type'] == 'PeriodicInputsRBFKernel_times_Scale_RBF':
        my_model.covar_module_input.kernels[0].period_length = config['1stKernel_period_init']
        my_model.covar_module_input.kernels[0].lengthscale = config['1stKernel_lengthscale_init']
        my_model.covar_module_input.kernels[1].outputscale = config['2ndKernel_outputscale_init']
        my_model.covar_module_input.kernels[1].base_kernel.lengthscale = config['2ndKernel_lengthscale_init']
    
    if config['input_kernel_type'] == 'Scale_RBF':
        # using default init ... 
        pass

    if not only_init_model:
        # Init inducing points in input space
        my_model.variational_strategy.inducing_points_input.data = Tensor(np.linspace(config['init_inducing_input_LB'], config['init_inducing_input_UB'], config['n_inducing_input']).reshape(-1, 1)).to(torch.double) 
        
        # Init noise scale in likelihood
        my_likelihood.noise = config['init_likelihood_noise']

    return my_model, my_likelihood