import sys
sys.path.append('/Users/jiangxiaoyu/Desktop/All Projects/Scalable_LVMOGP/')
from code_blocks.mlls.variational_elbo import VariationalELBO
from code_blocks.mlls.sum_variational_elbo import SumVariationalELBO
from code_blocks.utils.param_tracker import param_extractor1, ParamTracker
from utils_general import (
    pred4all_outputs_inputs, 
    neg_log_likelihood, 
    root_mean_square_error,
    normalised_mean_square_error,
    prepare_common_background_info
)
import torch
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, CyclicLR
from tqdm import trange
import matplotlib.pyplot as plt
import time
import random
import numpy as np
from linear_operator.utils.errors import NotPSDError
import gpytorch

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

    latent_ids = random.choices(range(len(ls_of_ls_inputs)), k=batch_size_latent)
    latent_id_list = [x for x in latent_ids for _ in range(batch_size_input)]
    
    # Use list comprehension for input_id_list
    input_id_list = [j for i in latent_ids for j in random.choices(ls_of_ls_inputs[i], k=batch_size_input)]

    # Recall, in variational strategy, two list of 'inputs' jointly determine the corresponding target. 
    assert len(latent_id_list) == len(input_id_list) == batch_size_latent * batch_size_input

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
        stds=None,
        args=None):
    
    number_all_train_data = config['number_all_train_data'] if 'number_all_train_data' in config else config['n_outputs'] * config['n_input_train']
    correction_term = (config['n_outputs'] * config['n_input_train'] ) / number_all_train_data # = 1 if all outputs have same number of training inputs.

    results_txt = f'{results_folder_path}/results.txt'
    with open(results_txt, 'w') as file:
        file.write(f'Random seed: {args.random_seed}\n')

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

    # Store the value of terms in the loss during training
    loss_list, log_likelihood_term_list, latent_kl_term_list, variational_kl_term_list = [], [], [], []

    param_tracker = ParamTracker(param_extractor=param_extractor1)
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
        loss, log_likelihood_term, latent_kl_term = 0.0, 0.0, 0.0
        for _ in range(config['num_latent_MC']):
            sample_batch_latent = my_model.sample_latent_variable(batch_idx=batch_index_latent, latent_info=latent_info)
            sample_batch_input = data_inputs[batch_index_input]
            output_batch = my_model(sample_batch_latent, sample_batch_input) # q(f)
            batch_index_Y = inhomogeneous_index_of_batch_Y(batch_index_latent, batch_index_input, config['n_outputs'], config['n_input'])
        
            ## log-likelihood term
            log_likelihood_batch = my_likelihood.expected_log_prob(input=output_batch, target=data_Y_squeezed[batch_index_Y]).sum(-1).div(output_batch.event_shape[0])
            loss += -log_likelihood_batch
            log_likelihood_term += -log_likelihood_batch.detach().item()

            ## x_kl term
            added_loss = torch.zeros_like(log_likelihood_batch)
            for added_loss_term in my_model.added_loss_terms():
                # ONLY one added loss here, which is KL in latent space
                added_loss.add_(correction_term * config['alpha'] * added_loss_term.loss())
            loss += added_loss
            latent_kl_term += added_loss.detach().item()

        ## KL divergence term
        kl_divergence = my_model.variational_strategy.kl_divergence().div(number_all_train_data / config['beta'])
        loss = loss / config['num_latent_MC'] + kl_divergence
        variational_kl_term = kl_divergence.detach().item()
        loss.backward()

        loss_value = loss.item()

        # store model every 50 iterations
        if i > 100 and i % 50 == 0 and loss_value < min_loss_value:
            print(f'A new model is stored, with current loss value {loss_value}.')
            torch.save(my_model.state_dict(), f'{results_folder_path}/min_model.pth')
            torch.save(my_likelihood.state_dict(), f'{results_folder_path}/min_likelihood.pth')  

            min_loss_value = loss_value

        loss_list.append(loss_value)
        log_likelihood_term_list.append(log_likelihood_term / config['num_latent_MC'])
        latent_kl_term_list.append(latent_kl_term / config['num_latent_MC'])
        variational_kl_term_list.append(variational_kl_term)

        iterator.set_description('Loss: ' + str(float(np.round(loss_value, 3))) + ", iter no: " + str(i))

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(my_model.parameters(), config['model_max_grad_norm'])
        torch.nn.utils.clip_grad_norm_(my_likelihood.parameters(), config['likeli_max_grad_norm'])

        optimizer.step()
        scheduler.step()

        param_tracker.update(my_model, my_likelihood)

    end_time = time.time()
    total_training_time = end_time - start_time

    with open(results_txt, 'a') as file:
        file.write(f'Training time: {total_training_time:.2f}\n')

    ####### making some plots ... 
    
    # plot how (hyper) parameters change during training 

    # plot the value of the components in the loss    
    plt.plot(log_likelihood_term_list)
    plt.savefig(f'{results_folder_path}/training_log_likelihood_term.png')
    plt.close()

    plt.plot(latent_kl_term_list)
    plt.savefig(f'{results_folder_path}/training_latent_kl_term_list.png')
    plt.close()

    plt.plot(variational_kl_term_list)
    plt.savefig(f'{results_folder_path}/training_variational_kl_term_list.png')
    plt.close()

    # plot training losses
    plt.plot(loss_list)
    plt.savefig(f'{results_folder_path}/training_loss.png')
    plt.close()

    param_tracker.plot(results_folder_path)
    
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
            train_rmse_ = root_mean_square_error(Target=data_Y_squeezed[train_sample_idx_ls], pred=train_data_predict_)
            # train_rmse_ = (train_data_predict_ - data_Y_squeezed[train_sample_idx_ls]).square().mean().sqrt()

            test_data_predict_ = all_pred_mean_[test_sample_idx_ls]
            test_rmse_ = root_mean_square_error(Target=data_Y_squeezed[test_sample_idx_ls], pred=test_data_predict_)
            # test_rmse_ = (test_data_predict_ - data_Y_squeezed[test_sample_idx_ls]).square().mean().sqrt()

            train_nmse_ = normalised_mean_square_error(Target=data_Y_squeezed[train_sample_idx_ls], pred=train_data_predict_)
            test_nmse_ = normalised_mean_square_error(Target=data_Y_squeezed[test_sample_idx_ls], pred=test_data_predict_)

            train_nll_ = neg_log_likelihood(Target=data_Y_squeezed[train_sample_idx_ls], GaussianMean=all_pred_mean_[train_sample_idx_ls], GaussianVar=all_pred_var_[train_sample_idx_ls])
            test_nll_ = neg_log_likelihood(Target=data_Y_squeezed[test_sample_idx_ls], GaussianMean=all_pred_mean_[test_sample_idx_ls], GaussianVar=all_pred_var_[test_sample_idx_ls])

            with open(results_txt, 'a') as file:
                file.write('On NORMALIZED dataset:\n')
                file.write(f'Evaluation results for {model_type} model with {approach} approach are: \n')
                file.write(f'train rmse is {train_rmse_:.4g} \n test rmse is {test_rmse_:.4g} \n train nmse is {train_nmse_:.4g} \n test nmse is {test_nmse_:.4g} \n train nll is {train_nll_:.4g} \n test nll is {test_nll_:.4g} \n')

            # Evaluate on original test points
            # if config['dataset_type'] != 'synthetic_regression':
            if means != None:
                orig_data_Y_squeezed = (data_Y_squeezed.reshape(config['n_outputs'] , config['n_input']) * stds.unsqueeze(1) + means.unsqueeze(1)).reshape(-1)    
                orig_all_pred_mean = (all_pred_mean_.reshape(config['n_outputs'] , config['n_input']) * stds.unsqueeze(1) + means.unsqueeze(1)).reshape(-1)
                orig_all_pred_var = (all_pred_var_.reshape(config['n_outputs'] , config['n_input']) * (stds**2).unsqueeze(1)).reshape(-1)
                
                train_data_predict = orig_all_pred_mean[train_sample_idx_ls]
                train_rmse = root_mean_square_error(Target=orig_data_Y_squeezed[train_sample_idx_ls], pred=train_data_predict)
                # train_rmse = (train_data_predict - orig_data_Y_squeezed[train_sample_idx_ls]).square().mean().sqrt()

                test_data_predict = orig_all_pred_mean[test_sample_idx_ls]
                test_rmse = root_mean_square_error(Target=orig_data_Y_squeezed[test_sample_idx_ls], pred=test_data_predict)
                # test_rmse = (test_data_predict - orig_data_Y_squeezed[test_sample_idx_ls]).square().mean().sqrt()

                train_nmse = normalised_mean_square_error(Target=orig_data_Y_squeezed[train_sample_idx_ls], pred=train_data_predict)
                test_nmse = normalised_mean_square_error(Target=orig_data_Y_squeezed[test_sample_idx_ls], pred=test_data_predict)

                train_nll = neg_log_likelihood(Target=orig_data_Y_squeezed[train_sample_idx_ls], 
                                                GaussianMean=orig_all_pred_mean[train_sample_idx_ls], 
                                                GaussianVar=orig_all_pred_var[train_sample_idx_ls])
                
                test_nll = neg_log_likelihood(Target=orig_data_Y_squeezed[test_sample_idx_ls], 
                                            GaussianMean=orig_all_pred_mean[test_sample_idx_ls], 
                                            GaussianVar=orig_all_pred_var[test_sample_idx_ls])
                
                # TODO: try to avoid following redundant code.
                with open(results_txt, 'a') as file:
                    file.write('On ORIGINAL dataset:\n')
                    file.write(f'Evaluation results for {model_type} model with {approach} approach are: \n')
                    file.write(f'train rmse is {train_rmse:.4g} \n test rmse is {test_rmse:.4g} \n train nmse is {train_nmse:.4g} \n test nmse is {test_nmse:.4g} \n train nll is {train_nll:.4g} \n test nll is {test_nll:.4g} \n')
            else: 
                pass
    
################################################   Train and Evaluate Multi-IndepSVGP Model  ################################################

def mini_batching_sampling_func(num_inputs, batch_size):
    assert batch_size <= num_inputs
    idx_list = random.sample(range(num_inputs), batch_size)
    return idx_list

# helper function to store list of indepSVGP models and list of their likelihoods.
def save_model_and_likelihoods(multi_variational_igp, filename):
    state_dicts = {
        'models': [model.state_dict() for model in multi_variational_igp.models],
        'likelihoods': [likelihood.state_dict() for likelihood in multi_variational_igp.likelihoods]
    }
    torch.save(state_dicts, filename)

def train_and_eval_multiIndepSVGP_parallel(
        data_inputs,
        data_Y_squeezed,
        ls_of_ls_train_input,
        ls_of_ls_test_input,
        train_sample_idx_ls,
        test_sample_idx_ls,
        my_multiIGP_parallel, # Multi_Variational_IGP_parallel
        config,
        results_folder_path,
        means,
        stds,
        args):
    
    results_txt = f'{results_folder_path}/results.txt'
    with open(results_txt, 'w') as file:
        file.write(f'Results for random seed: {args.random_seed}\n')

    ########    Prepare Datasets for All Outputs    ########
        
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
    
    ########    Training    ########

    my_model = my_multiIGP_parallel.ModelList
    my_likelihood = my_multiIGP_parallel.LikelihoodList

    my_model.train()
    my_likelihood.train()

    my_optimizer = torch.optim.Adam(my_model.parameters(), lr=config['lr'])

    if 'scheduler' not in config or config['scheduler'] == CyclicLR: # Default choice
        step_size_up = config['step_size_up'] if 'step_size_up' in config else 30
        my_scheduler = CyclicLR(my_optimizer, base_lr=config['lr'], max_lr=0.2*config['lr'], step_size_up=step_size_up, mode='triangular', cycle_momentum=False)

    elif config['scheduler'] == StepLR:
        step_size = config['step_size'] if 'step_size' in config else 20
        gamma = config['gamma'] if 'gamma' in config else 0.95
        my_scheduler = StepLR(my_optimizer, step_size=step_size, gamma=gamma) 

    # curr_scheduler = CyclicLR(curr_optimizer, base_lr=config['lr'], max_lr=0.2*config['lr'], step_size_up=config['step_size_up'], mode='triangular', cycle_momentum=False)
    
    num_data_list = [len(_list) for _list in ls_of_ls_train_input]
    mll = SumVariationalELBO(my_model, num_data_list)

    iterator = trange(config['n_iterations'], leave=True)
    start_time = time.time()
    for i in iterator: # Mini-batch training ... 
        # TODO: only works when all outputs has the same number of data points
        mini_batch_idx = random.choices(range(list_train_X[0].shape[0]), k=config['batch_size_input'])
        list_train_input_all_outputs = [list_train_X[j][mini_batch_idx] for j in range(config['n_outputs'])]
        list_train_target_all_outputs = [list_train_Y[j][mini_batch_idx] for j in range(config['n_outputs'])]

        my_optimizer.zero_grad()
        output = my_model(*list_train_input_all_outputs)
        loss = -mll(output, list_train_target_all_outputs)
        loss.backward()
        iterator.set_description('Loss: ' + str(float(np.round(loss.item(),3))) + ", iter no: " + str(i) + '/' + str(config['n_iterations']))

        # clip gradients
        # torch.nn.utils.clip_grad_norm_(my_model.parameters(), config['model_max_grad_norm'])

        my_optimizer.step()
        my_scheduler.step()

    end_time = time.time()
    total_training_time = end_time - start_time

    with open(results_txt, 'a') as file:
        file.write(f'Total time: {total_training_time}\n')

    torch.save(my_model, f'{results_folder_path}/MultiIGPs_parallel_models.pth')

    ########    Testing    ########

    my_model.eval()
    my_likelihood.eval()

    with torch.no_grad(): # gpytorch.settings.fast_pred_var()
        list_test_input_all_outputs = [list_test_X[j] for j in range(config['n_outputs'])]
        predictions = my_likelihood(*my_model(*list_test_input_all_outputs))
    
    tensor_test_target_all_outputs = torch.tensor([list_test_Y[j] for j in range(config['n_outputs'])]).reshape(-1)
    tensor_test_pred_mean_all_outputs = torch.tensor([prediction.mean.tolist() for prediction in predictions])
    tensor_test_pred_std_all_outputs = torch.tensor([prediction.stddev.tolist() for prediction in predictions])

    test_rmse = root_mean_square_error(tensor_test_target_all_outputs, tensor_test_pred_mean_all_outputs)

    test_nll = neg_log_likelihood(tensor_test_target_all_outputs, 
                                  tensor_test_pred_mean_all_outputs, 
                                  tensor_test_pred_std_all_outputs.square())
    
    test_nmse = normalised_mean_square_error(tensor_test_target_all_outputs, tensor_test_pred_mean_all_outputs)

    with open(results_txt, 'a') as file:
        file.write('Evaluation on normalized data:\n')
        file.write(f'global test rmse is {test_rmse:.4g}\n')
        file.write(f'global test nmse is {(test_nmse):.4g}\n')
        file.write(f'global test nll is {test_nll:.4g}\n')
    
    if means != None:
        # TODO
        pass
        # We also evaluate on original dataset (instead of normalized data)


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
        stds,
        args):
    
    '''
    NOTE try indepdent svgp models one by one, which is really slow. Prefer to use parallel based implementation ... 
    First 6 arguments are returned from functions in prepare_data.py, these data are naturally designed for MOGP.
    To train multiple IGPs, we need to reconstruct datasets based on them.
    '''
    results_txt = f'{results_folder_path}/results.txt'
    with open(results_txt, 'w') as file:
        file.write(f'Results for random seed: {args.random_seed}\n')

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

    ##### Train and Evaluate IGPs one by one (slow!)
    
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
    train_list_pred_mean_tensors,  test_list_pred_mean_tensors = [], []

    for j in range(config['n_outputs']):

        curr_model = my_model.get_model(j)
        curr_likelihood = my_model.get_likelihood(j)

        curr_model.eval()
        curr_likelihood.eval()

        # Inference for train and test data
        curr_train_output_dist = curr_likelihood(curr_model(list_train_X[j]))
        curr_test_output_dist  = curr_likelihood(curr_model(list_test_X[j]))
        curr_train_pred_mean = curr_train_output_dist.loc.detach()
        curr_test_pred_mean = curr_test_output_dist.loc.detach()

        # RMSE on normalized data
        curr_train_suqare_errors = (curr_train_pred_mean - list_train_Y[j]).square()
        curr_test_square_errors = (curr_test_pred_mean - list_test_Y[j]).square()
        train_error_square_sum += curr_train_suqare_errors.sum()
        test_error_square_sum  += curr_test_square_errors.sum()

        # prepare for NMSE on normalized data

        train_list_pred_mean_tensors.append(curr_train_pred_mean)
        test_list_pred_mean_tensors.append(curr_test_pred_mean)

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
        # On original data (before being transformed based on statistics from train split)
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

            # No NMSE on original data, looks strange!

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
    train_pred_mean_cat, test_pred_mean_cat = torch.cat(train_list_pred_mean_tensors, dim=0), torch.cat(test_list_pred_mean_tensors, dim=0)
    train_target_cat, test_target_cat = torch.tensor(np.array(list_train_Y).flatten()), torch.tensor(np.array(list_test_Y).flatten())
    train_nmse = (train_error_square_sum / train_error_length) / (train_target_cat - train_pred_mean_cat.mean()).square().mean()
    test_nmse = (test_error_square_sum / test_error_length) / (test_target_cat - test_pred_mean_cat.mean()).square().mean()
    
    with open(results_txt, 'a') as file:
        file.write('Evaluation on normalized data:\n')
        file.write(f'global train rmse is {((train_error_square_sum / train_error_length).sqrt()):.4g}\n')
        file.write(f'global test rmse is {((test_error_square_sum / test_error_length).sqrt()):.4g}\n')
        file.write(f'global train nmse is {(train_nmse):.4g}\n')
        file.write(f'global test nmse is {(test_nmse):.4g}\n')
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
        
        if config['dataset_type'] == 'exchange':
            my_model.covar_module_input.base_kernel.lengthscale = config['lengthscale_init']
            my_model.covar_module_input.outputscale = config['outputscale_init']

        else:
            # Or using default init ... 
            pass

    if not only_init_model:
        # Init inducing points in input space
        my_model.variational_strategy.inducing_points_input.data = torch.tensor(np.linspace(config['init_inducing_input_LB'], config['init_inducing_input_UB'], config['n_inducing_input']).reshape(-1, 1)).to(torch.double) 
        
        # Init noise scale in likelihood
        my_likelihood.noise = config['init_likelihood_noise']

    return my_model, my_likelihood

def helper_plot_model_and_likelihood_parameters_change(my_model, my_likelihood, config):
    '''
    helper function used in training lvmogp model. Record how parameter changed during training.
    '''
    # Kernel on input space

    if config['input_kernel_type'] == 'Scale_RBF':
        None


    # Kernel on latent space



    # Likelihood
     
    return None