import os
import argparse
import sys
sys.path.append('/Users/jiangxiaoyu/Desktop/All Projects/Scalable_LVMOGP/')
from run_experiments.prepare_dataset import *
from run_experiments.utils import *
from gplvm_init import specify_gplvm, train_gplvm
from code_blocks.our_models.lvmogp_svi import LVMOGP_SVI
from code_blocks.likelihoods.gaussian_likelihood import GaussianLikelihood, GaussianLikelihoodWithMissingObs
import torch
import numpy as np
import random
import yaml
from datetime import datetime

if __name__ == "__main__": 

    torch.set_default_dtype(torch.double)
    
    parser = argparse.ArgumentParser(description='which file to run')
    parser.add_argument('--config_name', type=str, help='config name')
    parser.add_argument('--random_seed', type=int, help='random seed')
    args = parser.parse_args()

    ### Load hyperparameters from .yaml file 

    root_config = '/Users/jiangxiaoyu/Desktop/All Projects/Scalable_LVMOGP/configs' 
    # NOTE: Specify name here for different experiments: 
    # rnd (fix) + unfix (fix) ; first referring to initialization, second referring to inducing points in input space 

    ## Examples: 
    # curr_config_name = 'spatiotemp/Periodic_times_Scale_RBF/lvmogp_catlatent_rnd_unfix' 
    # curr_config_name = 'synthetic/Scale_RBF/lvmogp_unfix'

    curr_config_name = args.config_name
    curr_config = f'{root_config}/{curr_config_name}.yaml'
    with open(curr_config, 'r') as file: 
        config = yaml.safe_load(file) 

    ### Create folder to save results
    root_folder_path = '/Users/jiangxiaoyu/Desktop/All Projects/Scalable_LVMOGP/experiments_results' 
    results_folder_path = os.path.join(root_folder_path, curr_config_name, f"{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}") 

    os.makedirs(results_folder_path, exist_ok=True)

    ### specify random seed
    
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    ### Specify the dataset

    if config['dataset_type'] == 'synthetic_regression':
        data_inputs, data_Y_squeezed, ls_of_ls_train_input, ls_of_ls_test_input, train_sample_idx_ls, test_sample_idx_ls = prepare_synthetic_regression_data(config)
        means, stds = None, None

    elif config['dataset_type'] == 'mocap':
        data_inputs, data_Y_squeezed, ls_of_ls_train_input, ls_of_ls_test_input, train_sample_idx_ls, test_sample_idx_ls, means, stds = prepare_mocap_data(config)
    
    elif config['dataset_type'] == 'exchange':
        data_inputs, data_Y_squeezed, ls_of_ls_train_input, ls_of_ls_test_input, train_sample_idx_ls, test_sample_idx_ls, means, stds = prepare_exchange_data(config)

    elif config['dataset_type'] == 'egg':
        data_inputs, data_Y_squeezed, ls_of_ls_train_input, ls_of_ls_test_input, train_sample_idx_ls, test_sample_idx_ls, means, stds = prepare_egg_data(config)

    elif config['dataset_type'] == 'spatio_temporal_data':
        data_inputs, data_Y_squeezed, ls_of_ls_train_input, ls_of_ls_test_input, lon_lat_tensor, train_sample_idx_ls, test_sample_idx_ls, means, stds = prepare_spatio_temp_data(config)
    
    ### Model Initialization (before instantiation) ... 
    latent_first_init, latent_second_init, latent_info = None, None, None

    if config['dataset_type'] == 'spatio_temporal_data':
        # Normally, for spatio-temp data, we use lon_lat tensor as latent information ... 
        latent_info = lon_lat_tensor

    gplvm_init = config['gplvm_init'] if 'gplvm_init' in config else True  # use trained gplvm's latents as initialization ...  

    if config['NNEncoder'] == False:

        if config['trainable_latent_dim'] > 0 and gplvm_init == True:
            ## Initialization by training GPLVM ... 
            print('Initialization by training GPLVM ...')
            
            if config['dataset_type'] == 'spatio_temporal_data':
                # data_Y = data_Y_squeezed.reshape(config['n_outputs'], config['n_input'])[:, :config['n_input_train']]
                data_Y = data_Y_squeezed.reshape(config['n_outputs'], config['n_input'])
                data_Y[:, config['n_input_train']:] = torch.nan

            elif config['dataset_type'] in ['synthetic_regression', 'mocap', 'exchange', 'egg']:
                data_Y_squeezed_copy = data_Y_squeezed.clone()
                data_Y_squeezed_copy[test_sample_idx_ls] = torch.nan
                data_Y = data_Y_squeezed_copy.reshape(config['n_outputs'], config['n_input'])

            gplvm_model = specify_gplvm(config)
            gplvm_likelihood = GaussianLikelihoodWithMissingObs()
            gplvm_model, gplvm_likelihood, losses = train_gplvm(gplvm_model, 
                                                                gplvm_likelihood,
                                                                data_Y)

            latent_first_init = gplvm_model.X.q_mu.detach().data

        if config['dataset_type'] == 'spatio_temporal_data' and config['trainable_latent_dim']  == 0:
            latent_second_init = latent_info 

    ### Define model and likelihood
    
    trainable_latent_dim = config['trainable_latent_dim'] if 'trainable_latent_dim' in config else None

    my_model = LVMOGP_SVI(
        n_outputs = config['n_outputs'],
        n_input = config['n_input_train'],                    # NOTE PAY ATTENTION, not total n_inputs.
        input_dim = config['input_dim'],
        latent_dim = config['latent_dim'],
        n_inducing_input = config['n_inducing_input'],
        n_inducing_latent = config['n_inducing_latent'],
        learn_inducing_locations_latent = config['learn_inducing_locations_latent'],
        learn_inducing_locations_input = config['learn_inducing_locations_input'],
        latent_kernel_type = config['latent_kernel_type'],
        input_kernel_type = config['input_kernel_type'],
        trainable_latent_dim = trainable_latent_dim,
        latent_first_init = latent_first_init,                # if None, random initialization
        latent_second_init = latent_second_init,              # if None, random initialization
        NNEncoder = config['NNEncoder'],
        layers = None                                         # if none, adopt default value [4, 8, 4]
    )

    my_likelihood = GaussianLikelihood()

    ### Model Initialization (after instantiation) ... 

    # Kernels hypers for latents
    try: 
        my_model = helper_init_latent_kernel(my_model, gplvm_model)

    except NameError: # gplvm_model is not defined ...
        pass

    # Kernels hypers for inputs, and likelihood
    my_model, my_likelihood = helper_init_model_and_likeli(my_model, config, my_likelihood, only_init_model=False)

    #### Train and evaluate the model ... 

    train_and_eval_lvmogp_model(
        data_inputs = data_inputs,
        data_Y_squeezed = data_Y_squeezed,
        ls_of_ls_train_input = ls_of_ls_train_input,
        ls_of_ls_test_input = ls_of_ls_test_input,
        my_model = my_model,
        my_likelihood = my_likelihood,
        config = config,
        latent_info = latent_info,
        results_folder_path = results_folder_path,
        train_sample_idx_ls = train_sample_idx_ls, 
        test_sample_idx_ls = test_sample_idx_ls,
        means = means,
        stds = stds,
        args = args
    )
        
    
        
    
    



