import os
import sys
sys.path.append('/Users/jiangxiaoyu/Desktop/All Projects/Scalable_LVMOGP/')
from run_experiments.prepare_dataset import *
from run_experiments.utils import *
from utils_general import pca_reduction
from gplvm_init import specify_gplvm, train_gplvm
from code_blocks.our_models.lvmogp_svi import LVMOGP_SVI
from code_blocks.likelihoods.gaussian_likelihood import GaussianLikelihood, GaussianLikelihoodWithMissingObs
import torch
import numpy as np
import random
import yaml
from datetime import datetime

if __name__ == "__main__": 

    ### Load hyperparameters from .yaml file 

    root_config = '/Users/jiangxiaoyu/Desktop/All Projects/Scalable_LVMOGP/configs/' 
    # NOTE: Specify name here for different experiments: 
    # rnd (fix) + unfix (fix) ; first referring to initialization, second referring to inducing points in input space 
    curr_config_name = 'spatiotemp/Periodic_times_Scale_RBF/lvmogp_catlatent_rnd_unfix' 
    curr_config = f'{root_config}/{curr_config_name}.yaml'
    with open(curr_config, 'r') as file: 
        config = yaml.safe_load(file) 

    ### Create folder to save results
    root_folder_path = '/Users/jiangxiaoyu/Desktop/All Projects/Scalable_LVMOGP/experiments_results' 
    results_folder_path = os.path.join(root_folder_path, curr_config_name, f"{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}") 

    os.makedirs(results_folder_path, exist_ok=True)

    ### specify random seed
    
    random.seed(config['random_seed'])
    np.random.seed(config['random_seed'])
    torch.manual_seed(config['random_seed'])

    ### Specify the dataset

    if config['dataset_type'] == 'synthetic_regression':
        data_inputs, data_Y_squeezed, ls_of_ls_train_input, ls_of_ls_test_input, train_sample_idx_ls, test_sample_idx_ls = prepare_synthetic_regression_data(config)
        means, stds = None, None
    
    elif config['dataset_type'] == 'spatio_temporal_data':
        data_inputs, data_Y_squeezed, ls_of_ls_train_input, ls_of_ls_test_input, lon_lat_tensor, train_sample_idx_ls, test_sample_idx_ls, means, stds = prepare_spatio_temp_data(config)
    
    ### Model Initialization (before instantiation) ... 
    latent_first_init, latent_second_init = None, None

    gplvm_init = True # use trained gplvm's latents as initialization ...  

    if config['NNEncoder'] == False:

        if config['trainable_latent_dim'] == 2 and gplvm_init == True:
            ## Initialization by training GPLVM ... 
            data_Y = data_Y_squeezed.reshape(config['n_outputs'], config['n_input'])[:, :config['n_input_train']]
            gplvm_model = specify_gplvm(config)
            gplvm_likelihood = GaussianLikelihoodWithMissingObs()
            gplvm_model, gplvm_likelihood, losses = train_gplvm(gplvm_model, 
                                                                gplvm_likelihood,
                                                                data_Y)

            latent_first_init = gplvm_model.X.q_mu.detach().data

        if config['dataset_type'] == 'spatio_temporal_data' and config['trainable_latent_dim']  == 0:
            latent_second_init = lon_lat_tensor 

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
        layers = None
    )

    my_likelihood = GaussianLikelihood()

    ### Model Initialization (after instantiation) ... 

    # Kernels hypers for latents
    try: 
        my_model = helper_init_latent_kernel(my_model, gplvm_model)

    except NameError: # gplvm_model is defined ...
        pass

    # Kernels hypers for inputs , and likelihood
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
        latent_info = lon_lat_tensor,
        results_folder_path = results_folder_path,
        train_sample_idx_ls = train_sample_idx_ls, 
        test_sample_idx_ls = test_sample_idx_ls,
        means = means,
        stds = stds
    )
        
    
        
    
    



