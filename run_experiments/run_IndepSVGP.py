import os
import argparse
import sys
sys.path.append('/Users/jiangxiaoyu/Desktop/All Projects/Scalable_LVMOGP/')
from run_experiments.prepare_dataset import *
from run_experiments.utils import *
from code_blocks.our_models.Multi_ISVGP import Multi_Variational_IGP
import torch
import numpy as np
import random
import yaml
from datetime import datetime

if __name__ == "__main__":

    torch.set_default_dtype(torch.double)

    parser = argparse.ArgumentParser(description='which file to run')
    parser.add_argument('--config_name', type=str, help='config name')
    args = parser.parse_args()
    ### Load hyperparameters from .yaml file

    root_config = '/Users/jiangxiaoyu/Desktop/All Projects/Scalable_LVMOGP/configs'
    # NOTE: Specify name here for different experiments:

    ## Examples:
    # curr_config_name = 'spatiotemp_IGP/PeriodicInputsRBFKernel_times_Scale_RBF/IndepSVGP_unfix' # fix refers to fixing inducing points

    curr_config_name = args.config_name
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
    
    ### Define Model
    
    MultiIGP = Multi_Variational_IGP(
        num_models = config['n_outputs'], 
        inducing_points = Tensor(np.linspace(config['init_inducing_input_LB'], config['init_inducing_input_UB'], config['n_inducing_input']).reshape(-1, 1)), 
        init_likelihood_noise = config['init_likelihood_noise'], 
        kernel_type = config['input_kernel_type'], 
        learn_inducing_locations= config['learn_inducing_locations_input'],
        input_dim = config['input_dim']
    )

    ### Model (Kernel) initialization ... 

    for i in range(config['n_outputs']):
       _, _ = helper_init_model_and_likeli(MultiIGP.get_model(i), config, only_init_model=True)
    
    ### Train and evaluate the model ... 
       
    train_and_eval_multiIndepSVGP_model(
        data_inputs = data_inputs,
        data_Y_squeezed = data_Y_squeezed,
        ls_of_ls_train_input = ls_of_ls_train_input,
        ls_of_ls_test_input = ls_of_ls_test_input,
        train_sample_idx_ls = train_sample_idx_ls, 
        test_sample_idx_ls = test_sample_idx_ls,
        my_model = MultiIGP,
        config = config,
        results_folder_path = results_folder_path,
        means = means,
        stds = stds
    )
    