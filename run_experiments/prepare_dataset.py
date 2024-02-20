# This piece of code is used to store all the functions for preparing dataset.
import numpy as np
import torch
from torch import Tensor
import pandas as pd
import random

def prepare_synthetic_regression_data(config):
    
    '''
    Prepare data.
    Args:
        config: containing all information used to construct the data (ready for model training)
    Return:
        data_inputs: Tensor, of shape n_inputs
        data_Y_squeezed: Tensor, of shape n_input * n_outputs
        ls_of_ls_train_input: list (list of list), n_outputs outer list, n_input_train inner list
        ls_of_ls_test_input: list (list of list), n_outputs outer list, n_input_test inner list
        train_sample_idx_ls: np.array, of shape (n_input_train * n_outputs, )
        test_sample_idx_ls: np.array, of shape (n_input_test * n_outputs, )
    '''

    data_Y_squeezed = Tensor(pd.read_csv(config['data_Y_squeezed_path']).to_numpy()).reshape(-1)
    data_inputs = Tensor(pd.read_csv(config['data_inputs_path']).to_numpy()).reshape(-1) 
    assert data_inputs.shape[0] == config['n_input']
    assert data_Y_squeezed.shape[0] == (config['n_input'] * config['n_outputs'])

    np.random.seed(config['random_seed'])
    list_expri_random_seeds = np.random.randn(config['n_outputs'])

    ls_of_ls_train_input = []
    ls_of_ls_test_input = []
    
    train_sample_idx_ls, test_sample_idx_ls = [], []

    for i in range(config['n_outputs']):
        # iterate across different output functions
        random.seed(list_expri_random_seeds[i])
        train_index = random.sample(range(config['n_input']), config['n_input_train'])
        test_index = [index for index in range(config['n_input']) if index not in train_index]
        ls_of_ls_train_input.append(train_index)
        ls_of_ls_test_input.append(test_index)

        train_sample_idx_ls = np.concatenate((train_sample_idx_ls, list(np.array(train_index) + config['n_input']*i)))
        test_sample_idx_ls = np.concatenate((test_sample_idx_ls, list(np.array(test_index) + config['n_input']*i)))

    return data_inputs, data_Y_squeezed, ls_of_ls_train_input, ls_of_ls_test_input, train_sample_idx_ls, test_sample_idx_ls


def prepare_spatio_temp_data(config):

    '''
    Prepare Spatio-Temporal data.
    Args:
        config: containing all information used to construct the data (ready for model training)
    Return:
        lon_lat_tensor: of shape (n_outputs, 2), lat and lon are normalized.
        data_Y_squeezed:  every output is normalized using training data points.
        means, stds: values used to normalized every output (they are obtained from training data points).
    '''
    data_Y_tensor = torch.load(config['data_Y_tensor'])
    n_times = data_Y_tensor.shape[0]
    n_lat = data_Y_tensor.shape[1]
    n_lon = data_Y_tensor.shape[2]

    data_Y_permuted = data_Y_tensor.permute(2, 1, 0)
    # longitude, latitude, time
    data_Y_reshaped_ = data_Y_permuted.reshape(n_lon * n_lat, n_times)
    #  first lon with all lat, second lon with all lat, ..., last lon with all lat.

    # Normalize the data
    data_Y_train_reshaped = data_Y_reshaped_[:, :config['n_input_train']]
    means = data_Y_train_reshaped.mean(dim=1)
    stds = data_Y_train_reshaped.std(dim=1, unbiased=False)
    assert means.shape[0] == stds.shape[0] == n_lon * n_lat
    means_expanded = means.unsqueeze(1)
    stds_expanded = stds.unsqueeze(1)
    data_Y_reshaped = (data_Y_reshaped_ - means_expanded) / stds_expanded

    assert config['n_outputs'] == data_Y_reshaped.shape[0]
    assert config['n_input_train'] + config['n_input_test'] == data_Y_reshaped.shape[-1] 
    
    data_Y_squeezed = data_Y_reshaped.reshape(-1)

    lat_tensor = torch.load(config['data_lat_tensor'])
    lon_tensor = torch.load(config['data_lon_tensor'])

    lon_lat_tensor_ = torch.zeros(n_lon, n_lat, 2)
    for i in range(n_lon):
        for j in range(n_lat):
            lon_lat_tensor_[i, j, :] = torch.tensor([lon_tensor[i].item(), lat_tensor[j].item()])

    lon_lat_tensor_reshape = lon_lat_tensor_.reshape(-1, 2)
    lon_lat_means, lon_lat_stds = lon_lat_tensor_reshape.mean(dim=0), lon_lat_tensor_reshape.std(dim=0, unbiased=False)
    lon_lat_tensor = (lon_lat_tensor_reshape - lon_lat_means.unsqueeze(0)) / lon_lat_stds.unsqueeze(0)

    # NOTE: The inputs domain can be defined by ourselves!

    translate_bias = config['min_input_bound']
    translate_scale = (config['max_input_bound'] - config['min_input_bound']) / config['n_input']

    data_inputs =  translate_bias + translate_scale * ( Tensor([i for i in range(config['n_input'])]) )

    # repeate
    ls_of_ls_train_input = [[i for i in range(config['n_input_train'])]] * config['n_outputs']
    ls_of_ls_test_input = [[(i + config['n_input_train']) for i in range(config['n_input_test'])]] * config['n_outputs']

    train_sample_idx_ls, test_sample_idx_ls = [], []
    for i in range(config['n_outputs']):
        train_sample_idx_ls = np.concatenate((train_sample_idx_ls, list(np.array(ls_of_ls_train_input[i]) + config['n_input']*i)))
        test_sample_idx_ls = np.concatenate((test_sample_idx_ls, list(np.array(ls_of_ls_test_input[i]) + config['n_input']*i)))

    return data_inputs, data_Y_squeezed, ls_of_ls_train_input, ls_of_ls_test_input, lon_lat_tensor, train_sample_idx_ls, test_sample_idx_ls, means, stds