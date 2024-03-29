# This piece of code is used to store all the functions for preparing dataset.
import numpy as np
import torch
import torch
import pandas as pd
import random

def prepare_synthetic_regression_data(config):
    
    '''
    Prepare data.
    Args:
        config: containing all information used to construct the data (ready for model training)
    Return:
        data_inputs: torch.tensor, of shape n_inputs
        data_Y_squeezed: torch.tensor, of shape n_input * n_outputs
        ls_of_ls_train_input: list (list of list), n_outputs outer list, n_input_train inner list
        ls_of_ls_test_input: list (list of list), n_outputs outer list, n_input_test inner list
        train_sample_idx_ls: np.array, of shape (n_input_train * n_outputs, )
        test_sample_idx_ls: np.array, of shape (n_input_test * n_outputs, )
    '''

    data_Y_squeezed = torch.tensor(pd.read_csv(config['data_Y_squeezed_path']).to_numpy()).reshape(-1)
    # data_inputs = torch.tensor(pd.read_csv(config['data_inputs_path']).to_numpy()).reshape(-1) 
    translate_bias = config['min_input_bound']
    translate_scale = (config['max_input_bound'] - config['min_input_bound']) / config['n_input']
    
    data_inputs =  translate_bias + translate_scale * ( torch.tensor([i for i in range(config['n_input'])]) )
    assert data_inputs.shape[0] == config['n_input']
    assert data_Y_squeezed.shape[0] == (config['n_input'] * config['n_outputs'])

    # randomly split data ... 
    data_random_seed = config['data_random_seed'] if 'data_random_seed' in config else 1
    np.random.seed(data_random_seed)
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

def prepare_mocap_data(config):
    '''
    Prepare data for MOCAP dataset.
        for each output, we randomly mask a group of inputs (possibly different) during training (used for test).
    '''
    # NOTE: first column used as index ... 
    data_Y = torch.tensor(pd.read_csv(config['data_Y_path'], index_col=0).to_numpy())
    n_outputs, n_input = data_Y.shape[0], data_Y.shape[-1]
    assert n_outputs == config['n_outputs']
    assert n_input == config['n_input']

    translate_bias = config['min_input_bound']
    translate_scale = (config['max_input_bound'] - config['min_input_bound']) / config['n_input']

    data_inputs =  translate_bias + translate_scale * ( torch.tensor([i for i in range(config['n_input'])]) )

    # randomly split data ... 
    data_random_seed = config['data_random_seed'] if 'data_random_seed' in config else 1
    np.random.seed(data_random_seed)
    list_expri_random_seeds = np.random.randn(config['n_outputs'])

    ls_of_ls_train_input = []
    ls_of_ls_test_input = []
    
    train_sample_idx_ls, test_sample_idx_ls = [], []

    means, stds = torch.zeros(config['n_outputs']), torch.zeros(config['n_outputs'])

    for i in range(config['n_outputs']):
        # iterate across different output functions
        random.seed(list_expri_random_seeds[i])
        train_index = random.sample(range(config['n_input']), config['n_input_train'])
        test_index = [index for index in range(config['n_input']) if index not in train_index]
        ls_of_ls_train_input.append(train_index)
        ls_of_ls_test_input.append(test_index)

        # compute mean and std for this output:
        means[i] = data_Y[i, train_index].mean()
        stds[i] = data_Y[i, train_index].std()

        data_Y[i, :] = (data_Y[i, :] - means[i]) / (stds[i] + 1e-8)

        train_sample_idx_ls = np.concatenate((train_sample_idx_ls, list(np.array(train_index) + config['n_input']*i)))
        test_sample_idx_ls = np.concatenate((test_sample_idx_ls, list(np.array(test_index) + config['n_input']*i)))

    data_Y_squeezed = data_Y.reshape(-1)
    assert data_Y_squeezed.shape[0] == (config['n_input'] * config['n_outputs'])
    
    return data_inputs, data_Y_squeezed, ls_of_ls_train_input, ls_of_ls_test_input, train_sample_idx_ls, test_sample_idx_ls, means, stds


def prepare_sliced_mocap_data(config):
    '''
    Prepare MOCAP dataset.
        Pick some outputs, mask a continuous part (either first 1/3, middle 1/3 or last 1/3) for each of them during training (used for test).
        Some of outputs are entirely used for training (no test points).
    '''
    def split_list_equally(input_list, num_splits=4, seed=1):

        random.seed(seed)
        random.shuffle(input_list)
        # split_size = len(input_list) // num_splits
        lists = [[] for _ in range(num_splits)]

        for i, item in enumerate(input_list):
            lists[i % num_splits].append(item)

        return lists
    
    data_Y = torch.tensor(pd.read_csv(config['data_Y_path'], index_col=0).to_numpy())

    n_outputs, n_input = data_Y.shape[0], data_Y.shape[-1]
    assert n_outputs == config['n_outputs']
    assert n_input == config['n_input']

    translate_bias = config['min_input_bound']
    translate_scale = (config['max_input_bound'] - config['min_input_bound']) / config['n_input']
    data_inputs =  translate_bias + translate_scale * ( torch.tensor([i for i in range(config['n_input'])]) )

    all_output_idxs = [i for i in range(config['n_outputs'])]
    no_missing_output_idx, first_missing_output_idx, second_missing_output_idx, last_missing_output_idx = split_list_equally(all_output_idxs, num_splits=4)

    ls_of_ls_train_input = []
    ls_of_ls_test_input = []
    all_inputs_idx = [i for i in range(config['n_input'])]
    first_split, second_split = int(config['n_input']/3), int(config['n_input']*2/3)

    means, stds = torch.zeros(config['n_outputs']), torch.zeros(config['n_outputs'])

    train_sample_idx_ls, test_sample_idx_ls = [], []

    for id in range(config['n_outputs']):

        if id in no_missing_output_idx:
            # all inputs used for training
            train_index, test_index = all_inputs_idx, []
            ls_of_ls_train_input.append(train_index)
            ls_of_ls_test_input.append(test_index)

            means[id] = data_Y[id, train_index].mean()
            stds[id] = data_Y[id, train_index].std()
            data_Y[id, :] = (data_Y[id, :] - means[id]) / (stds[id] + 1e-8)

            train_sample_idx_ls = np.concatenate((train_sample_idx_ls, list(np.array(train_index) + config['n_input']*id)))
            test_sample_idx_ls = np.concatenate((test_sample_idx_ls, list(np.array(test_index) + config['n_input']*id)))
        
        elif id in first_missing_output_idx:
            # first 1/3 used for testing, remaining parts for training
            train_index, test_index = all_inputs_idx[first_split:], all_inputs_idx[:first_split]
            ls_of_ls_train_input.append(train_index)
            ls_of_ls_test_input.append(test_index)

            means[id] = data_Y[id, train_index].mean()
            stds[id] = data_Y[id, train_index].std()
            data_Y[id, :] = (data_Y[id, :] - means[id]) / (stds[id] + 1e-8)

            train_sample_idx_ls = np.concatenate((train_sample_idx_ls, list(np.array(train_index) + config['n_input']*id)))
            test_sample_idx_ls = np.concatenate((test_sample_idx_ls, list(np.array(test_index) + config['n_input']*id)))
        
        elif id in second_missing_output_idx:
            # second 1/3 used for testing, remaining parts for training
            test_index = all_inputs_idx[first_split:second_split]
            train_index = [i for i in all_inputs_idx if i not in test_index]
            ls_of_ls_train_input.append(train_index)
            ls_of_ls_test_input.append(test_index)

            means[id] = data_Y[id, train_index].mean()
            stds[id] = data_Y[id, train_index].std()
            data_Y[id, :] = (data_Y[id, :] - means[id]) / (stds[id] + 1e-8)

            train_sample_idx_ls = np.concatenate((train_sample_idx_ls, list(np.array(train_index) + config['n_input']*id)))
            test_sample_idx_ls = np.concatenate((test_sample_idx_ls, list(np.array(test_index) + config['n_input']*id)))
        
        elif id in last_missing_output_idx:
            # last 1/3 used for testing, remaining parts for training 
            train_index, test_index = all_inputs_idx[:second_split], all_inputs_idx[second_split:]
            ls_of_ls_train_input.append(train_index)
            ls_of_ls_test_input.append(test_index)

            means[id] = data_Y[id, train_index].mean()
            stds[id] = data_Y[id, train_index].std()
            data_Y[id, :] = (data_Y[id, :] - means[id]) / (stds[id] + 1e-8)

            train_sample_idx_ls = np.concatenate((train_sample_idx_ls, list(np.array(train_index) + config['n_input']*id)))
            test_sample_idx_ls = np.concatenate((test_sample_idx_ls, list(np.array(test_index) + config['n_input']*id)))
    
    data_Y_squeezed = data_Y.reshape(-1)
    assert data_Y_squeezed.shape[0] == (config['n_input'] * config['n_outputs'])

    return data_inputs, data_Y_squeezed, ls_of_ls_train_input, ls_of_ls_test_input, train_sample_idx_ls, test_sample_idx_ls, means, stds


def prepare_exchange_data(config):

    '''
    exchange datasets from OILMM paper.
        The dataset rates_df contains 14 columns: 13 target time series, 1 input series. Each time series is of 251 rows,
        though some of the elements are missing for USD/XAG (8), USD/XAU (9), USD/XPT (42).
        We are interested in making predict on (artificially masked targets) of 3 time series: USD/CAD (#test = 51), USD/JPY (#test = 101), and USD/AUD (#test = 151).
    '''
    columns_as_targets = ['USD/CHF', 'USD/EUR', 'USD/GBP', 'USD/HKD', 'USD/KRW', 'USD/MXN', 'USD/NZD', 'USD/XAG', 'USD/XAU', 'USD/XPT', 'USD/CAD', 'USD/JPY', 'USD/AUD']

    rates_df = pd.read_csv(config['exchange_rate_path'])
    train_df = pd.read_csv(config['exchange_train_path'])
    # test_df = pd.read_csv(config['exchange_test_path'])

    assert rates_df.shape[0] == config['n_input']
    assert rates_df.shape[1] == config['n_outputs'] + 1

    # original inputs 
    data_inputs = torch.tensor(rates_df['year']) 

    # NOTE: apply proper transformation (make inputs evenly spaced)

    # translate_bias = config['min_input_bound']
    # translate_scale = (config['max_input_bound'] - config['min_input_bound']) / config['n_input']
    # data_inputs =  translate_bias + translate_scale * ( torch.tensor([i for i in range(data_inputs.shape[0])]))

    ls_of_ls_train_input = []
    ls_of_ls_test_input = []

    for col_name in rates_df.columns:
        if col_name != 'year' and col_name != 'USD/CAD' and col_name != 'USD/JPY' and col_name != 'USD/AUD':
            ls_of_ls_train_input.append(rates_df.index[rates_df[col_name].notna()].to_list())
            ls_of_ls_test_input.append([])

        elif col_name != 'year' and col_name == 'USD/CAD':
            test_index_1 = [i for i in range(49, 100)]
            train_index_1 = [i for i in range(config['n_input']) if i not in test_index_1]
            ls_of_ls_train_input.append(train_index_1)
            ls_of_ls_test_input.append(test_index_1)
        
        elif col_name != 'year' and col_name == 'USD/JPY':
            test_index_2 = [i for i in range(49, 150)]
            train_index_2 = [i for i in range(config['n_input']) if i not in test_index_2]
            ls_of_ls_train_input.append(train_index_2)
            ls_of_ls_test_input.append(test_index_2)
        
        elif col_name != 'year' and col_name == 'USD/AUD':
            test_index_3 = [i for i in range(49, 200)]
            train_index_3 = [i for i in range(config['n_input']) if i not in test_index_3]
            ls_of_ls_train_input.append(train_index_3)
            ls_of_ls_test_input.append(test_index_3)

    assert len(ls_of_ls_train_input) == 13 == len(ls_of_ls_test_input)

    train_sample_idx_ls, test_sample_idx_ls = [], []
    for i in range(config['n_outputs']):
        train_sample_idx_ls = np.concatenate((train_sample_idx_ls, list(np.array(ls_of_ls_train_input[i]) + config['n_input']*i)))
        test_sample_idx_ls = np.concatenate((test_sample_idx_ls, list(np.array(ls_of_ls_test_input[i]) + config['n_input']*i)))

    # make sure no test data statistics are leaked
    assert set(train_df.iloc[:, 1:].columns.to_list()) == set(columns_as_targets)   
    means, stds = torch.tensor(train_df.iloc[:, 1:].mean().to_numpy()), torch.tensor(train_df.iloc[:, 1:].std().to_numpy())
    normalized_rates_df = (rates_df - train_df.mean()) / train_df.std()

    data_Y_squeezed = torch.tensor(normalized_rates_df.iloc[:, 1:].to_numpy().T).reshape(-1)

    # make sure the train and test datasets contain NO nan value
    assert torch.isnan(data_Y_squeezed[train_sample_idx_ls]).any() == False
    assert torch.isnan(data_Y_squeezed[test_sample_idx_ls]).any() == False

    return data_inputs, data_Y_squeezed, ls_of_ls_train_input, ls_of_ls_test_input, train_sample_idx_ls, test_sample_idx_ls, means, stds


def prepare_egg_data(config):
    '''
    EGG dataset from OILMM paper.
        The egg_data dataset has 8 columns (column names: time, F1, F2, F3, F4, F5, F6, FZ) and 256 rows, no missing data.
        'time' column serves as the inputs, other columns serve as target time series.
        The last 100 rows of F1, F2, FZ are masked during training, they serve as test data.
    '''
    columns_as_targets = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'FZ']

    egg_data = pd.read_csv(config['egg_data_all_path'])
    egg_train_data = pd.read_csv(config['egg_data_train_path'])

    assert egg_data.shape[0] == config['n_input']
    assert egg_data.shape[1] == config['n_outputs'] + 1

    # original inputs 
    data_inputs = torch.tensor(egg_data['time']) 

    # TODO: apply proper transformation (possibly make inputs evenly spaced)

    ls_of_ls_train_input = []
    ls_of_ls_test_input = []

    input_full_list = [i for i in range(config['n_input'])]

    for col_name in egg_data.columns:
        if col_name != 'time' and ( col_name == 'F1' or col_name == 'F2' or col_name == 'FZ' ):
            ls_of_ls_train_input.append(input_full_list[:-100])
            ls_of_ls_test_input.append(input_full_list[-100:])
        elif col_name != 'time':
            ls_of_ls_train_input.append(input_full_list)
            ls_of_ls_test_input.append([])

    assert len(ls_of_ls_train_input) == 7 == len(ls_of_ls_test_input)    

    train_sample_idx_ls, test_sample_idx_ls = [], []
    for i in range(config['n_outputs']):
        train_sample_idx_ls = np.concatenate((train_sample_idx_ls, list(np.array(ls_of_ls_train_input[i]) + config['n_input']*i)))
        test_sample_idx_ls = np.concatenate((test_sample_idx_ls, list(np.array(ls_of_ls_test_input[i]) + config['n_input']*i)))  

    # make sure no test data statistics are leaked
    means, stds = torch.tensor(egg_train_data[columns_as_targets].mean().to_numpy()), torch.tensor(egg_train_data[columns_as_targets].std().to_numpy())  
    normalized_egg_data = (egg_data - egg_train_data.mean()) / egg_train_data.std()

    data_Y_squeezed = torch.tensor(normalized_egg_data[columns_as_targets].to_numpy().T).reshape(-1)

    # make sure the train and test datasets contain NO nan value
    assert torch.isnan(data_Y_squeezed[train_sample_idx_ls]).any() == False
    assert torch.isnan(data_Y_squeezed[test_sample_idx_ls]).any() == False

    assert data_Y_squeezed[train_sample_idx_ls].shape[0] == config['number_all_train_data']

    return data_inputs, data_Y_squeezed, ls_of_ls_train_input, ls_of_ls_test_input, train_sample_idx_ls, test_sample_idx_ls, means, stds


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

    data_inputs =  translate_bias + translate_scale * ( torch.tensor([i for i in range(config['n_input'])]) )

    # repeate
    ls_of_ls_train_input = [[i for i in range(config['n_input_train'])]] * config['n_outputs']
    ls_of_ls_test_input = [[(i + config['n_input_train']) for i in range(config['n_input_test'])]] * config['n_outputs']

    train_sample_idx_ls, test_sample_idx_ls = [], []
    for i in range(config['n_outputs']):
        train_sample_idx_ls = np.concatenate((train_sample_idx_ls, list(np.array(ls_of_ls_train_input[i]) + config['n_input']*i)))
        test_sample_idx_ls = np.concatenate((test_sample_idx_ls, list(np.array(ls_of_ls_test_input[i]) + config['n_input']*i)))

    return data_inputs, data_Y_squeezed, ls_of_ls_train_input, ls_of_ls_test_input, lon_lat_tensor, train_sample_idx_ls, test_sample_idx_ls, means, stds