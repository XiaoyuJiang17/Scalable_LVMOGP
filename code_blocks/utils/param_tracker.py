import matplotlib.pyplot as plt
from code_blocks.our_models.lvmogp_svi import LVMOGP_SVI
from code_blocks.likelihoods.gaussian_likelihood import GaussianLikelihood

class ParamTracker:
    def __init__(self, param_extractor):
        '''
        param_extractor: a function takes model config as input, output the dict of parameters we are interested for tracking
        '''
        self.param_extractor = param_extractor
        self.param_history_dict = {} # dict of list of dicts
        self.initialized = False
    
    def update(self, *models):

        if self.initialized == False:
            for i, model in enumerate(models):
                self.param_history_dict[f'param_history_{i}'] = []

            self.initialized = True

        for i, model in enumerate(models):
            # for each model, a list is used to store all params histories (i.e. list of dict)
            self.param_history_dict[f'param_history_{i}'].append(self.param_extractor(model))
    
    def plot(self, folder_path):

        for i in range(len(self.param_history_dict.keys())): # iterate over all models
            # for current model, param history is stored in the following list of dicts
            data = self.param_history_dict[f'param_history_{i}']
            values_dict = {key: [] for key in data[0]} 

            for item in data:
                for key in item:
                    values_dict[key].append(item[key])

            for key, values in values_dict.items():
                plt.figure()
                plt.plot(values)
                plt.title(f"Plot of Model {i} with param {key}")
                plt.xlabel("Index")
                plt.ylabel("Value")
                plt.savefig(f"{folder_path}/model_{i}_key_{key}.png") 
                plt.close()  

        
def param_extractor1(model):
    '''
    Used for lvmogp model with Scale_RBF kernel on both latent and input space. Likelihood is Gaussian.
    
    Args:
        model: the model we are interested for tracking parameters. 2 possible types: lvmogp and gaussian likelihood.
    '''

    param_dict = {}
    if isinstance(model, LVMOGP_SVI):
        for id in range(model.covar_module_latent.base_kernel.lengthscale.detach().shape[-1]): # lengthscales has form: [[1. , 2. , 4.]]
            param_dict[f'latent_lengthscale_{id}'] = model.covar_module_latent.base_kernel.lengthscale.detach()[0][id].item()
        
        param_dict['latent_outputscale'] = model.covar_module_latent.outputscale.detach().item()

        for id in range(model.covar_module_input.base_kernel.lengthscale.detach().shape[-1]):
            param_dict[f'input_lengthscale_{id}'] = model.covar_module_input.base_kernel.lengthscale.detach()[0][id].item()
        param_dict['input_outputscale'] = model.covar_module_input.outputscale.detach().item()

    elif isinstance(model, GaussianLikelihood):
        param_dict['noise_scale'] = model.noise.detach().item()

    else:
        NotImplementedError


    return param_dict
