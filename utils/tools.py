import os
import os.path as osp
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from .config import Config
import random
import torch.backends.cudnn as cudnn
import json

def load_config(filename:str = None, _print: bool = True):
    '''
    load and print config
    '''
    print('loading config from ' + filename + ' ...')
    configfile = Config(filename=filename)
    config = configfile._cfg_dict

    if _print == True:
        print_config(config)
    
    return config

def print_config(config):
    print('---------- params info: ----------')
    for k, v in config.items():
        print(k, ' : ', v)
    print('---------------------------------')

def get_log_name(path, config):
    # log_name =  config['dataset'] + '_' + config['algorithm'] + '_' + config['noise_type'] + '_' + \
    #             str(config['percent']) + '_seed' + str(config['seed']) + '.json'
    log_name =  config['dataset'] + '_' + config['algorithm'] + '_' + config['noise_type'] + '_' + \
                 str(config['percent']) + '_distribution_' + config['distribution_t'] + '.json'
    if osp.exists('./log') is False:
        os.mkdir('./log')
    log_name = osp.join('./log', log_name)
    return log_name

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True

def save_results(config, last_ten, best_acc, best_epoch, jsonfile):
    result_dict = config
    result_dict['last10_acc_mean'] = last_ten.mean()
    result_dict['last10_acc_std'] = last_ten.std()
    result_dict['best_acc'] = best_acc
    result_dict['best_epoch'] = best_epoch
    with open(jsonfile, 'w') as out:
        json.dump(result_dict, out, sort_keys=False, indent=4)

def plot_results(epochs, test_acc, plotfile):
    plt.style.use('ggplot')
    plt.plot(np.arange(1, epochs), test_acc, label='scratch - acc')
    plt.xticks(np.arange(0, epochs + 1, max(1, epochs // 20))) # train epochs
    plt.xlabel('Epoch')
    plt.yticks(np.arange(0, 101, 10)) # Acc range: [0, 100]
    plt.ylabel('Acc divergence')
    plt.savefig(plotfile)

def get_test_acc(acc):
    return (acc[0] + acc[1]) / 2. if isinstance(acc, tuple) else acc