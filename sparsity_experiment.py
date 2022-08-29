import argparse
import time
import json

import torch
import numpy as np
import matplotlib.pyplot as plt

from random_features.polynomial_sketch import PolynomialSketch

"""
Replication of Matlab experiment from
https://github.com/google-research/google-research/blob/master/poly_kernel_sketch/matlab/MaxError.m
"""

def sketch_error(data, D, config, args):
    comp_real = config['complex_real'] if 'complex_real' in config.keys() else False
    full_cov = config['full_cov'] if 'full_cov' in config.keys() else False

    feature_encoder = PolynomialSketch(
        data.shape[1], D,
        degree=args.degree, bias=args.bias,
        projection_type=config['proj'], hierarchical=config['hierarchical'],
        complex_weights=config['complex_weights'], complex_real=comp_real,
        full_cov=full_cov, lengthscale=1.0,
        device=('cuda' if args.use_gpu else 'cpu'),
        var=1.0, ard=False, trainable_kernel=False
    )
    feature_encoder.resample()

    features = feature_encoder.forward(data)
    k_hat = features @ features.t()
    k_target = data @ data.t()
    norms = data.norm(dim=1, keepdim=True) * data.norm(dim=1, keepdim=True).t()
    # relative error with ||x||^p because ||x^(p)||=||x||^p
    # i.e. we bound Pr{ |k_hat - k| >= e ||x|| ||y|| } <= \delta
    k_error = (k_hat - k_target).abs() / norms**args.degree

    return k_error


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rf_parameter_file', type=str, required=False, default='config/rf_parameters/poly3_ctr.json',
                        help='Path to RF parameter file')
    parser.add_argument('--datasets_file', type=str, required=False, default='config/active_datasets3.json',
                        help='List of datasets to be used for the experiments')
    parser.add_argument('--degree', type=int, required=False, default=2)
    parser.add_argument('--bias', type=int, required=False, default=0)
    parser.add_argument('--max_D', type=int, required=False, default=10000)
    parser.add_argument('--max_d_in', type=int, required=False, default=40,
                        help='Number of data samples for lengthscale estimation')
    parser.add_argument('--num_steps', type=int, required=False, default=20)
    parser.add_argument('--num_seeds', type=int, required=False, default=1000,
                        help='Number of seeds (runs)')
    parser.add_argument('--use_gpu', dest='use_gpu', action='store_true')
    parser.set_defaults(use_gpu=False)

    args = parser.parse_args()

    return args

configs = [
    {'name': 'TensorSketch', 'proj': 'countsketch_scatter', 'full_cov': False, 'complex_weights': False, 'complex_real': False, 'hierarchical': False},
    {'name': 'Rademacher', 'proj': 'rademacher', 'full_cov': False, 'complex_weights': False, 'complex_real': False, 'hierarchical': False},
    {'name': 'CtR-Rademacher', 'proj': 'rademacher', 'full_cov': False, 'complex_weights': True, 'complex_real': True, 'hierarchical': False},
    {'name': 'ProductSRHT', 'proj': 'srht', 'full_cov': False, 'complex_weights': False, 'complex_real': False, 'hierarchical': False},
    {'name': 'CtR-ProductSRHT', 'proj': 'srht', 'full_cov': False, 'complex_weights': True, 'complex_real': True, 'hierarchical': False},
]

def plot_max_error_sparsity(args):
    # errors = [[list()] * args.num_seeds] * args.num_steps
    step_size = args.max_d_in // args.num_steps
    steps = list(range(step_size, args.max_d_in+step_size, step_size))
    concentrations = list(range(1, 15, 1))
    # steps = [2, 4, 8, 16, 32] # , 64, 128
    # steps = [32,64]

    plt.figure()

    for config in configs:
        # we generate the same data set for every config
        torch.manual_seed(42)
        errors = torch.zeros(len(concentrations), args.num_seeds)

        for i, concentration in enumerate(concentrations):
            data = torch.zeros(1024, 64)
            data[:, :concentration+1] = 1.
            for k in range(len(data)):
                data[k,:] = data[k, torch.randperm(len(data[k]))]

            # data = data / data.norm(dim=1, keepdim=True)
            
            for j in range(args.num_seeds):
                errors[i,j] = sketch_error(data, 64, config, args).max()

        # we draw an error graph for every config
        error_means = errors.mean(dim=1)
        error_stds = errors.std(dim=1)
        plt.errorbar(concentrations, error_means.numpy(), yerr=error_stds.numpy(), label=config['name'])
    
    plt.xticks(concentrations)
    plt.xlabel('# non-zeros')
    plt.ylabel('Max. Error')
    # plt.yscale('log')
    plt.legend()
    plt.show()

def plot_max_error_adult(args):
    with open(args.datasets_file) as json_file:
        dataset_file = json.load(json_file)
        dataset_path = dataset_file['classification'][0]
    with open(dataset_path) as json_file:
        dataset_file = json.load(json_file)
        X, y = torch.load(dataset_file['train_data'])

    # we only unit-normalize in the other experiments
    # X = X[:,5:] # keep only one-hot data
    # X_pad = torch.zeros(len(X), 128)
    # X_pad[:,:X.shape[1]] = X
    X_pad = X
    X_pad = X_pad / X_pad.norm(dim=1, keepdim=True)
    X_pad = X_pad[torch.randperm(len(X_pad)), :]
    # concentrations are X.max(dim=1).values
    for config in configs:
        torch.manual_seed(42)
        errors = torch.zeros(args.num_seeds)
        for j in range(args.num_seeds):
            errors[j] = sketch_error(X_pad[:1000], 64, config, args).mean()

        error_mean = errors.mean()
        error_std = errors.std()
        print(config['name'], 'Mean: {}'.format(error_mean), 'Std: {}'.format(error_std))


if __name__ == '__main__':
    args = parse_args()

    ### plot over input dimensions
    # plot_max_error_sparsity(args)

    ### analyze adult data set
    plot_max_error_adult(args)
