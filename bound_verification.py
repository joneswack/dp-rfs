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

def relative_sketch_error(data, D, config, args):
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
    k_target = (data @ data.t())**args.degree
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
    parser.add_argument('--num_samples', type=int, required=False, default=1000)
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

def plot_error_hists(args):
    Ds = [32, 64, 128, 256*1, 256*2, 256*3, 256*4, 2048]

    plt.figure()

    d = 128

    # data = torch.ones(1, d).float()
    data = torch.ones(100, d).float()
    data[0, 1:] = 0 # nnz=2 is worst for TensorSketch
    data[1, 0] = 0
    data[1, 2:] = 0
    data = data / data.norm(dim=1, keepdim=True)

    D=1024

    for config in configs:
        # we generate the same data set for every config
        torch.manual_seed(42)
        errors = torch.zeros(args.num_seeds)

        for j in range(args.num_seeds):
            errors[j] = relative_sketch_error(data, D, config, args).max()

        # we draw a histogram per config
        plt.hist(errors, label=config['name'], bins=100)
    
    # plt.yscale('log')
    plt.legend()
    plt.show()

def plot_error_over_D(args):
    Ds = [32, 64, 128, 256*1, 256*2, 256*3, 256*4, 2048]

    plt.figure()

    d = 128

    data = torch.zeros(200, d).float()
    data[:100, :] = torch.ones(100, d).float()
    # data[0, 1:] = 0
    # data[0,0] = np.sqrt(d)
    # data[1, :] = 0
    # data[1, 1] = np.sqrt(d)
    data[100:, :100] = torch.eye(100)*np.sqrt(d)
    data = data / data.norm(dim=1, keepdim=True)

    for config in configs:
        # we generate the same data set for every config
        torch.manual_seed(42)
        errors = torch.zeros(len(Ds), args.num_seeds)

        for i, D in enumerate(Ds):
            for j in range(args.num_seeds):
                errors[i,j] = relative_sketch_error(data, D, config, args).mean()

        # we draw an error graph for every config
        error_means = errors.mean(dim=1)# .values
        error_stds = errors.std(dim=1)

        print('Config: {}, Std: {}'.format(config['name'], error_stds.numpy()))

        if config['name'] == 'TensorSketch':
            plt.errorbar(Ds, error_means.numpy(), yerr=error_stds.numpy(), label=config['name'] + '(Std: {:2f})'.format(error_stds[-1].item()))
        else:
            plt.plot(Ds, error_means.numpy(), label=config['name'] + '(Std: {:2f})'.format(error_stds[-1].item()))
    
    plt.xticks(Ds)
    plt.xlabel('D')
    plt.ylabel('|k_hat-k|')
    # plt.yscale('log')
    plt.legend()
    plt.show()

def plot_max_error_dataset(args):
    with open(args.datasets_file) as json_file:
        dataset_file = json.load(json_file)
        dataset_path = dataset_file['classification'][0]
    with open(dataset_path) as json_file:
        dataset_file = json.load(json_file)
        X, y = torch.load(dataset_file['train_data'])

    # we only unit-normalize in the other experiments
    # X = X[:,5:] # keep only one-hot data
    X_pad = torch.zeros(len(X), 1024)
    X = X - X.min()
    X = X.view(len(X), -1)
    X_pad[:,:X.shape[1]] = X
    # X_pad = X
    X_pad = X_pad / X_pad.norm(dim=1, keepdim=True)
    X_pad = X_pad[torch.randperm(len(X_pad)), :]
    # concentrations are X.max(dim=1).values

    if args.use_gpu:
        X_pad = X_pad.cuda()

    for config in configs:
        torch.manual_seed(42)
        errors = torch.zeros(args.num_seeds)
        for j in range(args.num_seeds):
            errors[j] = relative_sketch_error(X_pad[:args.num_samples], 2048, config, args).mean()

        error_mean = errors.mean().cpu().item()
        error_std = errors.std().cpu().item()
        print(config['name'], 'Mean: {}'.format(error_mean), 'Std: {}'.format(error_std))


if __name__ == '__main__':
    args = parse_args()

    ### plot over input dimensions
    # plot_max_error_sparsity(args)

    ### analyze adult data set
    # plot_error_over_D(args)
    plot_max_error_dataset(args)
    # plot_error_hists(args)
