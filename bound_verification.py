import argparse
import time
import json

import torch
import numpy as np
import matplotlib.pyplot as plt
# import multiprocessing

from random_features.polynomial_sketch import PolynomialSketch
from random_features.spherical import Spherical
import util.data

"""
Replication of Matlab experiment from
https://github.com/google-research/google-research/blob/master/poly_kernel_sketch/matlab/MaxError.m
"""

def sketch_error(data, D, config, args, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    comp_real = config['complex_real'] if 'complex_real' in config.keys() else False
    full_cov = config['full_cov'] if 'full_cov' in config.keys() else False

    if config['name'] == 'SRF':
        feature_encoder = Spherical(data.shape[1], D,
                            lengthscale=1.0, var=1.0, ard=False,
                            discrete_pdf=False, num_pdf_components=10,
                            complex_weights=config['complex_weights'],
                            projection_type=config['proj'], device=('cuda' if args.use_gpu else 'cpu'))
        feature_encoder.load_model('saved_models/poly_a{}.0_p{}_d{}.torch'.format(args.a, args.degree, data.shape[1]))
    else:
        feature_encoder = PolynomialSketch(
            data.shape[1], D,
            degree=args.degree, bias=args.bias,
            projection_type=config['proj'], hierarchical=config['hierarchical'],
            complex_weights=config['complex_weights'], complex_real=comp_real,
            full_cov=full_cov, lengthscale=1.0,
            device=('cuda' if args.use_gpu else 'cpu'),
            var=1.0, ard=False, trainable_kernel=False
        )
    
    if config['name'] == 'SRF':
        feature_encoder.resample(num_points_w=5000)
    else:
        feature_encoder.resample()

    features = feature_encoder.forward(data)
    k_hat = features @ features.t()
    k_target = (data @ data.t())**args.degree
    k_error = (k_hat - k_target).abs()

    return k_error


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rf_parameter_file', type=str, required=False, default='config/rf_parameters/poly3_ctr.json',
                        help='Path to RF parameter file')
    parser.add_argument('--datasets_file', type=str, required=False, default='config/active_datasets3.json',
                        help='List of datasets to be used for the experiments')
    parser.add_argument('--degree', type=int, required=False, default=3)
    parser.add_argument('--a', type=int, required=False, default=2)
    parser.add_argument('--bias', type=int, required=False, default=0)
    parser.add_argument('--max_D', type=int, required=False, default=10000)
    parser.add_argument('--max_d_in', type=int, required=False, default=40,
                        help='Number of data samples for lengthscale estimation')
    parser.add_argument('--num_steps', type=int, required=False, default=20)
    parser.add_argument('--num_seeds', type=int, required=False, default=30000,
                        help='Number of seeds (runs)')
    parser.add_argument('--num_samples', type=int, required=False, default=1000)
    parser.add_argument('--use_gpu', dest='use_gpu', action='store_true')
    parser.set_defaults(use_gpu=False)

    args = parser.parse_args()

    return args

configs = [
    {'name': 'TensorSketch', 'proj': 'countsketch_scatter', 'full_cov': False, 'complex_weights': False, 'complex_real': False, 'hierarchical': False},
    # {'name': 'SRF', 'proj': 'srf', 'full_cov': False, 'complex_weights': False, 'complex_real': False, 'hierarchical': False},
    {'name': 'Rademacher', 'proj': 'rademacher', 'full_cov': False, 'complex_weights': False, 'complex_real': False, 'hierarchical': False},
    {'name': 'CtR-Rademacher', 'proj': 'rademacher', 'full_cov': False, 'complex_weights': True, 'complex_real': True, 'hierarchical': False},
    {'name': 'ProductSRHT', 'proj': 'srht', 'full_cov': True, 'complex_weights': False, 'complex_real': False, 'hierarchical': False},
    {'name': 'CtR-ProductSRHT', 'proj': 'srht', 'full_cov': True, 'complex_weights': True, 'complex_real': True, 'hierarchical': False},
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
            errors[j] = sketch_error(data, D, config, args).max()

        # we draw a histogram per config
        plt.hist(errors, label=config['name'], bins=100)
    
    # plt.yscale('log')
    plt.legend()
    plt.show()

def sketch_error_seed(data, D, config, args, seed):
    return sketch_error(data, D, config, args, seed).view(-1) # .mean()

def plot_error_over_D(args):

    d = 2

    log_handler = util.data.Log_Handler('bound_plot', 'over_D_samples_{}_seeds_{}_d{}'.format(args.num_samples, args.num_seeds, d))
    csv_handler = util.data.DF_Handler('bound_plot', 'over_D_samples_{}_seeds_{}_d{}'.format(args.num_samples, args.num_seeds, d))

    # Ds = [32, 64, 128, 256*1, 256*2, 256*3, 256*4, 2048]
    Ds = [i*64 for i in range(1, 30)]
    # Ds = [896]

    plt.figure()

    # data = torch.zeros(100, d).float()
    # data[:100, :] = torch.ones(100, d).float()
    # data[0, 1:] = 0
    # data[0,0] = np.sqrt(d)
    # data[1, :] = 0
    # data[1, 1] = np.sqrt(d)
    # data[100:, :100] = torch.eye(100)*np.sqrt(d)
    # data[:,:100] = torch.eye(100)
    # bias = 1.-2./args.a**2
    # lengthscale = args.a / np.sqrt(2.)

    data = torch.zeros(1, 64)
    data[:, :d] = 1.0
    data = data / data.norm(dim=1, keepdim=True)

    norms = (data.norm(dim=1, keepdim=True) * data.norm(dim=1, keepdim=True).t()).double()

    # data = data / lengthscale
    # data[:, -1] = np.sqrt(bias)

    # pool = multiprocessing.Pool(4)
    if args.use_gpu:
        data = data.cuda()

    for config in configs:
        # we generate the same data set for every config
        # errors = torch.zeros(len(Ds), args.num_seeds, device=('cuda' if args.use_gpu else 'cpu'))

        for i, D in enumerate(Ds):
            # data, D, config, args, seed
            # output = pool.starmap(sketch_error_seed, zip(
            #     [data]*args.num_seeds,
            #     [D]*args.num_seeds,
            #     [config]*args.num_seeds,
            #     [args]*args.num_seeds,
            #     list(range(0, args.num_seeds, 1))
            # ))

            # errors = torch.hstack(output)

            errors = torch.zeros(args.num_seeds)

            for j, seed in enumerate(range(args.num_seeds)):
                errors[j] = sketch_error_seed(data, D, config, args, seed)

            # relative error with ||x||^p because ||x^(p)||=||x||^p
            # i.e. we bound Pr{ |k_hat - k| >= e ||x|| ||y|| } <= \delta
            errors = errors / norms**args.degree

            error_prob = (errors > 0.25).float().mean().item()

            log_dir = {
                'proj': config['proj'],
                'complex_real': config['complex_real'],
                'hierarchical': config['hierarchical'],
                'D': D,
                'error_prob': error_prob
            }

            log_handler.append(log_dir)
            csv_handler.append(log_dir)
            csv_handler.save()


def plot_error_over_sparsity(args):
    # concentrations = [2, 4, 8, 16, 32, 64] # list(range(2, 15, 2))

    log_handler = util.data.Log_Handler('bound_plot', 'over_sparsity_samples_{}_seeds_{}'.format(args.num_samples, args.num_seeds))
    csv_handler = util.data.DF_Handler('bound_plot', 'over_sparsity_samples_{}_seeds_{}'.format(args.num_samples, args.num_seeds))

    # Ds = [32, 64, 128, 256*1, 256*2, 256*3, 256*4, 2048]
    ds = [2,3,4,5,6,7,8,9,10]
    D = 1024
    # Ds = [896]

    plt.figure()

    # data = torch.zeros(100, d).float()
    # data[:100, :] = torch.ones(100, d).float()
    # data[0, 1:] = 0
    # data[0,0] = np.sqrt(d)
    # data[1, :] = 0
    # data[1, 1] = np.sqrt(d)
    # data[100:, :100] = torch.eye(100)*np.sqrt(d)
    # data[:,:100] = torch.eye(100)
    # bias = 1.-2./args.a**2
    # lengthscale = args.a / np.sqrt(2.)

    # data = data / lengthscale
    # data[:, -1] = np.sqrt(bias)

    pool = multiprocessing.Pool(4)
    if args.use_gpu:
        data = data.cuda()

    for config in configs:
        # we generate the same data set for every config
        # errors = torch.zeros(len(Ds), args.num_seeds, device=('cuda' if args.use_gpu else 'cpu'))

        for i, d in enumerate(ds):
            # data, D, config, args, seed
            data = torch.zeros(1, 64)
            data[:, :d] = 1.0
            data = data / data.norm(dim=1, keepdim=True)

            norms = (data.norm(dim=1, keepdim=True) * data.norm(dim=1, keepdim=True).t()).double()

            output = pool.starmap(sketch_error_seed, zip(
                [data]*args.num_seeds,
                [D]*args.num_seeds,
                [config]*args.num_seeds,
                [args]*args.num_seeds,
                list(range(0, args.num_seeds, 1))
            ))

            errors = torch.hstack(output)

            # relative error with ||x||^p because ||x^(p)||=||x||^p
            # i.e. we bound Pr{ |k_hat - k| >= e ||x|| ||y|| } <= \delta
            errors = errors / norms**args.degree

            error_prob = (errors > 0.25).float().mean().item()

            log_dir = {
                'proj': config['proj'],
                'complex_real': config['complex_real'],
                'hierarchical': config['hierarchical'],
                'D': D,
                'd': d,
                'error_prob': error_prob
            }

            log_handler.append(log_dir)
            csv_handler.append(log_dir)
            csv_handler.save()


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
            # mean error
            errors[j] = sketch_error(X_pad[:args.num_samples], 2048, config, args).mean()

        error_mean = errors.mean().cpu().item()
        error_std = errors.std().cpu().item()
        print(config['name'], 'Mean: {}'.format(error_mean), 'Std: {}'.format(error_std))


if __name__ == '__main__':
    args = parse_args()

    ### plot over input dimensions
    # plot_max_error_sparsity(args)

    ### analyze adult data set
    # plot_error_over_sparsity(args)
    plot_error_over_D(args)

