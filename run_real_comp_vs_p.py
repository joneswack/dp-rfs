import argparse
import time
import json

import torch
import numpy as np
import matplotlib.pyplot as plt

from random_features.polynomial_sketch import PolynomialSketch

def sketch_error(x, y, D, degree, config):
    comp_real = config['complex_real'] if 'complex_real' in config.keys() else False
    full_cov = config['full_cov'] if 'full_cov' in config.keys() else False

    feature_encoder = PolynomialSketch(
        x.shape[1], D,
        degree=degree, bias=0,
        projection_type=config['proj'], hierarchical=config['hierarchical'],
        complex_weights=config['complex_weights'], complex_real=comp_real,
        full_cov=full_cov, lengthscale=1.0,
        device=('cuda' if args.use_gpu else 'cpu'),
        var=1.0, ard=False, trainable_kernel=False
    )
    feature_encoder.resample()

    features_x = feature_encoder.forward(x)
    features_y = feature_encoder.forward(y)
    k_hat = features_x @ features_y.conj().t()
    k_target = x @ y.t()
    # norms = x.norm(dim=1, keepdim=True) * y.norm(dim=1, keepdim=True).t()

    k_error = (k_hat - k_target).abs()

    return k_error

configs = [
    {'name': 'TensorSketch', 'proj': 'countsketch_scatter', 'full_cov': False, 'complex_weights': False, 'complex_real': False, 'hierarchical': False},
    {'name': 'Rademacher', 'proj': 'rademacher', 'full_cov': False, 'complex_weights': False, 'complex_real': False, 'hierarchical': False},
    {'name': 'Complex Rademacher', 'proj': 'rademacher', 'full_cov': False, 'complex_weights': True, 'complex_real': False, 'hierarchical': False},
    {'name': 'Hier. Rademacher', 'proj': 'rademacher', 'full_cov': False, 'complex_weights': True, 'complex_real': False, 'hierarchical': True},
    #{'name': 'ProductSRHT', 'proj': 'srht', 'full_cov': False, 'complex_weights': False, 'complex_real': False, 'hierarchical': False},
    #{'name': 'CtR-ProductSRHT', 'proj': 'srht', 'full_cov': False, 'complex_weights': True, 'complex_real': True, 'hierarchical': False},
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_degree', type=int, required=False, default=2,
                        help='The maximum degree for which to measure the error')
    parser.add_argument('--num_seeds', type=int, required=False, default=100,
                        help='Number of seeds (runs)')
    parser.add_argument('--use_gpu', dest='use_gpu', action='store_true')
    parser.set_defaults(use_gpu=False)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    # also add different angles later on
    x = torch.ones([1, 1000], dtype=torch.float32) / np.sqrt(1000)
    y = torch.ones([1, 1000], dtype=torch.float32) / np.sqrt(1000)

    config_error_dict = {}

    for config in configs:
        torch.manual_seed(42)

        errors = torch.zeros(args.max_degree, args.num_seeds)

        for degree in range(1, args.max_degree+1):
            for j in range(args.num_seeds):
                errors[degree-1, j] = sketch_error(x, y, 5000, degree, config)

            error_mean = errors[degree-1].mean()
            error_std = errors[degree-1].std()
            print(config['name'], 'Mean: {}'.format(error_mean), 'Std: {}'.format(error_std))