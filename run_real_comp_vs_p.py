import argparse
import time
import json

import torch
import numpy as np
import matplotlib.pyplot as plt

import util.data

from random_features.polynomial_sketch import PolynomialSketch

def sketch_error(x, y, D, degree, config, device):
    comp_real = config['complex_real'] if 'complex_real' in config.keys() else False
    full_cov = config['full_cov'] if 'full_cov' in config.keys() else False

    feature_encoder = PolynomialSketch(
        x.shape[1], D,
        degree=degree, bias=0,
        projection_type=config['proj'], ahle=config['ahle'], tree=config['tree'],
        complex_weights=config['complex_weights'], complex_real=comp_real,
        full_cov=full_cov, lengthscale=1.0,
        device=device,
        var=1.0, ard=False, trainable_kernel=False
    )
    feature_encoder.resample()

    features_x = feature_encoder.forward(x)
    features_y = feature_encoder.forward(y)
    k_hat = torch.sum(features_x * features_y.conj(), dim=-1)
    k_target = (x * y).sum(axis=-1)**degree

    absolute_errors = (k_hat - k_target).abs()
    squared_errors = (k_hat - k_target).abs().pow(2)

    return absolute_errors, squared_errors

configs = [
    #{'name': 'TensorSketch', 'proj': 'countsketch_scatter', 'full_cov': False, 'complex_weights': False, 'complex_real': False, 'hierarchical': False},
    {'name': 'Gaussian', 'proj': 'gaussian', 'full_cov': False, 'complex_weights': False, 'complex_real': False, 'ahle': False, 'tree': False},
    {'name': 'Complex Gaussian', 'proj': 'gaussian', 'full_cov': False, 'complex_weights': True, 'complex_real': False, 'ahle': False, 'tree': False},
    {'name': 'Rademacher', 'proj': 'rademacher', 'full_cov': False, 'complex_weights': False, 'complex_real': False, 'ahle': False, 'tree': False},
    {'name': 'Complex Rademacher', 'proj': 'rademacher', 'full_cov': False, 'complex_weights': True, 'complex_real': False, 'ahle': False, 'tree': False},
    {'name': 'Ahle et al. Tree Rademacher', 'proj': 'rademacher', 'full_cov': False, 'complex_weights': False, 'complex_real': False, 'ahle': True, 'tree': True},
    {'name': 'Ahle et al. Tree Rademacher Comp.', 'proj': 'rademacher', 'full_cov': False, 'complex_weights': True, 'complex_real': False, 'ahle': True, 'tree': True},
    {'name': 'Ahle et al. Rademacher', 'proj': 'rademacher', 'full_cov': False, 'complex_weights': False, 'complex_real': False, 'ahle': True, 'tree': False},
    {'name': 'Ahle et al. Rademacher Comp.', 'proj': 'rademacher', 'full_cov': False, 'complex_weights': True, 'complex_real': False, 'ahle': True, 'tree': False},
    #{'name': 'ProductSRHT', 'proj': 'srht', 'full_cov': False, 'complex_weights': False, 'complex_real': False, 'hierarchical': False},
    #{'name': 'CtR-ProductSRHT', 'proj': 'srht', 'full_cov': False, 'complex_weights': True, 'complex_real': True, 'hierarchical': False},
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_degree', type=int, required=False, default=9,
                        help='The maximum degree for which to measure the error')
    parser.add_argument('--num_seeds', type=int, required=False, default=1000,
                        help='Number of seeds (runs)')
    parser.add_argument('--num_features', type=int, required=False, default=512,
                        help='Number of random features')
    parser.add_argument('--use_gpu', dest='use_gpu', action='store_true')
    parser.set_defaults(use_gpu=False)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    device = ('cuda' if args.use_gpu else 'cpu')

    csv_handler = util.data.DF_Handler('error_over_p', 'ones_D{}_reps{}'.format(args.num_features, args.num_seeds))

    # also add different angles later on
    #x = torch.randn([1000, 10], dtype=torch.float32, device=device) #.abs()
    #y = torch.randn([1000, 10], dtype=torch.float32, device=device) #.abs()
    x = torch.ones([1000, 10], dtype=torch.float32, device=device)
    y = torch.ones([1000, 10], dtype=torch.float32, device=device)
    #x[:, 5:] = 0
    #y[:, :5] = 0
    x /= x.norm(dim=-1, keepdim=True)
    y /= y.norm(dim=-1, keepdim=True)

    config_error_dict = {}

    for config in configs:
        torch.manual_seed(42)

        for degree in range(1, args.max_degree+1):

            all_abs_errors = []
            all_abs_squared_errors = []
            for j in range(args.num_seeds):
                abs_errors, abs_squared_errors = sketch_error(x, y, args.num_features, degree, config, device)
                all_abs_errors.append(abs_errors)
                all_abs_squared_errors.append(abs_squared_errors)

            mae = torch.cat(all_abs_errors).mean()
            mse = torch.cat(all_abs_squared_errors).mean()
            abs_err_std = torch.cat(all_abs_errors).std()
            abs_sq_err_std = torch.cat(all_abs_squared_errors).std()
            print(config['name'], 'MAE: {}'.format(mae), 'Std: {}'.format(abs_err_std))

            log_dir = {
                'name': config['name'],
                'p': degree,
                'proj': config['proj'],
                'full_cov': config['full_cov'],
                'complex_weights': config['complex_weights'],
                'complex_real': config['complex_real'],
                'ahle': config['ahle'],
                'tree': config['tree'],
                'mae': mae.item(),
                'mse': mse.item(),
                'abs_err_std': abs_err_std.item(),
                'abs_sq_err_std': abs_sq_err_std.item()
            }

            csv_handler.append(log_dir)
            csv_handler.save()


    import matplotlib.pyplot as plt
    import pandas as pd
    #plt.style.use('seaborn')
    #plt.style.use('ggplot')

    cm = plt.get_cmap('tab20')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    NUM_COLORS = len(configs)
    ax.set_prop_cycle('color', [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])

    df = pd.read_csv(csv_handler.file_path)

    for config in configs:
        ax.plot(list(range(1, args.max_degree+1)), df.loc[df['name']==config['name'], 'mae'], label=config['name'])

    plt.yscale('log')
    plt.legend()
    plt.show()