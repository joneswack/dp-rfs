from cgi import test
import os
import argparse
import json
import time
import itertools
import collections

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, Dataset, DataLoader
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

from models.vi_gp import VariationalGP

import util.data

from random_features.polynomial_sketch import PolynomialSketch
from random_features.spherical import Spherical
from random_features.rff import RFF

"""
Runs a Gaussian Process Classifier using Random Features with Stochastic Variational Inference (SVI)
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, required=False, default=1000,
                        help='Training batch size')
    parser.add_argument('--epochs', type=int, required=False, default=50,
                        help='Training epochs')
    parser.add_argument('--lr', type=float, required=False, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--num_seeds', type=float, required=False, default=25,
                        help='Number of random seeds')
    parser.add_argument('--use_gpu', dest='use_gpu', action='store_true')
    parser.set_defaults(use_gpu=False)

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_args()

    start_time = time.time()

    dataset_config = 'config/datasets/mnist.json'

    print('Loading dataset: {}'.format(dataset_config))
    data = util.data.load_dataset(dataset_config, standardize=False, maxmin=False, normalize=False)
    data_name, train_data, test_data, train_labels, test_labels = data
    train_labels[train_labels == -1.] = 0
    test_labels[test_labels == -1.] = 0
    min_val = torch.min(train_data, 0)[0]
    train_data = train_data - min_val
    test_data = test_data - min_val
    # normalize data
    # train_data = train_data / torch.max(train_data, 0)[0]
    train_data = train_data / train_data.norm(dim=1, keepdim=True)
    test_data = test_data / test_data.norm(dim=1, keepdim=True)

    # We make the bias trainable...
    train_data = util.data.pad_data_pow_2(train_data)[:, :-1]
    test_data = util.data.pad_data_pow_2(test_data)[:, :-1]

    log_handler = util.data.Log_Handler('poly_vi', '{}'.format(data_name))
    csv_handler = util.data.DF_Handler('poly_vi', '{}'.format(data_name))

    pow_2_shape = int(2**np.ceil(np.log2(train_data.shape[1])))
    # we use D=10d
    D = pow_2_shape * 10
    n_classes = train_labels.shape[1]
    degree = 6
    a = 2
    bias = 1.-2./a**2
    lengthscale = a / np.sqrt(2.)


    print('Comparing approximations...')

    configurations = [
        # weights for degrees (1,2,3,4), h01, has_constant
        {'proj': 'srf', 'full_cov': False, 'complex_weights': False, 'complex_real': False},
        {'proj': 'countsketch_scatter', 'full_cov': False, 'complex_weights': False, 'complex_real': False},
        # {'proj': 'gaussian', 'full_cov': False, 'complex_real': False},
        # {'proj': 'gaussian', 'full_cov': False, 'complex_real': True},
        # {'proj': 'rademacher', 'full_cov': False, 'complex_weights': False, 'complex_real': False},
        # {'proj': 'rademacher', 'full_cov': False, 'complex_weights': False, 'complex_real': True},
        # {'proj': 'srht', 'full_cov': False, 'complex_weights': False, 'complex_real': False},
        # {'proj': 'srht', 'full_cov': False, 'complex_weights': True, 'complex_real': False},
        {'proj': 'srht', 'full_cov': True, 'complex_weights': False, 'complex_real': False},
        {'proj': 'srht', 'full_cov': True, 'complex_weights': False, 'complex_real': True}
    ]

    for seed in range(args.num_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)

        for config in configurations:
            # we double the data dimension at every step

            model_name = 'sgp_{}_proj_{}_deg_{}_compreal{}_2'.format(data_name, config['proj'], degree, config['complex_real'])

            print('Model:', model_name, 'Seed:', seed)

            datasets = {'train': TensorDataset(train_data, train_labels), 'test': TensorDataset(test_data, test_labels)}
            dataloaders = {
                'train': torch.utils.data.DataLoader(datasets['train'], batch_size=args.batch_size,
                                                        shuffle=True, num_workers=0),
                'test': torch.utils.data.DataLoader(datasets['test'], batch_size=args.batch_size,
                                                        shuffle=False, num_workers=0)
            }

            if config['proj'] == 'srf':
                feature_encoder = Spherical(
                    train_data.shape[1], D,
                    lengthscale=1.0, var=1.0, ard=False,
                    discrete_pdf=False, num_pdf_components=10,
                    complex_weights=config['complex_weights'],
                    projection_type=config['proj'],
                    device=('cuda' if args.use_gpu else 'cpu'),
                    trainable_kernel=True
                )
                feature_encoder.load_model('saved_models/poly_a{}.0_p{}_d{}.torch'.format(a, degree, pow_2_shape))
            else:

                feature_encoder = PolynomialSketch(
                    train_data.shape[1], D,
                    degree=degree, bias=bias,
                    var=train_labels.var(),
                    lengthscale=lengthscale,
                    projection_type=config['proj'],
                    complex_weights=config['complex_weights'],
                    complex_real=config['complex_real'],
                    full_complex=False,
                    full_cov=config['full_cov'],
                    convolute_ts=True if config['proj'].startswith('countsketch') else False,
                    trainable_kernel=True,
                    device=('cuda' if args.use_gpu else 'cpu')
                )
            
            with torch.no_grad():
                feature_encoder.resample()

            vgp = VariationalGP(D, n_classes, feature_encoder, trainable_vars=True, covariance='factorized', use_gpu=args.use_gpu)
            if args.use_gpu:
                vgp.cuda()

            # lr = 1e-3 if config['proj'].startswith('countsketch') else 1e-2
            epochs = args.epochs if config['proj'].startswith('countsketch') else 3*args.epochs
            vgp.optimize_lower_bound(model_name, dataloaders['train'], dataloaders['test'], num_epochs=epochs,
                                        lr=args.lr, a=0.5, b=10, gamma=1)

            # for log_dict in log_dicts:
            #     log_handler.append(str(log_dict))
            #     csv_handler.append(log_dict)
            #     csv_handler.save()

    print('Total execution time: {:.2f}'.format(time.time()-start_time))
    print('Done!')