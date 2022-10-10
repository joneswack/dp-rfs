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
from random_features.projections import SRHT, GaussianTransform
from random_features.spherical import Spherical
from random_features.rff import RFF

"""
Runs a Gaussian Process Classifier using Random Features with Stochastic Variational Inference (SVI)
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, required=False, default=1000,
                        help='Training batch size')
    parser.add_argument('--epochs', type=int, required=False, default=150,
                        help='Training epochs')
    parser.add_argument('--lr', type=float, required=False, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--num_seeds', type=float, required=False, default=25,
                        help='Number of random seeds')
    parser.add_argument('--use_gpu', dest='use_gpu', action='store_true')
    parser.set_defaults(use_gpu=False)

    args = parser.parse_args()

    return args


class CRAFTEncoder(torch.nn.Module):
    def __init__(self, up_encoder, e_features, d_features, device='cpu'):
        super(CRAFTEncoder, self).__init__()
        self.up_encoder = up_encoder
        # potential speedup through Gaussian projection
        self.down_encoder = SRHT(e_features, d_features, complex_weights=False, shuffle=True, device=device)
        self.d_features = d_features
        self.device = device

    def resample(self):
        self.up_encoder.resample()
        self.down_encoder.resample()

    def forward(self, x):
        x = self.up_encoder.forward(x)
        x = self.down_encoder.forward(x) / np.sqrt(self.d_features)
        return x


if __name__ == '__main__':

    args = parse_args()

    start_time = time.time()

    dataset_config = 'config/datasets/mnist.json'

    print('Loading dataset: {}'.format(dataset_config))
    data = util.data.load_dataset(dataset_config, standardize=False, maxmin=False, normalize=False)
    data_name, train_data, test_data, train_labels, test_labels = data
    train_labels[train_labels == -1.] = 0
    test_labels[test_labels == -1.] = 0
    min_val = torch.min(train_data)
    train_data = train_data - min_val
    test_data = test_data - min_val
    # normalize data
    # train_data = train_data / torch.max(train_data)
    # test_data = test_data / torch.max(train_data)
    train_data = train_data / train_data.norm(dim=1, keepdim=True)
    test_data = test_data / test_data.norm(dim=1, keepdim=True)

    # mean_inner_product = (train_data[:5000] @ train_data[:5000].t()).mean()

    # We make the bias trainable...
    train_data = util.data.pad_data_pow_2(train_data)[:, :-1]
    test_data = util.data.pad_data_pow_2(test_data)[:, :-1]

    log_handler = util.data.Log_Handler('poly_vi', '{}'.format(data_name))
    csv_handler = util.data.DF_Handler('poly_vi', '{}'.format(data_name))

    pow_2_shape = int(2**np.ceil(np.log2(train_data.shape[1])))
    # we use D=10d
    # D = pow_2_shape * 10
    D = 2**13
    E = 2**15
    n_classes = train_labels.shape[1]
    degree = 7
    a = 2
    bias = 1.-2./a**2
    lengthscale = a / np.sqrt(2.)


    print('Comparing approximations...')

    configurations = [
        {'proj': 'srf', 'full_cov': False, 'complex_weights': False, 'complex_real': False, 'craft': False, 'ard': True},
        # weights for degrees (1,2,3,4), h01, has_constant
        {'proj': 'countsketch_scatter', 'full_cov': False, 'complex_weights': False, 'complex_real': False, 'craft': False, 'ard': True},
        {'proj': 'countsketch_scatter', 'full_cov': False, 'complex_weights': False, 'complex_real': False, 'craft': True, 'ard': True},
        # {'proj': 'gaussian', 'full_cov': False, 'complex_real': False},
        # {'proj': 'gaussian', 'full_cov': False, 'complex_real': True},
        # {'proj': 'rademacher', 'full_cov': False, 'complex_weights': False, 'complex_real': False},
        # {'proj': 'rademacher', 'full_cov': False, 'complex_weights': False, 'complex_real': True},
        {'proj': 'srht', 'full_cov': False, 'complex_weights': False, 'complex_real': False, 'craft': False, 'ard': True},
        {'proj': 'srht', 'full_cov': False, 'complex_weights': False, 'complex_real': True, 'craft': False, 'ard': True},
        {'proj': 'srht', 'full_cov': True, 'complex_weights': False, 'complex_real': False, 'craft': True, 'ard': True},
        {'proj': 'srht', 'full_cov': True, 'complex_weights': False, 'complex_real': True, 'craft': True, 'ard': True}
    ]

    for seed in range(args.num_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)

        for config in configurations:
            # we double the data dimension at every step

            model_name = 'sgp_{}_proj_{}_deg_{}_compreal_{}_craft_{}_ard_{}_norm_nocache_t4'.format(
                data_name, config['proj'], degree, config['complex_real'], config['craft'], config['ard'])

            print('Model:', model_name, 'Seed:', seed)

            datasets = {'train': TensorDataset(train_data, train_labels), 'test': TensorDataset(test_data, test_labels)}
            dataloaders = {
                'train': torch.utils.data.DataLoader(datasets['train'], batch_size=args.batch_size,
                                                        shuffle=True, num_workers=0),
                'test': torch.utils.data.DataLoader(datasets['test'], batch_size=args.batch_size,
                                                        shuffle=False, num_workers=0)
            }

            if config['proj'] == 'srf':
                up_encoder = Spherical(
                    train_data.shape[1], E if config['craft'] else D,
                    lengthscale=1.0,
                    var=train_labels.var(),
                    discrete_pdf=False, num_pdf_components=10,
                    complex_weights=config['complex_weights'],
                    projection_type=config['proj'],
                    trainable_kernel=True,
                    ard=False,
                    device=('cuda' if args.use_gpu else 'cpu'),
                )
                up_encoder.load_model('saved_models/poly_a{}.0_p{}_d{}.torch'.format(a, degree, pow_2_shape))
                if config['ard']:
                    up_encoder.log_lengthscale = torch.nn.Parameter(
                        torch.ones(up_encoder.d_in, device=up_encoder.device) * up_encoder.log_lengthscale.cpu().item(),
                        requires_grad=True
                )
            else:

                up_encoder = PolynomialSketch(
                    train_data.shape[1], E if config['craft'] else D,
                    degree=degree,
                    bias=1.0, # for non-unit norm data
                    var=train_labels.var(), # train_labels.var(),
                    lengthscale=1.0, # for non-unit norm data
                    projection_type=config['proj'],
                    complex_weights=config['complex_weights'],
                    complex_real=config['complex_real'],
                    full_cov=config['full_cov'],
                    trainable_kernel=True,
                    ard=config['ard'],
                    device=('cuda' if args.use_gpu else 'cpu')
                )

            if config['craft']:
                feature_encoder = CRAFTEncoder(up_encoder, E, D, device=up_encoder.device)
            else:
                feature_encoder = up_encoder
            
            with torch.no_grad():
                if config['proj'] == 'srf':
                    feature_encoder.resample(num_points_w=5000)
                else:
                    feature_encoder.resample()

            vgp = VariationalGP(D, n_classes, feature_encoder, trainable_vars=True, covariance='factorized', use_gpu=args.use_gpu)
            if args.use_gpu:
                vgp.cuda()

            # lr = 1e-3 if config['proj'].startswith('countsketch') else 1e-2
            epochs = 3 * args.epochs if config['proj'].startswith('srf') else args.epochs
            # epochs = args.epochs
            vgp.optimize_lower_bound(model_name, dataloaders['train'], dataloaders['test'], num_epochs=epochs,
                                        lr=args.lr, a=0.5, b=10, gamma=1)

            # for log_dict in log_dicts:
            #     log_handler.append(str(log_dict))
            #     csv_handler.append(log_dict)
            #     csv_handler.save()

    print('Total execution time: {:.2f}'.format(time.time()-start_time))
    print('Done!')