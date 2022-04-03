import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import torch
import pandas as pd
import os

from tqdm import tqdm
from timeit import default_timer as timer
import argparse

import scipy.io

import sys 
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import util.data
from util.data import DF_Handler
from util.kernels import gaussian_kernel
from util.helper_functions import optimize_marginal_likelihood, kl_factorized_gaussian, regression_scores
from util.measures import Exponential_Measure
from models.het_gp import HeteroskedasticGP, predictive_dist_exact
from random_features.gaussian_approximator import GaussianApproximator


configs = [
    {'name': 'Random Fourier Features', 'method': 'rff', 'proj': 'gaussian', 'degree': 4, 'hierarchical': False, 'complex_weights': False, 'complex_real': False},
    {'name': 'Struct. Orth. RFF', 'method': 'rff', 'proj': 'srht', 'degree': 4, 'hierarchical': False, 'complex_weights': False, 'complex_real': False},
    # {'method': 'rff', 'proj': 'gaussian', 'degree': 4, 'bias': 0, 'lengthscale': True, 'hierarchical': False, 'complex_weights': True},
    # {'method': 'maclaurin_exp_h01', 'proj': 'rademacher', 'degree': 15, 'hierarchical': False, 'complex_weights': False, 'complex_real': False},
    {'name': 'Maclaurin Radem. using $\\hat{{k}}_p$', 'method': 'maclaurin', 'proj': 'rademacher', 'degree': 15, 'hierarchical': False, 'complex_weights': False, 'complex_real': False, 'single_cluster': True},
    {'name': 'Maclaurin Radem. using $\\hat{{k}}_p^*$ (this work)', 'method': 'maclaurin', 'proj': 'rademacher', 'degree': 15, 'hierarchical': False, 'complex_weights': False, 'complex_real': False},
    {'name': 'Maclaurin T.SRHT using $\\hat{{k}}_p$', 'method': 'maclaurin', 'proj': 'srht', 'degree': 15, 'hierarchical': False, 'complex_weights': False, 'complex_real': False, 'single_cluster': True},
    {'name': 'Maclaurin T.SRHT using $\\hat{{k}}_p^*$ (this work)', 'method': 'maclaurin', 'proj': 'srht', 'degree': 15, 'hierarchical': False, 'complex_weights': False, 'complex_real': False},
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--house_price_file', type=str, required=False,
                        default='../datasets/export/uk_house_prices/pp-monthly-update-new-version.csv',
                        help='CSV File with the UK house prices')
    parser.add_argument('--post_code_file', type=str, required=False,
                        default='../datasets/export/uk_house_prices/ukpostcodes.csv',
                        help='CSV File with the UK house prices')
    parser.add_argument('--sarcos_dir', type=str, required=False,
                        default='../datasets/export/sarcos',
                        help='Directory of sarcos dataset')
    parser.add_argument('--kin40k_dir', type=str, required=False,
                        default='../datasets/export/kin40k',
                        help='Directory of kin-40k dataset')
    parser.add_argument('--dataset', choices=['uk_house_prices', 'sarcos', 'kin40k', 'uci'], required=False, default='sarcos',
                        help='Which dataset to evaluate')
    parser.add_argument('--uci_path', type=str, required=False, default='config/datasets/yacht.json',
                        help='Path to UCI dataset json file')
    parser.add_argument('--csv_dir', type=str, required=False,
                        default='csv', help='Directory to save CSV files to')
    parser.add_argument('--figure_dir', type=str, required=False,
                        default='figures', help='Directory to save CSV files to')
    parser.add_argument('--num_seeds', type=int, required=False, default=10,
                        help='Number of seeds (runs)')
    parser.add_argument('--num_train_samples', type=int, required=False, default=10000, # 10000
                        help='Number of data samples for training')
    parser.add_argument('--num_grid_samples', type=int, required=False, default=100,
                        help='Number of elements for each grid dimension')
    parser.add_argument('--num_lml_samples', type=int, required=False, default=5000,
                        help='Number of data samples for likelihood optimization')
    parser.add_argument('--lml_lr', type=float, required=False, default=1e-1, # 1e-1
                        help='Learning rate for likelihood optimization')
    parser.add_argument('--lml_iterations', type=int, required=False, default=20, # 20
                        help='Number of iterations for likelihood optimization')
    parser.add_argument('--num_dist_est_samples', type=int, required=False, default=500,
                        help='Number of datapoints used to estimate maclaurin distribution')
    parser.add_argument('--num_rfs', type=int, required=False, default=100,
                        help='Number of random features')
    parser.add_argument('--num_clusters', type=int, required=False, default=10000,
                        help='Number of random clusters')
    parser.add_argument('--cluster_method', choices=['random', 'farthest'], required=False, default='farthest',
                        help='Clustering method')
    parser.add_argument('--cluster_train', dest='cluster_train', action='store_true')
    parser.set_defaults(cluster_train=True)
    parser.add_argument('--use_gpu', dest='use_gpu', action='store_true')
    parser.set_defaults(use_gpu=False)

    args = parser.parse_args()

    return args

def prepare_house_price_data(args):
    house_prices_df = pd.read_csv(args.house_price_file, header=None)

    columns = [
        'trans_id',
        'price',
        'transfer_date',
        'postcode',
        'property_type',
        'old_new',
        'duration',
        'paon',
        'saon',
        'street',
        'locallity',
        'city',
        'district',
        'county',
        'ppd',
        'rec_status'
    ]
    house_prices_df.columns = columns

    post_code_df = pd.read_csv(args.post_code_file, header=None, index_col=0)
    post_code_df.columns = ['postcode', 'lat', 'lon']

    result = pd.merge(house_prices_df, post_code_df, how="inner", on=['postcode'])

    result = result[~result['lat'].isnull()]
    result = result[result['property_type']=='F']

    data = torch.from_numpy(result[['lon', 'lat']].values).float()
    labels = torch.from_numpy(result['price'].values).log().unsqueeze(1).float()

    return data, labels

def prepare_uci_data(args):
    data = util.data.load_dataset(args.uci_path, standardize=False, maxmin=False, normalize=False, split_size=0.9)
    data_name, train_data, test_data, train_labels, test_labels = data

    train_mean = train_data.mean(dim=0)
    train_std = train_data.std(dim=0)
    train_std[train_std==0] = 1.

    train_data = (train_data - train_mean) / train_std
    test_data = (test_data - train_mean) / train_std


    # pad to power of 2
    train_data = util.data.pad_data_pow_2(train_data, offset=0)
    test_data = util.data.pad_data_pow_2(test_data, offset=0)

    return torch.vstack([train_data, test_data]), torch.vstack([train_labels, test_labels])

def prepare_sarcos_data(args):

    train = scipy.io.loadmat(os.path.join(args.sarcos_dir, 'sarcos_inv.mat'))['sarcos_inv']
    test = scipy.io.loadmat(os.path.join(args.sarcos_dir, 'sarcos_inv_test.mat'))['sarcos_inv_test']

    # we use only the first of the 7 joint torques
    # TODO: maybe standardize the data
    train_data = torch.from_numpy(train[:,:21]).float()
    train_labels = torch.from_numpy(train[:, 21][:, np.newaxis]).float()
    test_data = torch.from_numpy(test[:,:21]).float()
    test_labels = torch.from_numpy(test[:, 21][:, np.newaxis]).float()


    train_mean = train_data.mean(dim=0)
    train_std = train_data.std(dim=0)
    train_std[train_std==0] = 1.

    train_data = (train_data - train_mean) / train_std
    test_data = (test_data - train_mean) / train_std

    # pad to power of 2
    train_data = util.data.pad_data_pow_2(train_data, offset=0)
    test_data = util.data.pad_data_pow_2(test_data, offset=0)

    return (train_data, test_data), (train_labels, test_labels)

def prepare_kin40k_data(args):

    train_data = torch.from_numpy(pd.read_csv(os.path.join(args.kin40k_dir, 'kin40k_train_data.asc'), sep='\s+', header=None).values).float()
    train_labels = torch.from_numpy(pd.read_csv(os.path.join(args.kin40k_dir, 'kin40k_train_labels.asc'), sep='\s+', header=None).values).float()
    test_data = torch.from_numpy(pd.read_csv(os.path.join(args.kin40k_dir, 'kin40k_test_data.asc'), sep='\s+', header=None).values).float()
    test_labels = torch.from_numpy(pd.read_csv(os.path.join(args.kin40k_dir, 'kin40k_test_labels.asc'), sep='\s+', header=None).values).float()

    train_mean = train_data.mean(dim=0)
    train_std = train_data.std(dim=0)
    train_std[train_std==0] = 1.

    train_data = (train_data - train_mean) / train_std
    test_data = (test_data - train_mean) / train_std

    # pad to power of 2
    train_data = util.data.pad_data_pow_2(train_data, offset=0)
    test_data = util.data.pad_data_pow_2(test_data, offset=0)

    return (train_data, test_data), (train_labels, test_labels)

def cluster_points(data, num_clusters=10, method='random', global_max_dist=1.):

    shuffled_data = data[torch.randperm(len(data))]

    # determine cluster centers
    if method=='farthest':
        cluster_centers = [shuffled_data.mean(dim=0)]

        for _ in range(num_clusters-1):
            distances = torch.cdist(shuffled_data, torch.stack(cluster_centers, dim=0), p=2)

            # distances to the closest cluster centers
            min_dists = distances.min(dim=1)[0]

            if min_dists.max() <= global_max_dist:
                break

            farthest_point = min_dists.argmax()
            cluster_centers.append(shuffled_data[farthest_point])

        print('Number of clusters found: {}'.format(len(cluster_centers)))
        cluster_centers = torch.stack(cluster_centers, dim=0)
    elif method == 'random':
        cluster_centers = shuffled_data[:num_clusters]

    return cluster_centers


def compute_local_predictions(
        train_data, test_data, train_labels,
        cluster_assignments, cluster_centers,
        feature_encoder, noise_var):

    predictive_means = torch.zeros(
        len(test_data), 1,
        dtype=train_data.dtype, device=train_data.device
    )
    predictive_stds = torch.zeros_like(predictive_means)

    test_feature_time_ms = 0
        
    for cluster_id, cluster_center in tqdm(enumerate(cluster_centers)):
        if (cluster_assignments==cluster_id).sum()==0:
            # skip empty clusters
            continue

        train_features = feature_encoder.forward(train_data - cluster_center)

        torch.cuda.synchronize()
        start = timer()
        assigned_test_data = test_data[cluster_assignments==cluster_id]
        if assigned_test_data.dim() == 1:
            assigned_test_data = assigned_test_data.unsqueeze(dim=0)
        test_features = feature_encoder.forward(assigned_test_data - cluster_center)
        torch.cuda.synchronize()
        test_feature_time_ms += (timer() - start) * 1000

        het_gp = HeteroskedasticGP(None)

        f_test_mean, f_test_stds = het_gp.predictive_dist(
            train_features, test_features,
            train_labels, noise_var * torch.ones_like(train_labels)
        )

        predictive_means[cluster_assignments==cluster_id] = f_test_mean
        predictive_stds[cluster_assignments==cluster_id] = f_test_stds

    return predictive_means, predictive_stds, test_feature_time_ms


def run_gp(args, config, D, train_data, test_data, train_labels, lengthscale, var, noise_var, cluster_assignments, cluster_centers):

    feature_encoder = GaussianApproximator(
        train_data.shape[1], D,
        approx_degree=config['degree'],
        lengthscale=1.0, var=var,
        trainable_kernel=False,
        method=config['method'],
        projection_type=config['proj'],
        hierarchical=config['hierarchical'],
        complex_weights=config['complex_weights'],
        complex_real=config['complex_real']
    )

    feature_encoder.log_lengthscale.data = lengthscale.log()

    if args.use_gpu:
        feature_encoder.cuda()

    if config['method'] == 'maclaurin' or config['method'] == 'maclaurin_exp_h01':
        feature_encoder.initialize_sampling_distribution(
            train_data[:args.num_dist_est_samples] - train_data[:args.num_dist_est_samples].mean(dim=0)
        )

        if config['method'] == 'maclaurin':
            feature_dist = feature_encoder.feature_encoder.measure.distribution
        else:
            feature_dist = None
        # feature_encoder.feature_encoder.measure.distribution = np.array(D * [1])

        feature_encoder.resample()

        if args.use_gpu:
            feature_encoder.cuda()
            feature_encoder.feature_encoder.move_submodules_to_cuda()

        predictive_means, predictive_stds, test_feature_time_ms = compute_local_predictions(
            train_data, test_data, train_labels,
            cluster_assignments, cluster_centers,
            feature_encoder, noise_var
        )
    else:
        feature_encoder.resample()

        cluster_assignments = cluster_centers = feature_dist = None

        if args.use_gpu:
            feature_encoder.cuda()

        train_features = feature_encoder.forward(train_data)

        torch.cuda.synchronize()
        start = timer()
        test_features = feature_encoder.forward(test_data)
        torch.cuda.synchronize()
        test_feature_time_ms = (timer() - start) * 1000

        het_gp = HeteroskedasticGP(None)

        predictive_means, predictive_stds = het_gp.predictive_dist(
            train_features, test_features,
            train_labels, noise_var * torch.ones_like(train_labels)
        )

    return predictive_means, predictive_stds, feature_dist, test_feature_time_ms


def run_gp_eval(
        train_data, test_data, train_labels, test_labels, label_mean,
        log_lengthscale, log_var, log_noise_var, args, csv_handler, seed
    ):

    print(train_data.shape)

    # ground truth GP
    kernel_fun = lambda x, y, star: log_var.exp().item() * gaussian_kernel(x, y, lengthscale=log_lengthscale.exp()) if not star else log_var.exp()
    f_test_mean_ref, f_test_stds_ref = predictive_dist_exact(
        train_data, test_data, train_labels, log_noise_var.exp().item() * torch.ones_like(train_labels), kernel_fun
    )

    if args.cluster_train:
        lengthscale_multipliers = [2**i for i in range(-3, 5, 1)]
    else:
        lengthscale_multipliers = [-1]

    for lengthscale_multiplier in lengthscale_multipliers:
        # determine clusters
        if args.cluster_train:
            cluster_centers = cluster_points(
                train_data,
                num_clusters=args.num_clusters,
                method=args.cluster_method,
                global_max_dist=lengthscale_multiplier*log_lengthscale.exp().item()
            )
        else:
            cluster_centers = cluster_points(
                test_data,
                num_clusters=args.num_clusters,
                method=args.cluster_method,
                global_max_dist= lengthscale_multiplier*log_lengthscale.exp().item()
            )

        # assign clusters
        torch.cuda.synchronize()
        start = timer()
        distances = torch.cdist(test_data, cluster_centers, p=2)
        cluster_assignments = distances.argmin(dim=1)
        torch.cuda.synchronize()
        cluster_assignment_time_ms = (timer() - start) * 1000

        if args.dataset == 'uk_house_prices':
            Ds = [8, 16, 32, 64, 128, 256, 512, 1024]
        elif args.dataset == 'sarcos':
            Ds = [32, 64, 128, 256, 256, 512, 1024]
        elif args.dataset == 'kin40k':
            Ds = [32, 64, 128, 256, 256, 512, 1024]
        else:
            Ds = [32, 64, 128, 256, 256, 512, 1024]

        dummy_assignments = torch.zeros_like(cluster_assignments)
        dummy_centers = train_data.mean(dim=0).unsqueeze(0)

        for D in Ds:
        # D = args.num_rfs

            for config in configs:
                dummy_config = ('single_cluster' in config and config['single_cluster'])

                f_test_mean, f_test_stds, feature_dist, test_feature_time_ms = run_gp(
                    args, config, D,
                    train_data, test_data, train_labels,
                    log_lengthscale.exp(),
                    log_var.exp().item(),
                    log_noise_var.exp().item(),
                    dummy_assignments if dummy_config else cluster_assignments,
                    dummy_centers if dummy_config else cluster_centers
                )

                test_kl = kl_factorized_gaussian(
                    f_test_mean+label_mean,
                    f_test_mean_ref+label_mean,
                    (f_test_stds**2+log_noise_var.exp()).sqrt(),
                    (f_test_stds_ref**2+log_noise_var.exp()).sqrt()
                ).sum(dim=0).mean().item()
                
                test_mean_mse = (f_test_mean_ref - f_test_mean).pow(2).mean().item()
                test_var_mse = (f_test_stds_ref**2 - f_test_stds**2).pow(2).mean().item()

                test_mse, test_mnll = regression_scores(
                    f_test_mean+label_mean,
                    f_test_stds**2 + log_noise_var.exp().item(),
                    test_labels + label_mean
                )

                feature_dist = str(feature_dist)

                log_dir = {
                    'seed': seed,
                    'N': args.num_train_samples,
                    'D': D,
                    'num_clusters': len(cluster_centers) if (config['method']=='maclaurin' and not dummy_config) else None,
                    'lengthscale_mult': lengthscale_multiplier,
                    'lengthscale': log_lengthscale.exp().item(),
                    'kernel_var': log_var.exp().item(),
                    'noise_var': log_noise_var.exp().item(),
                    'method': config['method'],
                    'proj': config['proj'],
                    'ctr': config['complex_real'],
                    'feature_dist': feature_dist,
                    'mse': test_mse,
                    'rmse': np.sqrt(test_mse),
                    'smse': test_mse / test_labels.var().item(),
                    'test_kl': test_kl,
                    'test_mnll': test_mnll,
                    'test_mean_mse': test_mean_mse,
                    'test_var_mse': test_var_mse,
                    'cluster_time_ms': cluster_assignment_time_ms,
                    'test_feature_time_ms': test_feature_time_ms
                }

                csv_handler.append(log_dir)
                csv_handler.save()


if __name__ == '__main__':
    args = parse_args()

    csv_handler = DF_Handler(args.dataset, 'cluster_selection', csv_dir=args.csv_dir)

    if args.dataset == 'uk_house_prices':
        data, labels = prepare_house_price_data(args)
    elif args.dataset == 'sarcos':
        (train_data_or, test_data), (train_labels_or, test_labels) = prepare_sarcos_data(args)
    elif args.dataset == 'kin40k':
        (train_data, test_data), (train_labels, test_labels) = prepare_kin40k_data(args)
    elif args.dataset == 'uci':
        data, labels = prepare_uci_data(args)


    for seed in range(args.num_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)

        if args.dataset == 'uk_house_prices' or args.dataset == 'uci':
            # shuffle data
            permutation = torch.randperm(len(data))
            data = data[permutation]
            labels = labels[permutation]

            cutoff = min(int(0.9 * len(data)), args.num_train_samples)
            
            train_data = data[:cutoff]
            train_labels = labels[:cutoff]
            test_data = data[cutoff:2*cutoff]
            test_labels = labels[cutoff:2*cutoff]
        elif args.dataset == 'sarcos':
            permutation = torch.randperm(len(train_data_or))
            train_data = train_data_or[permutation]
            train_labels = train_labels_or[permutation]

            train_data = train_data[:args.num_train_samples]
            train_labels = train_labels[:args.num_train_samples]


        label_mean = train_labels.mean()
        train_labels -= label_mean
        test_labels -= label_mean

        if args.use_gpu:
            train_data = train_data.cuda()
            train_labels = train_labels.cuda()
            test_data = test_data.cuda()
            test_labels = test_labels.cuda()
            label_mean = label_mean.cuda()

        noise_var = 1.0
        log_noise_var = torch.nn.Parameter((torch.ones(1, device=('cuda' if args.use_gpu else 'cpu')) * noise_var).log(), requires_grad=True)
        log_lengthscale = torch.nn.Parameter(torch.cdist(train_data[:args.num_train_samples], train_data[:args.num_train_samples]).median().log(), requires_grad=True)
        log_var = torch.nn.Parameter(torch.ones(1, device=('cuda' if args.use_gpu else 'cpu')) * train_labels.var().log(), requires_grad=True)

        kernel_fun = lambda x, y: log_var.exp() * gaussian_kernel(x, y, lengthscale=log_lengthscale.exp())

        optimize_marginal_likelihood(
            train_data[:args.num_lml_samples],
            train_labels[:args.num_lml_samples],
            kernel_fun, log_lengthscale,
            log_var,
            log_noise_var,
            num_iterations=args.lml_iterations,
            lr=args.lml_lr
        )

        # log_lengthscale = (log_lengthscale.exp() / 1.).log()
        
        print('Lengthscale:', log_lengthscale.exp().item())
        print('Kernel var:', log_var.exp().item())
        print('Noise var:', log_noise_var.exp().item())

        ### Run comparisons across feature dimensions and seeds ###
        run_gp_eval(
            train_data, test_data, train_labels, test_labels, label_mean,
            log_lengthscale, log_var, log_noise_var, args, csv_handler, seed
        )
