import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import torch
import pandas as pd
import os
# os.environ["PROJ_LIB"] = os.path.join(os.environ["CONDA_PREFIX"], "share", "proj")
from mpl_toolkits.basemap import Basemap

from tqdm import tqdm
import argparse

import sys 
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from util.data import DF_Handler
from util.kernels import gaussian_kernel
from util.helper_functions import optimize_marginal_likelihood, kl_factorized_gaussian, regression_scores
from models.het_gp import HeteroskedasticGP, predictive_dist_exact
from random_features.gaussian_approximator import GaussianApproximator


configs = [
    {'method': 'rff', 'proj': 'gaussian', 'degree': 4, 'hierarchical': False, 'complex_weights': True},
    # {'method': 'rff', 'proj': 'gaussian', 'degree': 4, 'bias': 0, 'lengthscale': True, 'hierarchical': False, 'complex_weights': True},
    {'method': 'maclaurin', 'proj': 'rademacher', 'degree': 15, 'hierarchical': False, 'complex_weights': True},
    # {'method': 'maclaurin', 'proj': 'rademacher', 'degree': 10, 'bias': 0, 'lengthscale': True, 'hierarchical': False, 'complex_weights': True}
]


# Base map coordinates for the uk
llcrnrlon=-6
llcrnrlat=50
urcrnrlon=2
urcrnrlat=55


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--house_price_file', type=str, required=False,
                        default='../../datasets/export/uk_house_prices/pp-monthly-update-new-version.csv',
                        help='CSV File with the UK house prices')
    parser.add_argument('--post_code_file', type=str, required=False,
                        default='../../datasets/export/uk_house_prices/ukpostcodes.csv',
                        help='CSV File with the UK house prices')
    parser.add_argument('--num_train_samples', type=int, required=False, default=10000,
                        help='Number of data samples for training')
    parser.add_argument('--num_grid_samples', type=int, required=False, default=100,
                        help='Number of elements for each grid dimension')
    parser.add_argument('--num_lml_samples', type=int, required=False, default=5000,
                        help='Number of data samples for likelihood optimization')
    parser.add_argument('--lml_lr', type=float, required=False, default=1e-2,
                        help='Learning rate for likelihood optimization')
    parser.add_argument('--lml_iterations', type=int, required=False, default=10,
                        help='Number of iterations for likelihood optimization')
    parser.add_argument('--num_dist_est_samples', type=int, required=False, default=500,
                        help='Number of datapoints used to estimate maclaurin distribution')
    parser.add_argument('--num_rfs', type=int, required=False, default=100,
                        help='Number of random features')
    parser.add_argument('--num_seeds', type=int, required=False, default=5,
                        help='Number of seeds (runs)')
    parser.add_argument('--plot_map', dest='plot_map', action='store_true')
    parser.set_defaults(plot_map=False)
    parser.add_argument('--use_gpu', dest='use_gpu', action='store_true')
    parser.set_defaults(use_gpu=False)

    args = parser.parse_args()

    return args

def run_gp(args, config, D, train_data, test_data, train_labels, lengthscale, var, noise_var):
    feature_encoder = GaussianApproximator(
        2, D,
        approx_degree=10,
        lengthscale=lengthscale, var=var,
        trainable_kernel=False,
        method=config['method'],
        projection_type=config['proj'],
        hierarchical=config['hierarchical'],
        complex_weights=config['complex_weights']
    )

    if args.use_gpu:
        feature_encoder.cuda()

    if config['method'] == 'maclaurin':
        feature_encoder.initialize_sampling_distribution(
            train_data[:args.num_dist_est_samples] - train_data[:args.num_dist_est_samples].mean(dim=0)
        )
        # feature_encoder.feature_encoder.measure = Exponential_Measure(True)
        # feature_encoder.feature_encoder.measure.distribution = np.array(D * [1])
        feature_dist = feature_encoder.feature_encoder.measure.distribution

        feature_encoder.resample()

        if args.use_gpu:
            feature_encoder.cuda()
            feature_encoder.feature_encoder.move_submodules_to_cuda()

        # solve one GP per test input
        test_means = []
        test_stds = []
        for test_point in tqdm(test_data):
            train_features = feature_encoder.forward(train_data - test_point) #  
            test_features = feature_encoder.forward(torch.zeros_like(test_point).unsqueeze(0)) # 

            het_gp = HeteroskedasticGP(None)

            f_test_mean, f_test_stds = het_gp.predictive_dist(
                train_features, test_features,
                train_labels, noise_var * torch.ones_like(train_labels)
            )

            test_means.append(f_test_mean)
            test_stds.append(f_test_stds)

        f_test_mean = torch.cat(test_means, dim=0)
        f_test_stds = torch.cat(test_stds, dim=0)

    else:
        feature_encoder.resample()

        feature_dist = None

        if args.use_gpu:
            feature_encoder.cuda()

        train_features = feature_encoder.forward(train_data)
        test_features = feature_encoder.forward(test_data)

        het_gp = HeteroskedasticGP(None)

        f_test_mean, f_test_stds = het_gp.predictive_dist(
            train_features, test_features,
            train_labels, noise_var * torch.ones_like(train_labels)
        )

    return f_test_mean, f_test_stds, feature_dist


if __name__ == '__main__':
    args = parse_args()

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

    # shuffle data
    permutation = torch.randperm(len(data))
    data = data[permutation]
    labels = labels[permutation]

    train_data = data[:args.num_train_samples]
    train_labels = labels[:args.num_train_samples]
    label_mean = train_labels.mean()
    train_labels -= label_mean
    test_data = data[args.num_train_samples:]
    test_labels = labels[args.num_train_samples:]
    test_labels -= label_mean

    if args.use_gpu:
        train_data = train_data.cuda()
        train_labels = train_labels.cuda()
        test_data = test_data.cuda()
        test_labels = test_labels.cuda()
        label_mean = label_mean.cuda()

    noise_var = 0.8
    log_noise_var = torch.nn.Parameter((torch.ones(1, device=('cuda' if args.use_gpu else 'cpu')) * noise_var).log(), requires_grad=False)
    log_lengthscale = torch.nn.Parameter(torch.cdist(train_data, train_data).median().log(), requires_grad=True)
    log_var = torch.nn.Parameter((torch.ones(1, device=('cuda' if args.use_gpu else 'cpu')) * train_labels.var()), requires_grad=True)

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

    print('Lengthscale:', log_lengthscale.exp().item())
    print('Kernel var:', log_var.exp().item())
    print('Noise var:', log_noise_var.exp().item())

    ### Run comparisons across feature dimensions and seeds ###

    # ground truth GP
    kernel_fun = lambda x, y: log_var.exp().item() * gaussian_kernel(x, y, lengthscale=log_lengthscale.exp().item())
    f_test_mean_ref, f_test_stds_ref = predictive_dist_exact(
        train_data, test_data, train_labels, log_noise_var.exp().item() * torch.ones_like(train_labels), kernel_fun
    )

    csv_handler = DF_Handler(
        'uk_house_prices',
        'ntrain{}_nll{}'.format(args.num_train_samples, args.num_lml_samples),
        csv_dir='../csv'
    )

    for seed in range(args.num_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)

        for D in [10, 50, 100, 200, 300]:
            for config in configs:
                f_test_mean, f_test_stds, feature_dist = run_gp(
                    args, config, D,
                    train_data, test_data, train_labels,
                    log_lengthscale.exp().item(),
                    log_var.exp().item(),
                    log_noise_var.exp().item()
                )

                test_kl = kl_factorized_gaussian(
                    f_test_mean+label_mean, f_test_mean_ref+label_mean,
                    f_test_stds, f_test_stds_ref
                )
                # 18 nan values in reference std
                test_kl = test_kl[~test_kl.isnan()].sum(dim=0).mean().item()
                
                test_mean_mse = (f_test_mean_ref - f_test_mean).pow(2).mean().item()
                test_var_mse = (f_test_stds_ref**2 - f_test_stds**2)
                test_var_mse = test_var_mse[~test_var_mse.isnan()].pow(2).mean().item()

                test_rmse, test_mnll = regression_scores(
                    f_test_mean+label_mean,
                    f_test_stds**2 + log_noise_var.exp().item(),
                    test_labels+label_mean
                )

                feature_dist = str(feature_dist)

                log_dir = {
                    'seed': seed,
                    'D': D,
                    'method': config['method'],
                    'proj': config['proj'],
                    'complex_weights': config['complex_weights'],
                    'feature_dist': feature_dist,
                    'rmse': test_rmse,
                    'test_kl': test_kl,
                    'test_mnll': test_mnll,
                    'test_mean_mse': test_mean_mse,
                    'test_var_mse': test_var_mse
                }

                csv_handler.append(log_dir)
                csv_handler.save()


    ### plot map ###
    if not args.plot_map:
        exit()

    a = torch.linspace(-6, 2.0, args.num_grid_samples, device=('cuda' if args.use_gpu else 'cpu'))
    b = torch.linspace(50, 55, args.num_grid_samples, device=('cuda' if args.use_gpu else 'cpu'))
    grid_data = torch.meshgrid(a, b)
    grid_inputs = torch.stack(grid_data, dim=-1).view(-1,2)

    kernel_fun = lambda x, y: log_var.exp().item() * gaussian_kernel(x, y, lengthscale=log_lengthscale.exp().item())
    f_test_mean_ref, f_test_stds_ref = predictive_dist_exact(
        train_data, grid_inputs, train_labels, log_noise_var.exp().item() * torch.ones_like(train_labels), kernel_fun
    )

    label_ref = f_test_mean_ref+label_mean
    label_range = (label_ref.max() - label_ref.min()).item()
    level_space = np.linspace(label_ref.min().item() - 0.2 * label_range, label_ref.max().item() + 0.2 * label_range, 20)

    # Plot ground truth GP
    num_plots = 1+len(configs)
    fig, axes = plt.subplots(
        ncols=num_plots,
        nrows=1,
        figsize=(
            # width (we add a small offset for the color bar)
            (urcrnrlat - llcrnrlat) * (num_plots+0.1),
            # height
            (urcrnrlon - llcrnrlon) * num_plots
        )
    )

    axes[0].set_title('Full GP')

    m = Basemap(projection='merc',
        resolution = 'i', ax=axes[0],
        llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat,
        urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat
    )

    # m.scatter(result['lon'].values, result['lat'].values, latlon=True,
    #           c=np.log(result['price'].values), s=1,
    #           cmap='jet', alpha=1)

    m.drawcoastlines(linewidth = 1.0, color='black')
    contour_obj = m.contour(
        grid_data[0].cpu().numpy(),
        grid_data[1].cpu().numpy(),
        (f_test_mean_ref.reshape(len(grid_data[0]), len(grid_data[0]))+label_mean).cpu().numpy(),
        cmap='jet',
        latlon=True,
        levels=level_space
    )

    for j, config in enumerate(configs):
        
        # run gp
        f_test_mean, f_test_stds = run_gp(
            args, config, args.num_rfs,
            train_data, test_data, train_labels,
            log_lengthscale.exp().item(),
            log_var.exp().item(),
            log_noise_var.exp().item()
        )

        axes[j+1].set_title('{}{} {}'.format(
            'Complex ' if config['complex_weights'] else '',
            config['method'],
            config['proj']
        ))
        m = Basemap(projection='merc',
            resolution = 'i', ax=axes[j+1],
            llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat,
            urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat
        )

        # m.scatter(result['long'].values, result['lat'].values, latlon=True,
        #           c=np.log(result['price'].values), s=1,
        #           cmap='jet', alpha=1)

        m.drawcoastlines(linewidth = 1.0, color='black')
        m.contour(
            grid_data[0].cpu().numpy(),
            grid_data[1].cpu().numpy(),
            (f_test_mean.reshape(len(grid_data[0]),len(grid_data[0]))+label_mean).cpu().numpy(),
            cmap='jet',
            latlon=True,
            levels=level_space
        )

    colorbar = fig.colorbar(contour_obj, ax=axes[:], location='bottom', shrink=0.5, pad=0.01)
    colorbar.set_label('log(Sales Price)')

    # plt.tight_layout()
    plt.savefig('../figures/house_prices.pdf', dpi=300, bbox_inches='tight')