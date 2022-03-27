import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import torch
import pandas as pd
import os
# os.environ["PROJ_LIB"] = os.path.join(os.environ["CONDA_PREFIX"], "share", "proj")
from mpl_toolkits.basemap import Basemap

from tqdm import tqdm
from timeit import default_timer as timer
import argparse

import sys 
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from util.data import DF_Handler
from util.kernels import gaussian_kernel
from util.helper_functions import optimize_marginal_likelihood, kl_factorized_gaussian, regression_scores
from util.measures import Exponential_Measure
from models.het_gp import HeteroskedasticGP, predictive_dist_exact
from random_features.gaussian_approximator import GaussianApproximator


configs = [
    {'method': 'rff', 'proj': 'gaussian', 'degree': 4, 'hierarchical': False, 'complex_weights': False, 'complex_real': False},
    {'method': 'rff', 'proj': 'srht', 'degree': 4, 'hierarchical': False, 'complex_weights': False, 'complex_real': False},
    # {'method': 'rff', 'proj': 'gaussian', 'degree': 4, 'bias': 0, 'lengthscale': True, 'hierarchical': False, 'complex_weights': True},
    # {'method': 'maclaurin_exp_h01', 'proj': 'rademacher', 'degree': 15, 'hierarchical': False, 'complex_weights': False, 'complex_real': False},
    {'method': 'maclaurin', 'proj': 'rademacher', 'degree': 15, 'hierarchical': False, 'complex_weights': False, 'complex_real': False},
    {'method': 'maclaurin', 'proj': 'rademacher', 'degree': 15, 'hierarchical': False, 'complex_weights': False, 'complex_real': True},
    # {'method': 'maclaurin', 'proj': 'rademacher', 'degree': 15, 'hierarchical': False, 'complex_weights': False, 'complex_real': True},
    # {'method': 'maclaurin_exp_h01', 'proj': 'srht', 'degree': 15, 'hierarchical': False, 'complex_weights': False, 'complex_real': False},
    {'method': 'maclaurin', 'proj': 'srht', 'degree': 15, 'hierarchical': False, 'complex_weights': False, 'complex_real': False},
    {'method': 'maclaurin', 'proj': 'srht', 'degree': 15, 'hierarchical': False, 'complex_weights': False, 'complex_real': True}
    # {'method': 'maclaurin', 'proj': 'srht', 'degree': 15, 'hierarchical': False, 'complex_weights': False, 'complex_real': True}
]


# Base map coordinates for the uk
llcrnrlon=-6
llcrnrlat=50
urcrnrlon=2
urcrnrlat=55


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--house_price_file', type=str, required=False,
                        default='../datasets/export/uk_house_prices/pp-monthly-update-new-version.csv',
                        help='CSV File with the UK house prices')
    parser.add_argument('--post_code_file', type=str, required=False,
                        default='../datasets/export/uk_house_prices/ukpostcodes.csv',
                        help='CSV File with the UK house prices')
    parser.add_argument('--csv_dir', type=str, required=False,
                        default='csv', help='Directory to save CSV files to')
    parser.add_argument('--csv_filename', type=str, required=False,
                        default='cluster_selection', help='The name of the CSV file to be saved')
    parser.add_argument('--figure_dir', type=str, required=False,
                        default='figures', help='Directory to save CSV files to')
    parser.add_argument('--num_seeds', type=int, required=False, default=10,
                        help='Number of seeds (runs)')
    parser.add_argument('--num_train_samples', type=int, required=False, default=10000,
                        help='Number of data samples for training')
    parser.add_argument('--num_grid_samples', type=int, required=False, default=100,
                        help='Number of elements for each grid dimension')
    parser.add_argument('--num_lml_samples', type=int, required=False, default=5000,
                        help='Number of data samples for likelihood optimization')
    parser.add_argument('--lml_lr', type=float, required=False, default=1e-1,
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
    parser.add_argument('--run_gp_eval', dest='run_gp_eval', action='store_true')
    parser.set_defaults(run_gp_eval=True)
    parser.add_argument('--plot_map', dest='plot_map', action='store_true')
    parser.set_defaults(plot_map=False)
    parser.add_argument('--use_gpu', dest='use_gpu', action='store_true')
    parser.set_defaults(use_gpu=False)

    args = parser.parse_args()

    return args

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
        test_features = feature_encoder.forward(test_data[cluster_assignments==cluster_id] - cluster_center)
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
        2, D,
        approx_degree=config['degree'],
        lengthscale=lengthscale, var=var,
        trainable_kernel=False,
        method=config['method'],
        projection_type=config['proj'],
        hierarchical=config['hierarchical'],
        complex_weights=config['complex_weights'],
        complex_real=config['complex_real']
    )

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

    # ground truth GP
    while True:
        try:
            kernel_fun = lambda x, y: log_var.exp().item() * gaussian_kernel(x, y, lengthscale=log_lengthscale.exp().item())
            f_test_mean_ref, f_test_stds_ref = predictive_dist_exact(
                train_data, test_data, train_labels, log_noise_var.exp().item() * torch.ones_like(train_labels), kernel_fun
            )
            break
        except RuntimeError:
            log_noise_var.data = (log_noise_var.exp()*1.1).log()
            print('Inversion error. New noise var:', log_noise_var.exp().item())
            continue


    for lengthscale_multiplier in [2**i for i in range(-3, 5, 1)]:
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
                global_max_dist=lengthscale_multiplier*log_lengthscale.exp().item()
            )

        # assign clusters
        torch.cuda.synchronize()
        start = timer()
        distances = torch.cdist(test_data, cluster_centers, p=2)
        cluster_assignments = distances.argmin(dim=1)
        torch.cuda.synchronize()
        cluster_assignment_time_ms = (timer() - start) * 1000

        for D in [8, 16, 32, 64, 128, 256, 512, 1024]:
        # D = args.num_rfs

            for config in configs:
                f_test_mean, f_test_stds, feature_dist, test_feature_time_ms = run_gp(
                    args, config, D,
                    train_data, test_data, train_labels,
                    log_lengthscale.exp().item(),
                    log_var.exp().item(),
                    log_noise_var.exp().item(),
                    cluster_assignments,
                    cluster_centers
                )

                test_kl = kl_factorized_gaussian(
                    f_test_mean+label_mean,
                    f_test_mean_ref+label_mean,
                    (f_test_stds**2+log_noise_var.exp()).sqrt(),
                    (f_test_stds_ref**2+log_noise_var.exp()).sqrt()
                ).sum(dim=0).mean().item()
                
                test_mean_mse = (f_test_mean_ref - f_test_mean).pow(2).mean().item()
                test_var_mse = (f_test_stds_ref**2 - f_test_stds**2).pow(2).mean().item()

                test_rmse, test_mnll = regression_scores(
                    f_test_mean+label_mean,
                    f_test_stds**2 + log_noise_var.exp().item(),
                    test_labels + label_mean
                )

                feature_dist = str(feature_dist)

                log_dir = {
                    'seed': seed,
                    'N': args.num_train_samples,
                    'D': D,
                    'num_clusters': len(cluster_centers) if cluster_centers is not None else None,
                    'lengthscale_mult': lengthscale_multiplier,
                    'lengthscale': log_lengthscale.exp().item(),
                    'kernel_var': log_var.exp().item(),
                    'noise_var': log_noise_var.exp().item(),
                    'method': config['method'],
                    'proj': config['proj'],
                    'ctr': config['complex_real'],
                    'feature_dist': feature_dist,
                    'rmse': test_rmse,
                    'test_kl': test_kl,
                    'test_mnll': test_mnll,
                    'test_mean_mse': test_mean_mse,
                    'test_var_mse': test_var_mse,
                    'cluster_time_ms': cluster_assignment_time_ms,
                    'test_feature_time_ms': test_feature_time_ms
                }

                csv_handler.append(log_dir)
                csv_handler.save()

def plot_gp_map(
        train_data, train_labels, test_labels, label_mean,
        log_lengthscale, log_var, log_noise_var, args
    ):
    a = torch.linspace(llcrnrlon, urcrnrlon, args.num_grid_samples, device=('cuda' if args.use_gpu else 'cpu'))
    b = torch.linspace(llcrnrlat, urcrnrlat, args.num_grid_samples, device=('cuda' if args.use_gpu else 'cpu'))
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

    # determine clusters
    if args.cluster_train:
        cluster_centers = cluster_points(
            train_data,
            num_clusters=args.num_clusters,
            method=args.cluster_method,
            global_max_dist=1.0
        )
    else:
        cluster_centers = cluster_points(
            grid_data,
            num_clusters=args.num_clusters,
            method=args.cluster_method,
            global_max_dist=1.0
        )

    # assign clusters
    distances = torch.cdist(test_data, cluster_centers, p=2)
    cluster_assignments = distances.argmin(dim=1)

    for j, config in enumerate(configs):
        
        # run gp
        f_test_mean, f_test_stds, feature_dist = run_gp(
                args, config, args.num_rfs,
                train_data, test_data, train_labels,
                log_lengthscale.exp().item(),
                log_var.exp().item(),
                log_noise_var.exp().item(),
                cluster_assignments,
                cluster_centers
        )

        test_kl = kl_factorized_gaussian(
            f_test_mean+label_mean,
            f_test_mean_ref+label_mean,
            (f_test_stds**2+log_noise_var.exp()).sqrt(),
            (f_test_stds_ref**2+log_noise_var.exp()).sqrt()
        ).sum(dim=0).mean().item()
        
        test_mean_mse = (f_test_mean_ref - f_test_mean).pow(2).mean().item()
        test_var_mse = (f_test_stds_ref**2 - f_test_stds**2).pow(2).mean().item()

        # test_rmse, test_mnll = regression_scores(
        #     f_test_mean+label_mean,
        #     f_test_stds**2 + log_noise_var.exp().item(),
        #     test_labels + label_mean
        # )

        axes[j+1].set_title('{}{} {} - KL: {:.2f}'.format(
            'Complex ' if config['complex_weights'] else '',
            config['method'],
            config['proj'],
            test_kl
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

        if cluster_centers is not None:
            m.scatter(
                cluster_centers[:,0].cpu().numpy(),
                cluster_centers[:,1].cpu().numpy(),
                alpha=0.1, c='black', # np.arange(len(cluster_centers))
                latlon=True, s=2,
                cmap='jet'
            )

            # m.scatter(
            #     grid_data[0].cpu().numpy(),
            #     grid_data[1].cpu().numpy(),
            #     alpha=1, c=cluster_assignments.reshape(len(grid_data[0]),len(grid_data[0])).cpu().numpy(),
            #     latlon=True, s=2,
            #     cmap='jet'
            # )

    colorbar = fig.colorbar(contour_obj, ax=axes[:], location='bottom', shrink=0.5, pad=0.01)
    colorbar.set_label('log(Sales Price)')

    # plt.tight_layout()
    plt.savefig(os.path.join(args.figure_dir, 'house_prices.pdf'), dpi=300, bbox_inches='tight')


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


    csv_handler = DF_Handler('uk_house_prices', args.csv_filename, csv_dir=args.csv_dir)

    seeds = args.num_seeds if args.run_gp_eval else 1

    for seed in range(seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)

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

        noise_var = 1.0
        log_noise_var = torch.nn.Parameter((torch.ones(1, device=('cuda' if args.use_gpu else 'cpu')) * noise_var).log(), requires_grad=True)
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
        if args.run_gp_eval:
            run_gp_eval(
                train_data, test_data, train_labels, test_labels, label_mean,
                log_lengthscale, log_var, log_noise_var, args, csv_handler, seed
            )

        ### plot map ###
        if args.plot_map:
            plot_gp_map(
                train_data, train_labels, test_labels, label_mean,
                log_lengthscale, log_var, log_noise_var, args
            )

