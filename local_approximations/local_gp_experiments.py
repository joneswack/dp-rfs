import argparse
import time
import json

import torch
import numpy as np
import pandas as pd

from models.het_gp import HeteroskedasticGP, predictive_dist_exact
from random_features.gaussian_approximator import GaussianApproximator
from random_features.spherical import Spherical
from random_features.polynomial_sketch import PolynomialSketch
from random_features.maclaurin import Maclaurin
from random_features.rff import RFF
import util.data

from util.helper_functions import kl_factorized_gaussian
from util.helper_functions import classification_scores, regression_scores
from util.kernels import gaussian_kernel, polynomial_kernel
from util.measures import Fixed_Measure, Polynomial_Measure, P_Measure, Exponential_Measure

from util.helper_functions import cholesky_solve

from util.LBFGS import FullBatchLBFGS

"""
Runs Gaussian Process Classification experiments as closed form GP regression on transformed labels.
"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rf_parameter_file', type=str, required=False, default='config/rf_parameters/gaussian2.json',
                        help='Path to RF parameter file')
    parser.add_argument('--datasets_file', type=str, required=False, default='config/active_datasets3.json',
                        help='List of datasets to be used for the experiments')
    parser.add_argument('--num_data_samples', type=int, required=False, default=5000,
                        help='Number of data samples for lengthscale estimation')
    parser.add_argument('--num_mc_samples', type=int, required=False, default=1000,
                        help='Number of mc samples for predictive distribution')
    parser.add_argument('--num_seeds', type=int, required=False, default=10,
                        help='Number of seeds (runs)')
    parser.add_argument('--use_gpu', dest='use_gpu', action='store_true')
    parser.set_defaults(use_gpu=False)

    args = parser.parse_args()

    return args


def exact_marginal_log_likelihood(kernel_train, training_labels, log_noise_var):
    n = len(training_labels)
    L_train = torch.cholesky(kernel_train + torch.exp(log_noise_var) * torch.eye(n, dtype=torch.float), upper=False)
    alpha = cholesky_solve(training_labels, L_train)
    mll = -0.5 * training_labels.t().mm(alpha) - L_train.diagonal().log().sum() - (n / 2) * np.log(2*np.pi)

    return mll

def optimize_marginal_likelihood(training_data, training_labels, kernel_fun, log_lengthscale, log_var, log_noise_var, num_iterations=10, lr=1e-3):
    trainable_params = [log_lengthscale, log_var, log_noise_var]

    for iteration in range(num_iterations):
        print('### Iteration {} ###'.format(iteration))
        optimizer = FullBatchLBFGS(trainable_params, lr=lr, history_size=10, line_search='Wolfe')

        def closure():
            optimizer.zero_grad()

            kernel_train = kernel_fun(training_data, training_data)
            loss = - exact_marginal_log_likelihood(kernel_train, training_labels, log_noise_var)
            print('Loss: {}'.format(loss.item()))

            return loss

        loss = closure()
        loss.backward()
        options = {'closure': closure, 'current_loss': loss, 'max_ls': 10}
        loss, _, lr, _, F_eval, G_eval, _, _ = optimizer.step(options)


def prepare_data(config, args, rf_parameters, data_name, current_train, current_test, train_labels, test_labels,
                    noise_var, regression=False, fit_ref_gp=False):
    
    train_idxs = torch.randperm(len(current_train))[:args.num_data_samples]
    test_idxs = torch.randperm(len(current_test))[:args.num_data_samples]

    if regression:
        vv = torch.ones_like(train_labels) * noise_var
        mm = train_labels
        ymean = train_labels.mean(0)
    else:
        # we convert the training labels according to Milios et al., 2018
        vv = torch.log(1.0 + 1.0 / (train_labels + noise_var))
        mm = torch.log(train_labels + noise_var) - vv / 2.0
        ymean = train_labels.mean(0).log() + torch.mean(mm-train_labels.mean(0).log())

    mm = mm - ymean
    kernel_var = mm.var().item()

    current_train, current_test = util.data.standardize_data(current_train, current_test)
    lengthscale = torch.cdist(current_train[train_idxs], current_train[train_idxs]).median().item()

    if fit_ref_gp and regression:
        # optionally fit reference GP
        log_noise_var = torch.nn.Parameter((torch.ones(1) * noise_var).log(), requires_grad=False)
        log_lengthscale = torch.nn.Parameter((torch.ones(1) * lengthscale).log(), requires_grad=True)
        log_var = torch.nn.Parameter((torch.ones(1) * kernel_var).log(), requires_grad=True)

        # kernel_fun = lambda x, y: rbf_kernel(x, y, lengthscale=1.)
        kernel_fun = lambda x, y: log_var.exp() * gaussian_kernel(x, y, lengthscale=log_lengthscale.exp())

        print('Lengthscale:', log_lengthscale.exp().item())
        print('Kernel var:', log_var.exp().item())
        print('Noise var:', log_noise_var.exp().item())

        optimize_marginal_likelihood(
            current_train, mm, kernel_fun, log_lengthscale, log_var, log_noise_var, num_iterations=100, lr=1e-2
        )

        kernel_var = log_var.exp().item()
        lengthscale = log_lengthscale.exp().item() / 4.
        noise_var = log_noise_var.exp().item()

        print('Lengthscale:', lengthscale)
        print('Kernel var:', kernel_var)
        print('Noise var:', noise_var)



    kernel_fun = lambda x, y: kernel_var * gaussian_kernel(
        x, y, lengthscale=lengthscale)

    # float conversion
    current_train = current_train.float()
    current_test = current_test.float()
    train_labels = train_labels.float()
    test_labels = test_labels.float()
    mm = mm.float()
    vv = vv.float()
    ref_kernel = kernel_fun(current_test[test_idxs], current_test[test_idxs])

    f_test_mean_ref, f_test_stds_ref = predictive_dist_exact(
        current_train[train_idxs],
        current_test[test_idxs],
        mm[train_idxs], vv[train_idxs], kernel_fun
    )

    meta_data_dict = {
        'data_name': data_name,
        'train_data': current_train,
        'test_data': current_test,
        'train_labels': mm,
        'train_label_mean': ymean,
        'train_label_vars': vv,
        'test_labels': test_labels,
        'lengthscale': lengthscale,
        'kernel_var': kernel_var,
        'noise_var': noise_var,
        'ref_kernel': ref_kernel,
        'f_test_mean_ref': f_test_mean_ref,
        'f_test_stds_ref': f_test_stds_ref,
        'train_idxs': train_idxs,
        'test_idxs': test_idxs,
        'regression': regression
    }

    return meta_data_dict

def evaluate_test_points(feature_encoder, het_gp, train_data_padded, test_data_padded, train_labels, train_label_vars):

    if isinstance(feature_encoder.feature_encoder, Maclaurin):
        test_means = []
        test_stds = []

        for test_point_padded in test_data_padded:
            train_features = feature_encoder.forward(train_data_padded - test_point_padded) #   
            test_features = feature_encoder.forward(torch.zeros_like(test_point_padded).unsqueeze(0)) # torch.zeros_like(test_point_padded).unsqueeze(0)

            ### run subsampled GP for KL divergence
            f_test_mean, f_test_stds = het_gp.predictive_dist(
                train_features, test_features,
                train_labels, train_label_vars
            )

            test_means.append(f_test_mean)
            test_stds.append(f_test_stds)

        return torch.cat(test_means, dim=0), torch.cat(test_stds, dim=0)

    else:
        train_features = feature_encoder.forward(train_data_padded)
        test_features = feature_encoder.forward(test_data_padded)

        ### run subsampled GP for KL divergence
        f_test_mean, f_test_stds = het_gp.predictive_dist(
            train_features, test_features,
            train_labels, train_label_vars
        )

        return f_test_mean, f_test_stds


def run_rf_gp(data_dict, d_features, config, args, rf_params, seed):

    if (rf_params['kernel'] == 'polynomial' and config['method'] == 'poly_sketch' and config['bias'] != 0) \
        or (rf_params['kernel'] == 'gaussian' and config['method'] == 'poly_sketch'):
        offset = 1
    else:
        offset = 0

    comp_real = config['complex_real'] if 'complex_real' in config.keys() else False
    full_cov = config['full_cov'] if 'full_cov' in config.keys() else False
    
    train_data_padded = util.data.pad_data_pow_2(data_dict['train_data'], offset=offset)
    test_data_padded = util.data.pad_data_pow_2(data_dict['test_data'], offset=offset)
    train_idxs = data_dict['train_idxs']
    test_idxs = data_dict['test_idxs']
    train_labels = data_dict['train_labels']
    train_label_mean = data_dict['train_label_mean']
    train_label_vars = data_dict['train_label_vars']
    test_labels = data_dict['test_labels']
    noise_var = data_dict['noise_var']
    ref_kernel = data_dict['ref_kernel']
    f_test_mean_ref = data_dict['f_test_mean_ref']
    f_test_stds_ref = data_dict['f_test_stds_ref']
    regression = data_dict['regression']

    feature_encoder = GaussianApproximator(
        train_data_padded.shape[1], d_features,
        approx_degree=rf_params['max_sampling_degree'], lengthscale=data_dict['lengthscale'],
        var=data_dict['kernel_var'], trainable_kernel=False, method=config['method'],
        projection_type=config['proj'], hierarchical=config['hierarchical'],
        complex_weights=config['complex_weights'], device=('cuda' if args.use_gpu else 'cpu')
    )
    feature_encoder.initialize_sampling_distribution(train_data_padded[train_idxs],
        min_sampling_degree=rf_params['min_sampling_degree'])
    # feature_encoder.feature_encoder.measure = Exponential_Measure(True)

    feature_encoder.resample()

    het_gp = HeteroskedasticGP(None)

    # evaluate each test point
    f_test_mean, f_test_stds = evaluate_test_points(
        feature_encoder, het_gp,
        train_data_padded, test_data_padded,
        train_labels, train_label_vars
    )

    test_kl = kl_factorized_gaussian(
        f_test_mean[test_idxs], f_test_mean_ref,
        f_test_stds[test_idxs], f_test_stds_ref
    ).sum(dim=0).mean()
    test_mean_mse = (f_test_mean_ref - f_test_mean[test_idxs]).pow(2).mean()
    test_var_mse = (f_test_stds_ref**2 - f_test_stds[test_idxs]**2).pow(2).mean()

    ### gp prediction
    # torch.cuda.synchronize()
    start = time.time()

    if regression:
        f_test_mean += train_label_mean
        # test rmse
        test_error, test_mnll = regression_scores(f_test_mean, f_test_stds**2 + noise_var, test_labels)
    else:
        epsilon = torch.randn(args.num_mc_samples, *f_test_mean.shape, device=train_data.device)
        test_predictions = f_test_mean + f_test_stds * epsilon

        test_predictions += train_label_mean
        test_error, test_mnll = classification_scores(test_predictions, test_labels)

    test_label_var = test_labels.var(unbiased=False).item()

    if isinstance(feature_encoder, GaussianApproximator):
        if config['method'] == 'maclaurin' and hasattr(feature_encoder.feature_encoder.measure, 'distribution'):
            feature_dist = str(feature_encoder.feature_encoder.measure.distribution)
        else:
            feature_dist = None

    feature_time = 0
    prediction_time = 0

    log_dir = {
        'dataset': data_name,
        'method': config['method'],
        'degree': config['degree'],
        'bias': config['bias'],
        'proj': config['proj'],
        'comp': config['complex_weights'],
        'comp_real': comp_real,
        'full_cov': full_cov,
        'hier': config['hierarchical'],
        'kernel_var': feature_encoder.log_var.exp().item(),
        'kernel_len': feature_encoder.log_lengthscale.exp().item(),
        'test_error': test_error,
        'test_mnll': test_mnll,
        'test_label_var': test_label_var,
        'test_kl': test_kl.item(),
        'test_mean_mse': test_mean_mse.item(),
        'test_var_mse': test_var_mse.item(),
        'D': d_features,
        'feature_dist': feature_dist,
        'noise_var': noise_var,
        'feature_time': feature_time,
        'pred_time': prediction_time,
        'seed': seed
    }
        
    return log_dir


if __name__ == '__main__':
    args = parse_args()

    # load RF and dataset config file
    try:
        with open(args.rf_parameter_file) as json_file:
            rf_parameters = json.load(json_file)
        with open(args.datasets_file) as json_file:
            datasets = json.load(json_file)
    except Exception as e:
        print('Cound not load file!', e)
        exit()

    start_time = time.time()

    for dataset_config in datasets['regression'] + datasets['classification']:
        print('Loading dataset: {}'.format(dataset_config))
        torch.manual_seed(42)
        np.random.seed(42)

        data = util.data.load_dataset(dataset_config, standardize=False, maxmin=False, normalize=False, split_size=0.9)
        data_name, train_data, test_data, train_labels, test_labels = data
        
        regression = False if dataset_config in datasets['classification'] else True

        if args.use_gpu:
            train_data = train_data.cuda()
            train_labels = train_labels.cuda()
            test_data = test_data.cuda()
            test_labels = test_labels.cuda()

        pow_2_shape = int(2**np.ceil(np.log2(train_data.shape[1])))
        n_samples = train_data.shape[0] + test_data.shape[0]

        log_handler = util.data.Log_Handler(rf_parameters['save_name'], '{}_d{}_n{}'.format(data_name, pow_2_shape, n_samples))
        csv_handler = util.data.DF_Handler(rf_parameters['save_name'], '{}_d{}_n{}'.format(data_name, pow_2_shape, n_samples))
        baseline_config = rf_parameters['baseline_config']

        baseline_config['bias'] = 0
        baseline_config['degree'] = 0
        if 'hierarchical' not in baseline_config.keys():
            baseline_config['hierarchical'] = False

        noise_var_opt = 4 #10**(-2)

        data_dict = prepare_data(
            # for the polynomial kernel we need to pass on the kernel parameters or kernel function
            baseline_config, args, rf_parameters, data_name,
            train_data, test_data, train_labels, test_labels,
            noise_var_opt, regression=regression, fit_ref_gp=True
        )

        noise_var_opt = data_dict['noise_var']
        
        configurations = rf_parameters['configurations']

        print('Comparing approximations...')
        
        dimensions = [pow_2_shape * i for i in range(
            rf_parameters['projection_range']['min'],
            rf_parameters['projection_range']['max']+1,
            rf_parameters['projection_range']['step']
        )]

        for seed in range(args.num_seeds):
            torch.manual_seed(seed)
            np.random.seed(seed)
            # create new train/val split for UCI datasets
            # if data_name not in ['MNIST', 'FashionMNIST', 'Adult', 'Cod_rna']:
            #     train_data = torch.cat([train_data, test_data], dim=0)
            #     train_labels = torch.cat([train_labels, test_labels], dim=0)
            #     current_train, current_test, current_train_labels, current_test_labels = util.data.create_train_val_split(train_data, train_labels, train_size=0.9)
            # else:
            #     current_train, current_test = train_data, test_data
            #     current_train_labels, current_test_labels = train_labels, test_labels

            # data_dict = prepare_data(
            #     # for the polynomial kernel we need to pass on the kernel parameters or kernel function
            #     baseline_config, args, rf_parameters, data_name,
            #     current_train, current_test, current_train_labels, current_test_labels,
            #     noise_var_opt, regression=regression
            # )

            # del current_train, current_test

            for d_features in dimensions:
                for config in configurations:
                    # add bias, lengthscale and degree for the polynomial kernel
                    if rf_parameters['kernel'] == 'polynomial':
                        config['bias'] = baseline_config['bias']
                        config['lengthscale'] = baseline_config['lengthscale']
                        config['degree'] = baseline_config['degree']
                    else:
                        config['bias'] = 0
                        config['degree'] = 0
                    if 'hierarchical' not in config.keys():
                        config['hierarchical'] = False

                    with torch.no_grad():
                        # try:
                        log_dir = run_rf_gp(data_dict, d_features, config, args, rf_parameters, seed)
                        log_handler.append(log_dir)
                        csv_handler.append(log_dir)
                        csv_handler.save()
                        # except Exception as e:
                        #     print(e)
                        #     print('Skipping current configuration...')
                        #     continue


        print('Total execution time: {:.2f}'.format(time.time()-start_time))
        print('Done!')
