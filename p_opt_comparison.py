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
import util.data

from util.helper_functions import kl_factorized_gaussian, spectral_norm, frobenius_norm
from util.helper_functions import classification_scores, regression_scores
from util.kernels import gaussian_kernel, polynomial_kernel
from util.measures import Fixed_Measure, Polynomial_Measure, P_Measure


"""
This is to compare p_opt for different polynomial and Gaussian kernels.
We show how p_max was determined.
"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_name', type=str, required=False, default='p_opt_comparison',
                        help='Path to csv output')
    parser.add_argument('--datasets_file', type=str, required=False, default='config/active_datasets3.json',
                        help='List of datasets to be used for the experiments')
    parser.add_argument('--num_data_samples', type=int, required=False, default=2000,
                        help='Number of data samples for lengthscale estimation')
    parser.add_argument('--num_seeds', type=int, required=False, default=20,
                        help='Number of seeds (runs)')
    parser.add_argument('--zero_center', dest='zero_center', action='store_true')
    parser.set_defaults(zero_center=True)
    parser.add_argument('--use_gpu', dest='use_gpu', action='store_true')
    parser.set_defaults(use_gpu=False)

    args = parser.parse_args()

    return args



def prepare_data(config, args, rf_parameters, data_name, current_train, current_test, train_labels, test_labels,
                    noise_var, regression=False):
    
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

    if args.zero_center:
        # we zero center the data
        current_train, current_test = util.data.standardize_data(current_train, current_test)
    else:
        # we subtract the minimum value of the training data to make data positive
        min_val = torch.min(current_train, 0)[0]
        current_train = current_train - min_val
        current_test = current_test - min_val

    if rf_parameters['kernel'] == 'gaussian':
        lengthscale = torch.cdist(current_train[train_idxs], current_train[train_idxs]).median().item()
        kernel_fun = lambda x, y: kernel_var * gaussian_kernel(
            x, y, lengthscale=lengthscale)
    else:
        # unit normalization
        current_train = current_train / current_train.norm(dim=1, keepdim=True)
        current_test = current_test / current_test.norm(dim=1, keepdim=True)

        lengthscale = config['lengthscale']
        kernel_fun = lambda x, y: kernel_var * polynomial_kernel(
            x, y, lengthscale=lengthscale, k=config['degree'], c=config['bias'])

    # float conversion
    current_train = current_train.float()
    current_test = current_test.float()
    train_labels = train_labels.float()
    test_labels = test_labels.float()
    mm = mm.float()
    vv = vv.float()
    ref_kernel = kernel_fun(current_test[test_idxs], current_test[test_idxs])

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
        'train_idxs': train_idxs,
        'test_idxs': test_idxs,
        'regression': regression
    }

    return meta_data_dict

def run_rf_gp(data_dict, down_features, up_features, config, args, rf_params, seed):
    """
    Runs a random feature GP and computes performance compared to ground truth.
    Returns a dictionary of performance metrics and meta data to be logged later on.

    data_dict: meta data dictionary obtained through prepare_data method
    d_features: projection dimension
    config: random feature method configuration
    args: command-line arguments (e.g. use_gpu)
    rf_params: random feature parameters from parameter file
    seed: current seed
    """
    
    # if there is a bias appended to the input data, we need to make sure that input_dim+1 is padded with zeros
    if (rf_params['kernel'] == 'polynomial' and config['method'] == 'poly_sketch' and config['bias'] != 0) \
        or (rf_params['kernel'] == 'gaussian' and config['method'] == 'poly_sketch'):
        offset = 1
    else:
        offset = 0
    
    train_data_padded = util.data.pad_data_pow_2(data_dict['train_data'], offset=offset)
    test_data_padded = util.data.pad_data_pow_2(data_dict['test_data'], offset=offset)
    train_idxs = data_dict['train_idxs']
    test_idxs = data_dict['test_idxs']
    ref_kernel = data_dict['ref_kernel']
    noise_var = data_dict['noise_var']

    for optional_key in ['complex_real', 'ahle', 'tree', 'craft', 'full_cov']:
        if optional_key not in config.keys():
            config[optional_key] = False

    proj_dim = up_features if config['craft'] else down_features

    if args.use_gpu:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    start = time.time()

    if rf_params['kernel'] == 'gaussian':
        # we use a wrapper class that summarizes all Gaussian approximators
        feature_encoder = GaussianApproximator(
            train_data_padded.shape[1], proj_dim,
            approx_degree=rf_params['max_sampling_degree'], lengthscale=data_dict['lengthscale'],
            var=data_dict['kernel_var'], trainable_kernel=False, method=config['method'],
            projection_type=config['proj'], # ahle=config['ahle'], tree=config['tree'],
            complex_weights=config['complex_weights'], device=('cuda' if args.use_gpu else 'cpu')
        )
        feature_encoder.initialize_sampling_distribution(train_data_padded[train_idxs],
            min_sampling_degree=rf_params['min_sampling_degree'])
    # otherwise we use the polynomial kernel
    elif config['method'].startswith('maclaurin'):
        # the maclaurin series for the polynomial kernel function
        kernel_coefs = lambda x: Polynomial_Measure.coefs(x, config['degree'], config['bias'])
        # we initialize the distribution over degrees to be uniform (this will be overridden later)
        measure = Fixed_Measure(False, [1]*config['degree'], True)

        feature_encoder = Maclaurin(train_data_padded.shape[1], proj_dim, coef_fun=kernel_coefs,
                                    module_args={
                                        'projection': config['proj'],
                                        'ahle': config['ahle'],
                                        'tree': config['tree'],
                                        'complex_weights': config['complex_weights'],
                                        'complex_real': config['complex_real']
                                    },
                                    measure=measure, bias=0, device=('cuda' if args.use_gpu else 'cpu'),
                                    lengthscale=data_dict['lengthscale'],
                                    var=data_dict['kernel_var'], ard=False, trainable_kernel=False)

        random_samples = train_data_padded[train_idxs]
        target_kernel = polynomial_kernel(
            random_samples, lengthscale=feature_encoder.log_lengthscale.exp(),
            k=config['degree'], c=config['bias']
        )
        exp_vars, exp_covs, exp_sq_biases = feature_encoder.expected_variances_and_biases(
            random_samples, target_kernel, gaussian_kernel=False)
        feature_encoder.optimize_sampling_distribution(exp_vars, exp_covs, exp_sq_biases,
            min_degree=rf_parameters['min_sampling_degree'])
        print('Optimized distribution: {}'.format(feature_encoder.measure.distribution))

    if args.use_gpu:
        torch.cuda.synchronize()

    maclaurin_time = time.time() - start

    if config['method'] == 'srf':
        feature_encoder.resample(num_points_w=5000)
    else:
        feature_encoder.resample()

    # before computing the random features, we empty the cuda cache
    del data_dict

    if args.use_gpu:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    start = time.time()

    num_elements = 5000

    train_features = torch.zeros(len(train_data_padded), down_features, device=('cuda' if args.use_gpu else 'cpu'), dtype=torch.cfloat)
    test_features = torch.zeros(len(test_data_padded), down_features, device=('cuda' if args.use_gpu else 'cpu'), dtype=torch.cfloat)

    for phase in ['train', 'test']:
        if phase == 'train':
            data = train_data_padded
        else:
            data = test_data_padded

        num_splits = int(np.ceil(len(data)/float(num_elements)))
        # projections = []
        for i in range(num_splits):
            features = feature_encoder.forward(data[i*num_elements:(i+1)*num_elements])

            # projections.append(features)

            if phase == 'train':
                # train_features = torch.cat(projections, dim=0)
                train_features[i*num_elements:(i+1)*num_elements, :] = features
            else:
                # test_features = torch.cat(projections, dim=0)
                test_features[i*num_elements:(i+1)*num_elements, :] = features

    if args.use_gpu:
        torch.cuda.synchronize()

    feature_time = time.time() - start

    ### kernel approximation on a subset of the test data
    approx_kernel = test_features[test_idxs] @ test_features[test_idxs].conj().t()
    if config['complex_real']:
        approx_kernel = approx_kernel.real

    frob_error, rel_frob_error = frobenius_norm(approx_kernel, ref_kernel)
    # spec_error, rel_spec_error = spectral_norm(approx_kernel, ref_kernel)

    if isinstance(feature_encoder, GaussianApproximator):
        feature_dist = str(feature_encoder.feature_encoder.measure.distribution) if config['method'] == 'maclaurin' else 'None'
    else:
        feature_dist = str(feature_encoder.measure.distribution) if config['method'] == 'maclaurin' else 'None'

    log_dir = {
        'dataset': data_name,
        'kernel': rf_params['kernel'],
        'method': config['method'],
        'degree': config['degree'],
        'bias': config['bias'],
        'proj': config['proj'],
        'comp': config['complex_weights'],
        'comp_real': config['complex_real'],
        'full_cov': config['full_cov'],
        'ahle': config['ahle'],
        'tree': config['tree'],
        'kernel_var': feature_encoder.log_var.exp().item(),
        'kernel_len': feature_encoder.log_lengthscale.exp().item(),
        'k_frob_error': frob_error.item(),
        'k_rel_frob_error': rel_frob_error.item(),
        # 'k_spec_error': spec_error.item(),
        # 'k_rel_spec_error': rel_spec_error.item(),
        'D': down_features,
        'E': up_features,
        'feature_dist': feature_dist,
        'noise_var': noise_var,
        'feature_time': feature_time,
        'maclaurin_time': maclaurin_time,
        'seed': seed
    }
        
    return log_dir



if __name__ == '__main__':
    args = parse_args()

    # load RF and dataset config file
    try:
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

        log_handler = util.data.Log_Handler(args.save_name, '{}_d{}_n{}_centering_{}_samples_{}'.format(data_name, pow_2_shape, n_samples, args.zero_center, args.num_data_samples))
        csv_handler = util.data.DF_Handler(args.save_name, '{}_d{}_n{}_centering_{}_samples_{}'.format(data_name, pow_2_shape, n_samples, args.zero_center, args.num_data_samples))

        kernels = [
            #{'kernel': 'polynomial', 'a': 2, 'degree': 3, 'min_sampling_degree': 2, 'max_sampling_degree': 20},
            #{'kernel': 'polynomial', 'a': 2, 'degree': 7, 'min_sampling_degree': 2, 'max_sampling_degree': 20},
            #{'kernel': 'polynomial', 'a': 2, 'degree': 10, 'min_sampling_degree': 2, 'max_sampling_degree': 20},
            #{'kernel': 'polynomial', 'a': 2, 'degree': 20, 'min_sampling_degree': 2, 'max_sampling_degree': 20},
            {'kernel': 'gaussian', 'min_sampling_degree': 1, 'max_sampling_degree': 20}
        ]

        for kernel_config in kernels:

            if kernel_config['kernel'] == 'polynomial':
                # we follow the spherical random features paper for unit-norm. data
                # k(x,y) = alpha * (q + xTy)^p with alpha=(2/a^2)^p, q=a^2/2-1
                # => k(x,y) = ((1-2/a^2) + 2/a^2 cos(theta))^p <= 1
                # a=2: emphasizes degree 5
                # a=3: emphasizes degree 2, then 1&3, then 0&4
                # a=4: emphasizes degree 1, then 0&2, then rest
                kernel_config['bias'] = 1.-2./kernel_config['a']**2
                kernel_config['lengthscale'] = kernel_config['a'] / np.sqrt(2.)
            else:
                kernel_config['bias'] = 0
                kernel_config['degree'] = 0
                kernel_config['a'] = 0

            noise_var_opt = 10**(-3)
            
            configurations = [
                {"method": "maclaurin", "proj": "rademacher", "ahle": False, "complex_weights": False},
                {"method": "maclaurin", "proj": "rademacher", "ahle": False, "complex_weights": True},
                {"method": "maclaurin", "proj": "srht", "ahle": False, "complex_weights": False, "full_cov": False},
                {"method": "maclaurin", "proj": "srht", "ahle": False, "complex_weights": True, "full_cov": False}
            ]

            rf_parameters = {
                # 'configurations': configurations,
                'kernel': kernel_config['kernel'],
                'a': kernel_config['a'],
                'degree': kernel_config['degree'],
                'min_sampling_degree': 2,
                'max_sampling_degree': 20
            }

            print('Comparing approximations...')

            down_features = pow_2_shape * 5

            for seed in range(args.num_seeds):
                torch.manual_seed(seed)
                np.random.seed(seed)
                # create new train/val split for UCI datasets
                if data_name not in ['MNIST', 'FashionMNIST', 'Adult', 'Cod_rna', 'cifar10', 'cifar10_conv']:
                    train_data = torch.cat([train_data, test_data], dim=0)
                    train_labels = torch.cat([train_labels, test_labels], dim=0)
                    current_train, current_test, current_train_labels, current_test_labels = util.data.create_train_val_split(train_data, train_labels, train_size=0.9)
                else:
                    current_train, current_test = train_data, test_data
                    current_train_labels, current_test_labels = train_labels, test_labels

                data_dict = prepare_data(
                    # for the polynomial kernel we need to pass on the kernel parameters or kernel function
                    kernel_config, args, rf_parameters, data_name,
                    current_train, current_test, current_train_labels, current_test_labels,
                    noise_var_opt, regression=regression
                )

                del current_train, current_test

                for config in configurations:

                    with torch.no_grad():
                        #try:
                            log_dir = run_rf_gp(data_dict, down_features, 0, {**config, **kernel_config}, args, rf_parameters, seed)
                            log_handler.append(log_dir)
                            csv_handler.append(log_dir)
                            csv_handler.save()
                        # except Exception as e:
                        #     print(e)
                        #     print('Skipping current configuration...')
                        #     continue


            print('Total execution time: {:.2f}'.format(time.time()-start_time))
            print('Done!')
