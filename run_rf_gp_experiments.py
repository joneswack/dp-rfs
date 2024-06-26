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
Runs Gaussian Process Classification experiments as closed form GP regression on transformed labels.
"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rf_parameter_file', type=str, required=False, default='config/rf_parameters/poly7_a2.json',
                        help='Path to RF parameter file')
    parser.add_argument('--datasets_file', type=str, required=False, default='config/active_datasets3.json',
                        help='List of datasets to be used for the experiments')
    parser.add_argument('--num_data_samples', type=int, required=False, default=5000,
                        help='Number of data samples for lengthscale estimation')
    parser.add_argument('--num_mc_samples', type=int, required=False, default=1000,
                        help='Number of mc samples for predictive distribution')
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
    """
    Returns a dictionary with:
        - normalized/rescaled data
        - mean-subtracted training labels
            (including conversion for classification according to Milios et al., 2018)
        - kernel variance (mean of the training labels)
        - reference kernel evaluated on num_samples
        - exact GP predictive distribution obtained from the same training samples
        - ... all other meta data needed later on

    config: method config containing kernel parameters
    args: command line arguments
    rf_params: random feature parameters from parameter file
    data_name: name of the dataset
    current_train: current train split
    current_test: current test split
    train_labels: current train split labels
    test_labels: current test split labels
    noise_var: noise variance for regression / transformed classification labels
    regression: whether to use regression or transformed classification labels
    """
    
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
    train_labels = data_dict['train_labels']
    train_label_mean = data_dict['train_label_mean']
    train_label_vars = data_dict['train_label_vars']
    test_labels = data_dict['test_labels']
    noise_var = data_dict['noise_var']
    ref_kernel = data_dict['ref_kernel']
    f_test_mean_ref = data_dict['f_test_mean_ref'].clone()
    f_test_stds_ref = data_dict['f_test_stds_ref'].clone()
    regression = data_dict['regression']

    for optional_key in ['complex_real', 'ahle', 'tree', 'craft', 'full_cov']:
        if optional_key not in config.keys():
            config[optional_key] = False

    proj_dim = up_features if config['craft'] else down_features

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
    elif config['method'] == 'srf':
        feature_encoder = Spherical(train_data_padded.shape[1], proj_dim,
                            lengthscale=1.0, var=1.0, ard=False,
                            discrete_pdf=False, num_pdf_components=10,
                            complex_weights=config['complex_weights'],
                            projection_type=config['proj'], device=('cuda' if args.use_gpu else 'cpu'))
        feature_encoder.load_model(rf_params['srf_model_path_prefix'] + '_d{}.torch'.format(
            int(train_data_padded.shape[1])
        ))
        # loading the model resets the lengthscale and variance
        feature_encoder.log_lengthscale.data = torch.ones_like(feature_encoder.log_lengthscale) * np.log(data_dict['lengthscale'])
        feature_encoder.log_var.data = torch.ones_like(feature_encoder.log_lengthscale) * np.log(data_dict['kernel_var'])
    elif config['method'].startswith('maclaurin'):
        if config['method'] == 'maclaurin':
            # the maclaurin series for the polynomial kernel function
            kernel_coefs = lambda x: Polynomial_Measure.coefs(x, config['degree'], config['bias'])
            # we initialize the distribution over degrees to be uniform (this will be overridden later)
            measure = Fixed_Measure(False, [1]*config['degree'], True)
        elif config['method'] == 'maclaurin_p':
            kernel_coefs = lambda x: Polynomial_Measure.coefs(x, config['degree'], config['bias'])
            measure = P_Measure(2., False, config['degree'])

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

        if config['method'] == 'maclaurin':
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
    else:
        feature_encoder = PolynomialSketch(train_data_padded.shape[1], proj_dim,
                                        degree=config['degree'], bias=config['bias'],
                                        projection_type=config['proj'], ahle=config['ahle'], tree=config['tree'],
                                        complex_weights=config['complex_weights'], complex_real=config['complex_real'],
                                        full_cov=config['full_cov'], lengthscale=data_dict['lengthscale'],
                                        device=('cuda' if args.use_gpu else 'cpu'),
                                        var=data_dict['kernel_var'], ard=False, trainable_kernel=False)
        
    if config['method'] == 'srf':
        feature_encoder.resample(num_points_w=5000)
    else:
        feature_encoder.resample()

    if config['craft']:
        feature_encoder_2 = PolynomialSketch(up_features, down_features,
                                        degree=1, bias=0,
                                        projection_type='srht', ahle=False, tree=False,
                                        complex_weights=False, complex_real=False,
                                        full_cov=False, lengthscale=1.,
                                        device=('cuda' if args.use_gpu else 'cpu'),
                                        var=1.0, ard=False, trainable_kernel=False)
        feature_encoder_2.resample()

    het_gp = HeteroskedasticGP(None)

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

            if config['craft']:
                features = feature_encoder_2.forward(features)

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

    ### run subsampled GP for KL divergence
    f_test_mean_est, f_test_stds_est = het_gp.predictive_dist(
        train_features[train_idxs],
        test_features[test_idxs],
        train_labels[train_idxs],
        train_label_vars[train_idxs]
    )

    test_kl = kl_factorized_gaussian(
        f_test_mean_est, f_test_mean_ref,
        f_test_stds_est, f_test_stds_ref
    ).sum(dim=0).mean()
    test_mean_mse = (f_test_mean_ref - f_test_mean_est).pow(2).mean()
    test_var_mse = (f_test_stds_ref**2 - f_test_stds_est**2).pow(2).mean()

    ### gp prediction
    if args.use_gpu:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    start = time.time()

    # run experiment on all data for test errors
    if regression:
        f_test_mean, f_test_stds = het_gp.predictive_dist(
            train_features, test_features,
            train_labels, train_label_vars
        )
        
        if args.use_gpu:
            torch.cuda.synchronize()
        prediction_time = time.time() - start

        f_test_mean += train_label_mean
        test_error, test_mnll = regression_scores(f_test_mean, f_test_stds**2 + noise_var, test_labels)

        # Compute test errors using full GP (make sure to set the sample size sufficiently high)
        # should work for all datasets except for protein
        f_test_mean_ref += train_label_mean
        test_error_ref, test_mnll_ref = regression_scores(f_test_mean_ref, f_test_stds_ref**2 + noise_var, test_labels[test_idxs])
    else:
        test_predictions = het_gp.predictive_sample(
            train_features, test_features, train_labels, train_label_vars,
            num_samples=args.num_mc_samples
        )

        if args.use_gpu:
            torch.cuda.synchronize()
        prediction_time = time.time() - start

        test_predictions += train_label_mean
        test_error, test_mnll = classification_scores(test_predictions, test_labels)

        # Compute test errors using full GP (make sure to set the sample size sufficiently high)
        # should work for all datasets except for protein
        f_test_mean_ref += train_label_mean
        # test_error_ref, test_error_ref = classification_scores(f_test_mean_ref, test_labels[test_idxs])
        test_acc_ref = (f_test_mean_ref.argmax(dim=1) == test_labels[test_idxs].argmax(dim=1)).float().mean().item()
        test_error_ref = 1.0 - test_acc_ref
        test_mnll_ref = 0

    test_label_var = test_labels.var(unbiased=False).item()

    if isinstance(feature_encoder, GaussianApproximator):
        feature_dist = str(feature_encoder.feature_encoder.measure.distribution) if config['method'] == 'maclaurin' else 'None'
    else:
        feature_dist = str(feature_encoder.measure.distribution) if config['method'] == 'maclaurin' else 'None'

    log_dir = {
        'dataset': data_name,
        'method': config['method'],
        'degree': config['degree'],
        'bias': config['bias'],
        'proj': config['proj'],
        'comp': config['complex_weights'],
        'comp_real': config['complex_real'],
        'craft': config['craft'],
        'full_cov': config['full_cov'],
        'ahle': config['ahle'],
        'tree': config['tree'],
        'kernel_var': feature_encoder.log_var.exp().item(),
        'kernel_len': feature_encoder.log_lengthscale.exp().item(),
        'test_error': test_error,
        'test_mnll': test_mnll,
        'test_error_ref': test_error_ref,
        'test_mnll_ref': test_mnll_ref,
        'test_label_var': test_label_var,
        'k_frob_error': frob_error.item(),
        'k_rel_frob_error': rel_frob_error.item(),
        # 'k_spec_error': spec_error.item(),
        # 'k_rel_spec_error': rel_spec_error.item(),
        'test_kl': test_kl.item(),
        'test_mean_mse': test_mean_mse.item(),
        'test_var_mse': test_var_mse.item(),
        'D': down_features,
        'E': up_features,
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

        log_handler = util.data.Log_Handler(rf_parameters['save_name'], '{}_d{}_n{}_centering_{}'.format(data_name, pow_2_shape, n_samples, args.zero_center))
        csv_handler = util.data.DF_Handler(rf_parameters['save_name'], '{}_d{}_n{}_centering_{}'.format(data_name, pow_2_shape, n_samples, args.zero_center))
        baseline_config = rf_parameters['baseline_config']

        if rf_parameters['kernel'] == 'polynomial':
            # we follow the spherical random features paper for unit-norm. data
            # k(x,y) = alpha * (q + xTy)^p with alpha=(2/a^2)^p, q=a^2/2-1
            # => k(x,y) = ((1-2/a^2) + 2/a^2 cos(theta))^p <= 1
            # a=2: emphasizes degree 5
            # a=3: emphasizes degree 2, then 1&3, then 0&4
            # a=4: emphasizes degree 1, then 0&2, then rest
            bias = 1.-2./rf_parameters['a']**2
            lengthscale = rf_parameters['a'] / np.sqrt(2.)
            baseline_config['bias'] = bias
            baseline_config['lengthscale'] = lengthscale
            baseline_config['degree'] = rf_parameters['degree']
        else:
            baseline_config['bias'] = 0
            baseline_config['degree'] = 0
        if 'ahle' not in baseline_config.keys():
            baseline_config['ahle'] = False
            baseline_config['tree'] = False

        print('Determining noise variance...')
        # we select noise_vars according to validation mnll on random fourier features
        # d_features = int(2 ** rf_parameters['projection_range']['max'])
        d_features = int(pow_2_shape * rf_parameters['projection_range']['max'])

        noise_var_csv_handler = util.data.DF_Handler(rf_parameters['save_name'] + '_noise_var', '{}'.format(data_name))
        sub_data, val_data, sub_labels, val_labels = util.data.create_train_val_split(train_data, train_labels, train_size=0.9)

        if regression:
            noise_var_params = rf_parameters['noise_var_range']['regression']
        else:
            noise_var_params = rf_parameters['noise_var_range']['classification']
        noise_vars = [
            float(noise_var_params['base'])**i for i in range(noise_var_params['min'], noise_var_params['max']+1)
        ]
        
        for noise_var in noise_vars:
            # config, data_name, current_train, current_test, train_labels, test_labels, num_samples, noise_var, regression=False
            with torch.no_grad():
                try:
                    data_dict = prepare_data(baseline_config, args, rf_parameters,
                        data_name, sub_data, val_data, sub_labels, val_labels,
                        noise_var, regression=regression
                    )

                    log_dir = run_rf_gp(data_dict, d_features, d_features, baseline_config, args, rf_parameters, 0)
                except Exception as e:
                    print(e)
                    print('Skipping current configuration...')
                    continue

            noise_var_csv_handler.append(log_dir)
            log_handler.append(log_dir)
            noise_var_csv_handler.save()

        noise_var_df = pd.read_csv(noise_var_csv_handler.file_path)
        noise_var_opt = noise_var_df.sort_values('test_mnll_ref', axis=0, ascending=True)['noise_var'].values[0]

        print('Optimal noise var: {}'.format(noise_var_opt))

        # noise_var_opt = 64.0 # 10**(-3)

        del sub_data, val_data
        
        configurations = rf_parameters['configurations']

        print('Comparing approximations...')
        
        down_features_list = [pow_2_shape * i for i in range(
            rf_parameters['projection_range']['min'],
            rf_parameters['projection_range']['max']+1,
            rf_parameters['projection_range']['step']
        )]

        # down_features_list = [2**i for i in range(
        #     rf_parameters['projection_range']['min'],
        #     rf_parameters['projection_range']['max']+1,
        #     rf_parameters['projection_range']['step']
        # )] + [10240]
        # up projection dimension of craft maps (must be power of 2 for subsequent srht)
        # up_features = pow_2_shape * rf_parameters['craft_factor']
        # up_features = int(2**np.ceil(np.log2(up_features)))
        up_features = rf_parameters['craft_factor'] * pow_2_shape # 2**rf_parameters['craft_factor']
        # dimensions = [int(pow_2_shape * i) for i in range(1,11)] # [0.125, 0.25, 0.5]
        # dimensions += [2**i for i in range(7, 14)]
        # dimensions = list(set(dimensions))

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
                baseline_config, args, rf_parameters, data_name,
                current_train, current_test, current_train_labels, current_test_labels,
                noise_var_opt, regression=regression
            )

            del current_train, current_test

            for down_features in down_features_list:
                for config in configurations:
                    
                    # add bias, lengthscale and degree for the polynomial kernel
                    if rf_parameters['kernel'] == 'polynomial':
                        config['bias'] = baseline_config['bias']
                        config['lengthscale'] = baseline_config['lengthscale']
                        config['degree'] = baseline_config['degree']
                    else:
                        config['bias'] = 0
                        config['degree'] = 0

                    with torch.no_grad():
                        try:
                            log_dir = run_rf_gp(data_dict, down_features, up_features, config, args, rf_parameters, seed)
                            log_handler.append(log_dir)
                            csv_handler.append(log_dir)
                            csv_handler.save()
                        except Exception as e:
                            print(e)
                            print('Skipping current configuration...')
                            continue


        print('Total execution time: {:.2f}'.format(time.time()-start_time))
        print('Done!')
