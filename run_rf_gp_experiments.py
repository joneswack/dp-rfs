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

from util.helper_functions import kl_factorized_gaussian
from util.helper_functions import classification_scores, regression_scores
from util.kernels import gaussian_kernel, polynomial_kernel
from util.measures import Fixed_Measure, Polynomial_Measure, P_Measure

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rf_parameter_file', type=str, required=True,
                        help='Path to RF parameter file')
    parser.add_argument('--datasets_file', type=str, required=False, default='config/active_datasets.json',
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

    if rf_parameters['kernel'] == 'gaussian':
        # we zero center the data
        current_train, current_test = util.data.standardize_data(current_train, current_test)
        lengthscale = torch.cdist(current_train[train_idxs], current_train[train_idxs]).median().item()
        kernel_fun = lambda x, y: kernel_var * gaussian_kernel(
            x, y, lengthscale=lengthscale)
    else:
        if data_name not in ['MNIST']:
            # we skip zero centering for mnist for the polynomial kernel
            # current_train, current_test = util.data.standardize_data(current_train, current_test)
            pass
        # unit normalization
        current_train = current_train / current_train.norm(dim=1, keepdim=True)
        current_test = current_test / current_test.norm(dim=1, keepdim=True)

        lengthscale = config['lengthscale']
        kernel_fun = lambda x, y: kernel_var * polynomial_kernel(
            x, y, lengthscale=lengthscale, k=config['degree'], c=config['bias'])

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

def run_rf_gp(data_dict, d_features, config, args, rf_params, seed):
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
    f_test_mean_ref = data_dict['f_test_mean_ref']
    f_test_stds_ref = data_dict['f_test_stds_ref']
    regression = data_dict['regression']

    if rf_params['kernel'] == 'gaussian':
        # we use a wrapper class that summarizes all Gaussian approximators
        feature_encoder = GaussianApproximator(
            train_data_padded.shape[1], d_features,
            approx_degree=rf_params['max_sampling_degree'], lengthscale=data_dict['lengthscale'],
            var=data_dict['kernel_var'], trainable_kernel=False, method=config['method'],
            projection_type=config['proj'], hierarchical=config['hierarchical'],
            complex_weights=config['complex_weights']
        )

        if args.use_gpu:
            feature_encoder.cuda()
        feature_encoder.initialize_sampling_distribution(train_data_padded[train_idxs],
            min_sampling_degree=rf_params['min_sampling_degree'])
    # otherwise we use the polynomial kernel
    elif config['method'] == 'srf':
        feature_encoder = Spherical(train_data_padded.shape[1], d_features,
                            lengthscale=1.0, var=1.0, ard=False,
                            discrete_pdf=False, num_pdf_components=10,
                            complex_weights=config['complex_weights'],
                            projection_type=config['proj'])
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

        feature_encoder = Maclaurin(train_data_padded.shape[1], d_features, coef_fun=kernel_coefs,
                                    module_args={
                                        'projection': config['proj'],
                                        'hierarchical': config['hierarchical'],
                                        'complex_weights': config['complex_weights']
                                    },
                                    measure=measure, bias=0, device=('cuda' if args.use_gpu else 'cpu'),
                                    lengthscale=data_dict['lengthscale'],
                                    var=data_dict['kernel_var'], ard=False, trainable_kernel=False)

        if config['method'] == 'maclaurin':
            # optimized maclaurin
            if args.use_gpu:
                feature_encoder.cuda()
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
        feature_encoder = PolynomialSketch(train_data_padded.shape[1], d_features,
                                        degree=config['degree'], bias=config['bias'],
                                        projection_type=config['proj'], hierarchical=config['hierarchical'],
                                        complex_weights=config['complex_weights'],
                                        lengthscale=data_dict['lengthscale'], device=('cuda' if args.use_gpu else 'cpu'),
                                        var=data_dict['kernel_var'], ard=False, trainable_kernel=False)
        
    if config['method'] == 'srf':
        feature_encoder.resample(num_points_w=5000)
    else:
        feature_encoder.resample()

    if args.use_gpu:
        if isinstance(feature_encoder, GaussianApproximator):
            feature_encoder.feature_encoder.cuda()
            if config['method'] != 'rff':
                feature_encoder.feature_encoder.move_submodules_to_cuda()
        else:
            feature_encoder.cuda()
            feature_encoder.move_submodules_to_cuda()

    het_gp = HeteroskedasticGP(None)

    # before computing the random features, we empty the cuda cache
    del data_dict
    torch.cuda.empty_cache()

    torch.cuda.synchronize()
    start = time.time()

    train_features = feature_encoder.forward(train_data_padded)
    test_features = feature_encoder.forward(test_data_padded)

    torch.cuda.synchronize()
    feature_time = time.time() - start

    ### kernel approximation on a subset of the test data
    approx_kernel = test_features[test_idxs] @ test_features[test_idxs].conj().t()
    if config['complex_weights']:
        approx_kernel = approx_kernel.real

    difference = approx_kernel - ref_kernel
    mse = difference.pow(2).mean()
    mean_error = difference.abs().mean()
    # ||K_hat - K|| / ||K||
    rel_frob_error = (difference.pow(2).sum().sqrt() / ref_kernel.pow(2).sum().sqrt())

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
    torch.cuda.synchronize()
    start = time.time()

    if regression:
        f_test_mean, f_test_stds = het_gp.predictive_dist(
            train_features, test_features,
            train_labels, train_label_vars
        )
        
        torch.cuda.synchronize()
        prediction_time = time.time() - start

        f_test_mean += train_label_mean
        test_error, test_mnll = regression_scores(f_test_mean, f_test_stds**2 + noise_var, test_labels)
    else:
        test_predictions = het_gp.predictive_sample(
            train_features, test_features, train_labels, train_label_vars,
            num_samples=args.num_mc_samples
        )

        torch.cuda.synchronize()
        prediction_time = time.time() - start

        test_predictions += train_label_mean
        test_error, test_mnll = classification_scores(test_predictions, test_labels)

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
        'hier': config['hierarchical'],
        'kernel_var': feature_encoder.log_var.exp().item(),
        'kernel_len': feature_encoder.log_lengthscale.exp().item(),
        'test_error': test_error,
        'test_mnll': test_mnll,
        'test_label_var': test_label_var,
        'k_mse': mse.item(),
        'k_mean_error': mean_error.item(),
        'k_rel_frob_error': rel_frob_error.item(),
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
        if 'hierarchical' not in baseline_config.keys():
            baseline_config['hierarchical'] = False

        print('Determining noise variance...')
        # we select noise_vars according to validation mnll on random fourier features
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

                    log_dir = run_rf_gp(data_dict, d_features, baseline_config, args, rf_parameters, 0)
                except Exception as e:
                    print(e)
                    print('Skipping current configuration...')
                    continue

            noise_var_csv_handler.append(log_dir)
            log_handler.append(log_dir)
            noise_var_csv_handler.save()

        noise_var_df = pd.read_csv(noise_var_csv_handler.file_path)
        noise_var_opt = noise_var_df.sort_values('test_mnll', axis=0, ascending=True)['noise_var'].values[0]

        print('Optimal noise var: {}'.format(noise_var_opt))

        # alpha_opt = 10**(-1)

        del sub_data, val_data
        
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
            if data_name not in ['MNIST', 'FashionMNIST', 'Adult', 'Cod_rna']:
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

            for d_features in dimensions:
                for config in configurations:
                    if config['complex_weights'] and d_features > 5*pow_2_shape:
                        continue
                    
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
        # Prevent script from finishing!
        # while True:
            # continue
