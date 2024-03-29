import torch
from random_features.polynomial_sketch import PolynomialSketch
from random_features.spherical import Spherical
import util.data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from timeit import default_timer as timer
from tqdm import tqdm

device = 'cuda'
repetitions = 100
torch.manual_seed(0)

"""
Measures Random Feature Computation Time vs. Kernel Approximation.
"""

# featuresdict_1 = {'2048': torch.load('saved_models/conv_features/cifar10-val-inc-v3-2048.pth')}
#featuresdict_2 = {'2048': torch.load('saved_models/conv_features/cifar10-train-inc-v3-2048.pth')}
# input_data_1 = featuresdict_1['2048'][:, :-1].to(device)
#input_data = featuresdict_2['2048'][:, :-1].to(device)
#del featuresdict_2

# data = util.data.load_dataset('config/datasets/adult.json', standardize=False, maxmin=False, normalize=False, split_size=0.9)
# data_name, train_data, test_data, train_labels, test_labels = data
# # pad with zeros
# train_data = train_data / torch.max(train_data, 0)[0]
# input_data = torch.zeros(len(train_data), 127, dtype=train_data.dtype)
# input_data[:, :train_data.shape[1]] = train_data
# # we need min-max scaling
# d = 128

# data = util.data.load_dataset('config/datasets/mocap.json', standardize=False, maxmin=False, normalize=False, split_size=0.9)
# data_name, train_data, test_data, train_labels, test_labels = data
# # pad with zeros
# # make data positive
# train_data = train_data - torch.min(train_data, 0)[0]
# # normalize data
# # train_data = train_data / torch.max(train_data, 0)[0]
# train_data = train_data / train_data.norm(dim=1, keepdim=True)
# input_data = torch.zeros(len(train_data), 1023, dtype=train_data.dtype, device=device)
# input_data[:, :train_data.shape[1]] = train_data.to(device)
# # we need min-max scaling
# d = 64

data = util.data.load_dataset('config/datasets/mnist.json', standardize=False, maxmin=False, normalize=False, split_size=0.9)
data_name, train_data, test_data, train_labels, test_labels = data
# pad with zeros
# make data positive
train_data = train_data - torch.min(train_data, 0)[0]
# normalize data
# train_data = train_data / torch.max(train_data, 0)[0]
train_data = train_data / train_data.norm(dim=1, keepdim=True)
input_data = torch.zeros(len(train_data), 1023, dtype=train_data.dtype)
input_data[:, :train_data.shape[1]] = train_data
input_data = input_data.to(device)
# we need min-max scaling
d = 1024

# input_data_1 = torch.from_numpy(np.load('saved_models/lenet_test.npy'))[:, :-1].to(device)
# input_data = torch.from_numpy(np.load('saved_models/conv_features/lenet_train.npy'))[:, :-1].to(device)
# d=512



# polynomial kernel parameters
p=3
# bias=1
# ONLY IF NOT UNIT-NORMALIZED
# lengthscale = np.sqrt(input_data.shape[1])
# IF UNIT-NORMALIZED
# lengthscale = 1.
# for SRF
a = 2
bias = 1.-2./a**2
lengthscale = a / np.sqrt(2.)

# size of the random subset of the input data
subsample_size = 1000
# this data sample will be recomputed for every repetition later on
input_data_sample = input_data[torch.randperm(len(input_data), device=device)[:subsample_size]]
# rf_dims = [64, 128, 256, 512, 1024, 2048, 2048*2, 2048*3, 2048*4, 2048*5, 2048*6]
rf_dims_slow = [i*d for i in range(1, 21)]
rf_dims_fast = [i*d for i in range(1, 31)]
# rf_dims = [512]
# rf_dims = [512*2]
rf_configs = [
    {'proj': 'srf', 'full_cov': False, 'complex_weights': False, 'complex_real': False, 'hierarchical': False},
    {'proj': 'countsketch_scatter', 'full_cov': False, 'complex_weights': False, 'complex_real': False},
    # {'proj': 'gaussian', 'full_cov': False, 'complex_real': False},
    # {'proj': 'gaussian', 'full_cov': False, 'complex_real': True},
    {'proj': 'srht', 'full_cov': True, 'complex_weights': False, 'complex_real': False},
    {'proj': 'srht', 'full_cov': True, 'complex_weights': False, 'complex_real': True},
    {'proj': 'rademacher', 'full_cov': False, 'complex_weights': False, 'complex_real': False},
    # {'proj': 'rademacher', 'full_cov': False, 'complex_weights': False, 'complex_real': True},
    # {'proj': 'srht', 'full_cov': False, 'complex_weights': False, 'complex_real': False},
    # {'proj': 'srht', 'full_cov': False, 'complex_weights': True, 'complex_real': False},
]

log_handler = util.data.Log_Handler('time_benchmark', 'rep{}_p{}_bias{}_len_{:.2f}_mnist'.format(repetitions, p, bias, lengthscale))
csv_handler = util.data.DF_Handler('time_benchmark', 'rep{}_p{}_bias{}_len_{:.2f}_mnist'.format(repetitions, p, bias, lengthscale))

def polynomial_kernel(X, Y, degree=3, gamma=None, coef0=1):
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    K = (X @ Y.t() * gamma + coef0) ** degree
    return K

for config in rf_configs:
    if config['proj'] == 'srht' and (not config['complex_real']) \
        and (not config['complex_weights']):
        # R-TensorSRHT (DB/IB)
        rf_dims = rf_dims_fast
    elif config['proj'] in ['countsketch_scatter', 'srf']:
        # TensorSketch
        rf_dims = rf_dims_fast
    else:
        rf_dims = rf_dims_slow

    for D in rf_dims:
        if config['proj'] == 'srf':
            sketch = Spherical(input_data.shape[1], D,
                                lengthscale=1.0, var=1.0, ard=False,
                                discrete_pdf=False, num_pdf_components=10,
                                complex_weights=config['complex_weights'],
                                projection_type=config['proj'], device=device)
            sketch.load_model('saved_models/poly_a{}.0_p{}_d{}.torch'.format(a, p, d))
            sketch.to(device)
        else:
            sketch = PolynomialSketch(
                input_data.shape[1], D, degree=p, bias=bias, lengthscale=lengthscale, device=device,
                projection_type=config['proj'], full_cov=config['full_cov'], complex_real=config['complex_real'],
                complex_weights=config['complex_weights']
            ).to(device)

        kernel_mse_errors = np.zeros(repetitions)
        kernel_frob_errors = np.zeros(repetitions)
        kernel_abs_errors = np.zeros(repetitions)

        # try:
        torch.cuda.empty_cache()

        # Phase 1: Warm up
        for _ in range(repetitions):
            if config['proj'] == 'srf':
                sketch.resample(num_points_w=5000)
            else:
                sketch.resample()

            _ = sketch.forward(input_data_sample)

        # Phase 2: RF time measurement
        elapsed_time_ms = 0

        for _ in range(repetitions):
            # we do not measure the resampling time now
            if config['proj'] == 'srf':
                sketch.resample(num_points_w=5000)
            else:
                sketch.resample()

            torch.cuda.synchronize()
            start = timer()
            _ = sketch.forward(input_data_sample)

            torch.cuda.synchronize()
            elapsed_time_ms += (timer() - start) * 1000

        # Phase 3: Kernel estimation
        for i in range(repetitions):
            input_data_sample = input_data[torch.randperm(len(input_data), device=device)[:subsample_size]]
            if config['proj'] == 'srf':
                sketch.resample(num_points_w=5000)
            else:
                sketch.resample()
            y = sketch.forward(input_data_sample)
            approx_kernel = y @ y.conj().t()
            exact_kernel = polynomial_kernel(input_data_sample, input_data_sample, degree=p, gamma=1./lengthscale**2, coef0=bias)
            #exact_kernel = exact_kernel.double()
            #approx_kernel.real = approx_kernel.type(torch.complex128)
            kernel_dif = exact_kernel - approx_kernel
            # kernel_dif is now complex
            if config['complex_weights']:
                kernel_mse_errors[i] = (kernel_dif.real.pow(2) + kernel_dif.imag.pow(2)).mean()
                kernel_abs_errors[i] = (kernel_dif.real.pow(2) + kernel_dif.imag.pow(2)).sqrt().mean()
                kernel_frob_errors[i] = ((kernel_dif.real.pow(2) + kernel_dif.imag.pow(2)).sum().sqrt() / exact_kernel.pow(2).sum().sqrt())
            else:
                kernel_mse_errors[i] = kernel_dif.pow(2).mean()
                kernel_abs_errors[i] = kernel_dif.abs().mean()
                kernel_frob_errors[i] = (kernel_dif.pow(2).sum().sqrt() / exact_kernel.pow(2).sum().sqrt())

        log_dir = {
            'method': 'rf',
            'D': D,
            'proj': config['proj'],
            'full_cov': config['full_cov'],
            'complex_weights': config['complex_weights'],
            'complex_real': config['complex_real'],
            'metric_time_ms': elapsed_time_ms / repetitions,
            'mse_mean': kernel_mse_errors.mean(),
            'mse_std': kernel_mse_errors.std(),
            'mae_mean': kernel_abs_errors.mean(),
            'mae_std': kernel_abs_errors.std(),
            'frob_mean': kernel_frob_errors.mean(),
            'frob_std': kernel_frob_errors.std()
        }
        log_handler.append(log_dir)
        csv_handler.append(log_dir)
        csv_handler.save()
        # except Exception as e:
        #     print(e)
        #     print('Skipping current configuration...')
        #     continue