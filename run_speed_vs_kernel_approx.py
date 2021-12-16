import torch
from random_features.polynomial_sketch import PolynomialSketch
import util.data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from timeit import default_timer as timer
from tqdm import tqdm

device = 'cpu'
repetitions = 100
torch.manual_seed(0)

# featuresdict_1 = {'2048': torch.load('saved_models/conv_features/cifar10-val-inc-v3-2048.pth')}
featuresdict_2 = {'2048': torch.load('saved_models/conv_features/cifar10-train-inc-v3-2048.pth')}
# input_data_1 = featuresdict_1['2048'][:, :-1].to(device)
input_data = featuresdict_2['2048'][:, :-1].to(device)
del featuresdict_2

# input_data_1 = torch.from_numpy(np.load('saved_models/lenet_test.npy'))[:, :-1].to(device)
# input_data = torch.from_numpy(np.load('saved_models/conv_features/lenet_train.npy'))[:, :-1].to(device)

# polynomial kernel parameters
p=3
bias=1
lengthscale = 1./input_data.shape[1]

# size of the random subset of the input data
subsample_size = 1000
# this data sample will be recomputed for every repetition later on
input_data_sample = input_data[torch.randperm(len(input_data), device=device)[:subsample_size]]
rf_dims = [64, 128, 256, 512, 1024, 2048, 2048*2, 2048*3, 2048*4, 2048*5, 2048*6]
# rf_dims = [512]
# rf_dims = [512*2]
rf_configs = [
    {'proj': 'countsketch_scatter', 'full_cov': False, 'complex_real': False},
    # {'proj': 'gaussian', 'full_cov': False, 'complex_real': False},
    # {'proj': 'gaussian', 'full_cov': False, 'complex_real': True},
    # {'proj': 'rademacher', 'full_cov': False, 'complex_real': False},
    # {'proj': 'rademacher', 'full_cov': False, 'complex_real': True},
    # {'proj': 'srht', 'full_cov': False, 'complex_real': False},
    # {'proj': 'srht', 'full_cov': False, 'complex_real': True},
    {'proj': 'srht', 'full_cov': True, 'complex_real': False},
    {'proj': 'srht', 'full_cov': True, 'complex_real': True}
]

log_handler = util.data.Log_Handler('time_benchmark', 'rep{}_p{}_bias{}_cifarinc'.format(repetitions, p, bias))
csv_handler = util.data.DF_Handler('time_benchmark', 'rep{}_p{}_bias{}_cifarinc'.format(repetitions, p, bias))

def polynomial_kernel(X, Y, degree=3, gamma=None, coef0=1):
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    K = (X @ Y.t() * gamma + coef0) ** degree
    return K

for config in rf_configs:
    for D in rf_dims:
        sketch = PolynomialSketch(
            input_data.shape[1], D, degree=p, bias=bias, lengthscale=np.sqrt(input_data.shape[1]),
            projection_type=config['proj'], full_cov=config['full_cov'], complex_real=config['complex_real'],
            convolute_ts=(config['proj'].startswith('countsketch')), device=device
        )

        kernel_mse_errors = np.zeros(repetitions)
        kernel_frob_errors = np.zeros(repetitions)
        kernel_abs_errors = np.zeros(repetitions)

        try:
            # torch.cuda.empty_cache()

            # Phase 1: Warm up
            for _ in range(repetitions):
                sketch.resample()
                _ = sketch.forward(input_data_sample)

            # Phase 2: RF time measurement
            elapsed_time_ms = 0

            for _ in range(repetitions):
                # we do not measure the resampling time now
                sketch.resample()

                # torch.cuda.synchronize()
                start = timer()
                _ = sketch.forward(input_data_sample)

                # torch.cuda.synchronize()
                elapsed_time_ms += (timer() - start) * 1000

            # Phase 3: Kernel estimation
            for i in range(repetitions):
                input_data_sample = input_data[torch.randperm(len(input_data), device=device)[:subsample_size]]
                sketch.resample()
                y = sketch.forward(input_data_sample)
                approx_kernel = y @ y.t()
                exact_kernel = polynomial_kernel(input_data_sample, input_data_sample, degree=p, gamma=1./input_data.shape[1], coef0=bias)
                kernel_dif = exact_kernel - approx_kernel
                kernel_mse_errors[i] = kernel_dif.pow(2).mean()
                kernel_abs_errors[i] = kernel_dif.abs().mean()
                kernel_frob_errors[i] = (kernel_dif.pow(2).sum().sqrt() / exact_kernel.pow(2).sum().sqrt())

            log_dir = {
                'method': 'rf',
                'D': D,
                'proj': config['proj'],
                'full_cov': config['full_cov'],
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
        except Exception as e:
            print(e)
            print('Skipping current configuration...')
            continue