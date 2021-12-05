import torch
import torch_fidelity

import time

from torch_fidelity.utils import create_feature_extractor, extract_featuresdict_from_input_id_cached
from torch_fidelity.metric_kid import kid_featuresdict_to_metric

import numpy as np
import matplotlib.pyplot as plt

from util.hadamard_cuda.fwht import FastWalshHadamardTransform

# feat_extractor = create_feature_extractor('inception-v3-compat', ['2048'], **kwargs)
# featuresdict_1 = extract_featuresdict_from_input_id_cached(1, feat_extractor, **kwargs)
# torch.save(featuresdict_1['2048'], 'saved_models/conv_features/cifar10-val-inc-v3-2048.pth')
# featuresdict_2 = extract_featuresdict_from_input_id_cached(2, feat_extractor, **kwargs)
# torch.save(featuresdict_2['2048'], 'saved_models/conv_features/cifar10-train-inc-v3-2048.pth')

featuresdict_1 = {'2048': torch.load('saved_models/conv_features/cifar10-val-inc-v3-2048.pth')}
featuresdict_2 = {'2048': torch.load('saved_models/conv_features/cifar10-train-inc-v3-2048.pth')}

### Measure projection time
from random_features.polynomial_sketch import PolynomialSketch

device = 'cpu'

input_data = featuresdict_1['2048'][:, :-1].to(device)
# input_data = torch.zeros(len(featuresdict_1['2048']), 4095, device=device)
# input_data[:,:2048] = featuresdict_1['2048'].to(device)

d = input_data.shape[1]
for D in [2*2048, 4*2048, 6*2048, 8*2048]: # , 4*2048, 6*2048, 8*2048
    sketch = PolynomialSketch(
        d, D, degree=3, bias=1, lengthscale=np.sqrt(2047),
        projection_type='srht', full_cov=True, complex_real=False,
        device=device
    )

    # sketch = PolynomialSketch(
    #     d, D, degree=3, bias=1, lengthscale=np.sqrt(d), complex_real=False, convolute_ts=True,
    #     projection_type='countsketch_sparse', full_cov=False, device=device
    # )
    # sketch.resample()
    tic = time.time()

    mses_dim = []

    for seed in range(100):
        torch.manual_seed(seed)
        np.random.seed(seed)
        sketch.resample()
        # sketch.to(device)
        y = sketch.forward(input_data[:1000])

        kernel = (input_data[:1000] @ input_data[:1000].t() / 2047 + 1)**3
        # kernel = (input_data[:1000] @ input_data[:1000].t() / 2048 + 1)**3
        mse = (y @ y.t() - kernel).pow(2).mean().item()
        mses_dim.append(mse)

    print('Dim:', D, 'Time:', time.time() - tic)
    print('MSE:', np.array(mses_dim).mean())


### End projection measurement

# mmds = []
# subsets_sizes = [50, 100, 200, 300, 400, 500, 750, 1000, 1500, 2000]

# for subset_size in subsets_sizes:

#     kwargs = {
#         #'input1': torch_fidelity.GenerativeModelModuleWrapper(G, 128, 'normal', 0),
#         #'input1_model_num_samples': 64,
#         #'input2': torch_fidelity.GenerativeModelModuleWrapper(G, 128, 'normal', 0),
#         #'input2_model_num_samples': 64,
#         'kid_subset_size': subset_size,
#         'kid_subsets': 100,
#         'input1': 'cifar10-val',
#         'input2': 'cifar10-train',
#         'cuda': True
#     }

#     metric_kid = kid_featuresdict_to_metric(featuresdict_1, featuresdict_2, '2048', **kwargs)
#     print(metric_kid['kernel_inception_distance_mean'])
#     mmds.append(metric_kid['vals'])

# plt.errorbar(subsets_sizes, [mmd.mean() for mmd in mmds], [mmd.std() for mmd in mmds])
# plt.show()


### FWHT vs FFT comparison
# input_dims = [2**i for i in range(1,16)]
# n_seeds = 1000
# fwht_times = []
# fft_times = []
# device = 'cuda'

# projs = [torch.ones(d,d) for d in input_dims]

# with torch.no_grad():

#     for i, d in enumerate(input_dims):
#         tic = time.time()

#         for seed in range(n_seeds):
#             torch.manual_seed(seed)
#             x = torch.randn(1000, d, device=device)
            
#             FastWalshHadamardTransform.apply(x)

#         fwht_times.append(time.time() - tic)

#     print('FWHT', fwht_times)

#     for i, d in enumerate(input_dims):

#         tic = time.time()

#         for seed in range(n_seeds):
#             torch.manual_seed(seed)
#             x = torch.randn(1000, d, device=device)
#             torch.rfft(x, signal_ndim=1)
#             # x @ projs[i]
#         fft_times.append(time.time() - tic)

# print('FFT', fft_times)
# print('Ratio', np.array(fwht_times) / np.array(fft_times))
### FWHT vs FFT comparison end

## translate subset sizes to random feature budget
# for both, we are taking 100 trials
# mmd2 m = subset_size
# cost of unbiased mmd2 = 2m(m-1) + 2m^2 = 4m^2 - 2m kernel evals => approx variable m out of N
# cost of polynomial sketch = N D log_2(d) => try wall-clock! if it doesn't work, go for approx OPs!

# returns {KEY_METRIC_KID_MEAN: float(np.mean(mmds)),KEY_METRIC_KID_STD: float(np.std(mmds)),}

