import torch
import torch_fidelity

from timeit import default_timer as timer
from tqdm import tqdm

from torch_fidelity.utils import create_feature_extractor, extract_featuresdict_from_input_id
from torch_fidelity.metric_kid import kid_featuresdict_to_metric

import numpy as np
import matplotlib.pyplot as plt
import util.data

# if int(torch.__version__.split('.')[1]) > 1:
#     # if number after first dot is larger than 1, use the new library
#     from util.fwht.fwht import FastWalshHadamardTransform
# else:
#     from util.fwht_old.fwht import FastWalshHadamardTransform


from util.hadamard_cuda.fwht import FastWalshHadamardTransform
from random_features.polynomial_sketch import PolynomialSketch

device = 'cuda'
repetitions = 100
verbose = True
torch.manual_seed(0)
mmd_est = 'unbiased'
p=3

block_mmd_samples = [100, 1000, 5000, 10000, 15000, 20000]
rf_dims = [64, 128, 256, 512, 1024] + [i*2048 for i in range(1,11)] # 2048*5
# rf_dims = [2048*4]
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

log_handler = util.data.Log_Handler('kid', 'kid_rep{}_mmdest{}_p{}'.format(repetitions, mmd_est, p))
csv_handler = util.data.DF_Handler('kid', 'kid_rep{}_mmdest{}_p{}'.format(repetitions, mmd_est, p))

### Feature extraction time
kwargs = {
    'input1': 'cifar10-val',
    'input2': 'cifar10-train',
    'cuda': device == 'cuda'
}
feat_extractor = create_feature_extractor('inception-v3-compat', ['2048'], **kwargs)

# warm up
_ = extract_featuresdict_from_input_id(1, feat_extractor, **kwargs)

torch.cuda.synchronize()
start = timer()

featuresdict_1 = extract_featuresdict_from_input_id(1, feat_extractor, **kwargs)
featuresdict_2 = extract_featuresdict_from_input_id(2, feat_extractor, **kwargs)

torch.cuda.synchronize()
feature_extraction_time_ms = (timer() - start) * 1000
print('Feature extraction time:', feature_extraction_time_ms)

# torch.save(featuresdict_2['2048'], 'saved_models/conv_features/cifar10-train-inc-v3-2048.pth')
# torch.save(featuresdict_1['2048'], 'saved_models/conv_features/cifar10-val-inc-v3-2048.pth')

# Inception CIFAR-10
# feature_extraction_time_ms = 0
# featuresdict_1 = {'2048': torch.load('../datasets/export/conv_features/cifar10-val-inc-v3-2048.pth')}
# featuresdict_2 = {'2048': torch.load('../datasets/export/conv_features/cifar10-train-inc-v3-2048.pth')}

input_data_1 = featuresdict_1['2048'][:, :-1].to(device)
input_data_2 = featuresdict_2['2048'][:, :-1].to(device)
del featuresdict_1, featuresdict_2

# LeNet MNIST
# input_data_1 = torch.from_numpy(np.load('saved_models/lenet_train.npy'))[:, :-1].to(device)
# input_data_2 = torch.from_numpy(np.load('saved_models/lenet_test.npy'))[:, :-1].to(device)

### Classical MMD implementation ###
def mmd2(K_XX, K_XY, K_YY, unit_diagonal=False, mmd_est='unbiased'):
    assert mmd_est in ('biased', 'unbiased', 'u-statistic')

    m = K_XX.shape[0]
    n = K_YY.shape[1]
    # assert K_XX.shape == (m, m)
    # assert K_XY.shape == (m, m)
    # assert K_YY.shape == (m, m)

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if unit_diagonal:
        diag_X = diag_Y = 1
        sum_diag_X = sum_diag_Y = m
    else:
        diag_X = torch.diagonal(K_XX)
        diag_Y = torch.diagonal(K_YY)

        sum_diag_X = diag_X.sum()
        sum_diag_Y = diag_Y.sum()

    # Kt_sums = sum(K) - sum(diag(K))
    Kt_XX_sums = K_XX.sum(dim=1) - diag_X
    Kt_YY_sums = K_YY.sum(dim=1) - diag_Y
    K_XY_sums_0 = K_XY.sum(dim=0)

    Kt_XX_sum = Kt_XX_sums.sum()
    Kt_YY_sum = Kt_YY_sums.sum()
    K_XY_sum = K_XY_sums_0.sum()

    if mmd_est == 'biased':
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
              + (Kt_YY_sum + sum_diag_Y) / (n * n)
              - 2 * K_XY_sum / (m * n))
    else:
        mmd2 = Kt_XX_sum / (m * (m-1)) + Kt_YY_sum / (n * (n-1))
        if mmd_est == 'unbiased':
            mmd2 -= 2 * K_XY_sum / (m * n)
        else:
            assert m == n
            mmd2 -= 2 * (K_XY_sum - torch.trace(K_XY)) / (m * (m-1))

    return mmd2

def polynomial_kernel(X, Y, degree=3, gamma=None, coef0=1):
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    K = (X @ Y.t() * gamma + coef0) ** degree
    return K

def polynomial_mmd(features_1, features_2, degree, gamma, coef0):
    k_11 = polynomial_kernel(features_1, features_1, degree=degree, gamma=gamma, coef0=coef0)
    k_22 = polynomial_kernel(features_2, features_2, degree=degree, gamma=gamma, coef0=coef0)
    k_12 = polynomial_kernel(features_1, features_2, degree=degree, gamma=gamma, coef0=coef0)
    return mmd2(k_11, k_12, k_22, mmd_est=mmd_est)

def block_mmd_estimate(repetitions, num_samples=10000):
    mmds = np.zeros(repetitions)

    for i in tqdm(
            range(repetitions), disable=not verbose, leave=False, unit='subsets',
            desc='Kernel Inception Distance'
    ):
        # if num_samples > len(input_data), returns input_data
        choice1 = torch.randperm(len(input_data_1), device=device)[:num_samples]
        choice2 = torch.randperm(len(input_data_2), device=device)[:num_samples]
        # it should reuse cached features
        f1 = input_data_1[choice1]
        f2 = input_data_2[choice2]
        o = polynomial_mmd(f1, f2, p, None, 1).item()
        mmds[i] = o

    return mmds.mean(), mmds.std()

results_block_mmd = []

for num_samples in block_mmd_samples:
    # warmups
    try:
        torch.cuda.empty_cache()

        _, _ = block_mmd_estimate(repetitions, num_samples=num_samples)

        torch.cuda.synchronize()
        start = timer()

        mean, std = block_mmd_estimate(repetitions, num_samples=num_samples)

        torch.cuda.synchronize()

        elapsed_time_ms = (timer() - start) * 1000

        print(num_samples, elapsed_time_ms)
        log_dir = {
            'method': 'block-mmd',
            'n': num_samples,
            'metric_time_ms': elapsed_time_ms,
            'feature_time_ms': feature_extraction_time_ms,
            'mean': mean,
            'std': std
        }
        log_handler.append(log_dir)
        csv_handler.append(log_dir)
        csv_handler.save()
    except Exception as e:
        print(e)
        print('Skipping current configuration...')
        continue

### Random Feature MMD

def random_feature_mmd(features_1, features_2, mmd_est='unbiased'):
    # biased mmd2
    # mmd2 = (features_1 - features_2).pow(2).sum()

    n = len(features_1)
    m = len(features_2)

    mean_1 = features_1.mean(dim=0)
    mean_2 = features_2.mean(dim=0)

    if mmd_est == 'biased':
        return (mean_1 - mean_2).pow(2).sum()
    else:
        mmk_unb_1 = n/(n-1) * (mean_1.pow(2).sum() - 1/n**2 * features_1.pow(2).sum(dim=1).sum(dim=0))
        mmk_unb_2 = m/(m-1) * (mean_2.pow(2).sum() - 1/m**2 * features_2.pow(2).sum(dim=1).sum(dim=0))
        mmk_unb_12 = mean_1.dot(mean_2)

        return mmk_unb_1 + mmk_unb_2 - 2*mmk_unb_12

def rf_mmd_estimate(input_data_1, input_data_2, repetitions, D, proj='srht', full_cov=True, complex_real=False):
    mmds = np.zeros(repetitions)
    mmds_explicit = np.zeros(repetitions)
    
    ### Debug
    err_K_xx = np.zeros(repetitions)
    err_K_yy = np.zeros(repetitions)
    err_K_xy = np.zeros(repetitions)
    err_mmd_biased = np.zeros(repetitions)
    err_mmd_unbiased = np.zeros(repetitions)

    sketch = PolynomialSketch(
        input_data_1.shape[1], D, degree=p, bias=1, lengthscale=np.sqrt(2047),
        projection_type=proj, full_cov=full_cov, complex_real=complex_real,
        device=device, convolute_ts=(proj.startswith('countsketch'))
    )

    for i in tqdm(
            range(repetitions), disable=not verbose, leave=False, unit='subsets',
            desc='Kernel Inception Distance'
    ):
        sketch.resample()
        y_1 = sketch.forward(input_data_1).double()
        y_2 = sketch.forward(input_data_2).double()
        o = random_feature_mmd(y_1, y_2, mmd_est=mmd_est).item()
        mmds[i] = o

        ### Debug
        # K_xx = polynomial_kernel(input_data_1, input_data_1, degree=p, gamma=None, coef0=1).double()
        # K_yy = polynomial_kernel(input_data_2, input_data_2, degree=p, gamma=None, coef0=1).double()
        # K_xy = polynomial_kernel(input_data_1, input_data_2, degree=p, gamma=None, coef0=1).double()
        # m = len(K_xx)
        # n = len(K_yy)
        # diag_K_xx = K_xx.diag().diag()
        # diag_K_yy = K_yy.diag().diag()

        # mmd_biased_exact = K_xx/(m * m) + K_yy/(n*n) - 2.*K_xy/(m*n)
        # mmd_unbiased_exact = (K_xx-diag_K_xx)/(m * (m-1)) + (K_yy - diag_K_yy)/(n*(n-1)) - 2.*K_xy/(m*n)
        
        # K_xx_hat = y_1 @ y_1.t()
        # K_yy_hat = y_2 @ y_2.t()
        # K_xy_hat = y_1 @ y_2.t()
        # diag_K_xx_hat = K_xx_hat.diag().diag()
        # diag_K_yy_hat = K_yy_hat.diag().diag()
        # mmd_biased_hat = K_xx_hat/(m*m) + K_yy_hat/(n*n) - 2.*K_xy_hat/(m*n)
        # mmd_unbiased_hat = (K_xx_hat - diag_K_xx_hat)/(m * (m-1)) + (K_yy_hat - diag_K_yy_hat)/(n*(n-1)) - 2.*K_xy_hat/(m*n)
        # mmds_explicit[i] = mmd_unbiased_hat.sum()
        # # collect mses
        # err_K_xx[i] = ((K_xx_hat - K_xx).abs().mean().item()) # abs().
        # err_K_yy[i] = ((K_yy_hat - K_yy).abs().mean().item()) # abs()
        # err_K_xy[i] = ((K_xy_hat - K_xy).abs().mean().item()) # abs()

    print('Proj:', proj, 'D:', D, 'Full Cov:', full_cov, 'Complex_real:', complex_real)
    # print('MMD biased exact:', mmd_biased_exact.sum())
    # print('MMD unbiased exact:', mmd_unbiased_exact.sum())
    # print('mean_err_K_xx:', err_K_xx.mean(), 'std_err_K_xx', err_K_xx.std())
    # print('mean_err_K_yy:', err_K_yy.mean(), 'std_err_K_yy', err_K_yy.std())
    # print('mean_err_K_xy:', err_K_xy.mean(), 'std_err_K_xy', err_K_xy.std())
    #print('mean_err_mmd_biased_hat:', err_mmd_biased.mean(), 'std_err_mmd_hat:', err_mmd_biased.std())
    #print('mean_err_mmd_unbiased_hat:', err_mmd_unbiased.mean(), 'std_err_mmd_hat:', err_mmd_unbiased.std())
    print('mean_mmd:', mmds.mean(), 'std_mmd', mmds.std())
    # print('mean mmd exp.:', mmds_explicit.mean(), 'std_mmd_exp.:', mmds_explicit.std())

    return mmds.mean(), mmds.std()

for config in rf_configs:
    for D in rf_dims:
        try:
            torch.cuda.empty_cache()

            _, _ = rf_mmd_estimate(
                input_data_1,
                input_data_2,
                repetitions,
                D,
                proj=config['proj'],
                full_cov=config['full_cov'],
                complex_real=config['complex_real']
            )

            torch.cuda.synchronize()
            start = timer()

            mean, std = rf_mmd_estimate(
                input_data_1,
                input_data_2,
                repetitions,
                D,
                proj=config['proj'],
                full_cov=config['full_cov'],
                complex_real=config['complex_real']
            )

            torch.cuda.synchronize()

            elapsed_time_ms = (timer() - start) * 1000

            log_dir = {
                'method': 'rf',
                'D': D,
                'proj': config['proj'],
                'full_cov': config['full_cov'],
                'complex_real': config['complex_real'],
                'metric_time_ms': elapsed_time_ms,
                'feature_time_ms': feature_extraction_time_ms,
                'mean': mean,
                'std': std
            }
            log_handler.append(log_dir)
            csv_handler.append(log_dir)
            csv_handler.save()
        except Exception as e:
            print(e)
            print('Skipping current configuration...')
            continue


### Measure projection time

# input_data = featuresdict_1['2048'][:, :-1].to(device)

# d = input_data.shape[1]
# for D in [2*2048, 4*2048, 6*2048, 8*2048]:
#     sketch = TensorSketch(
#         d, D, degree=3, bias=1, lengthscale=np.sqrt(2047),
#         projection='srht', full_cov=True, complex_real=False,
#         device=device
#     )

#     # sketch = TensorSketch(
#     #     d, D, degree=3, bias=1, lengthscale=np.sqrt(d), complex_real=False,
#     #     projection='countsketch_scatter', full_cov=False, device=device
#     # )
#     # sketch.resample()
#     tic = time.time()

#     mses_dim = []

#     for seed in range(100):
#         torch.manual_seed(seed)
#         np.random.seed(seed)
#         sketch.resample()
#         # sketch.to(device)
#         y = sketch.forward(input_data[:1000])

#         kernel = (input_data[:1000] @ input_data[:1000].t() / 2047 + 1)**3
#         # kernel = (input_data[:1000] @ input_data[:1000].t() / 2048 + 1)**3
#         mse = (y @ y.t() - kernel).pow(2).mean().item()
#         mses_dim.append(mse)

#     print('Dim:', D, 'Time:', time.time() - tic)
#     print('MSE:', np.array(mses_dim).mean())


### End projection measurement


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