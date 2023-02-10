import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

from scipy.special import factorial

import pickle

from verify_variances import var_gaussian_real, var_rademacher_comp_real, var_rademacher_real, var_tensor_srht_comp_real, var_tensor_srht_real
from random_features.polynomial_sketch import PolynomialSketch

def gaussian_kernel_coefs(n):
    return 1./factorial(n)

mc_samples = 1000

a = 2.
bias = 1.-2./a**2
lengthscale = a / np.sqrt(2.)
# lengthscale = 1.
# bias = 1.

if __name__ == "__main__":

    fig, axs = plt.subplots(1, 4, figsize=(14,4))

    results = []

    for idx, dataset in enumerate([
        # ('EEG', '../datasets/export/eeg/pytorch/eeg.pth'),
        #('Adult', '../datasets/export/adult/pytorch/train_adult.pth'),
        # ('Drive', '../datasets/export/drive/pytorch/drive.pth'),
        ('Letter', '../datasets/export/letter/pytorch/letter.pth'),
        ('Mocap', '../datasets/export/mocap/pytorch/mocap.pth'),
        # ('Covertype', '../datasets/export/covtype/pytorch/covtype.pth'),
        (
            'CIFAR10 Conv',
            '../datasets/export/cifar10/pytorch/resnet34_final_conv_train.pth',
            '../datasets/export/cifar10/pytorch/resnet34_final_conv_test.pth'
        ),
        ('MNIST', '../datasets/export/mnist/pytorch/train_mnist.pth'),
        # ('CIFAR10 Conv', '../datasets/export/conv_features/cifar10-train-inc-v3-2048.pth')
        # ('Fashion MNIST', '../datasets/export/fashion_mnist/pytorch/train_fashion_mnist.pth'),
        # ('Gisette', '../datasets/export/gisette/pytorch/train_gisette.pth')
    ]):

        if len(dataset) == 3:
            train_data, train_labels = torch.load(dataset[1])
            test_data, train_labels = torch.load(dataset[2])
        if len(dataset) == 2:
            train_data, train_labels = torch.load(dataset[1])
            #test_data, test_labels = test_data

        train_data = train_data.reshape(len(train_data), -1)
        # test_data = test_data.reshape(len(test_data), -1)

        train_data = train_data - train_data.mean(dim=0)

        torch.manual_seed(0) # 42
        np.random.seed(0) # 42

        # if dataset[0] == 'MNIST':
        # min/max-scaling

        # subtract min val per feature
        # min_val = torch.min(train_data, 0)[0]
        # val_range = torch.max(train_data, 0)[0] - min_val
        # val_range[val_range == 0] = 1
        # train_data = (train_data - min_val) #/ val_range
        
        #test_data = (test_data - min_val) #/ val_range

        indices = torch.randint(len(train_data), (1000,))
        #data = test_data[indices].float()
        train_data = train_data[indices].float()

        # lengthscale = torch.cdist(train_data, train_data, p=2.).median()
        # lengthscale = np.sqrt(data.shape[1])

        # squared_prefactor_train = torch.exp(-train_data.pow(2).sum(dim=1))

        train_data = train_data / train_data.norm(dim=1, keepdim=True)

        data = train_data

        power_2_pad = int(2**np.ceil(np.log2(data.shape[1])))

        placeholder = torch.zeros(len(data), power_2_pad)
        placeholder[:, :data.shape[1]] = data / lengthscale
        placeholder[:, -1] = np.sqrt(bias)

        ts_vars = []
        gaussian_vars = []
        rad_vars = []
        comp_rad_vars = []
        srht_vars = []
        comp_srht_vars = []

        degrees = list(range(1, 11, 1))
        # degrees = [1, 2, 3]

        #squared_prefactor = squared_prefactor_train.unsqueeze(1) * squared_prefactor_train.unsqueeze(0)
        squared_maclaurin_coefs = gaussian_kernel_coefs(np.array(degrees))**2

        D = int(2.*placeholder.shape[1])

        dataset_results = []

        for degree in degrees:
            # D = train_data.shape[1]**degree
            print('Degree', degree)
            # simulated TensorSketch
            ref_kernel = (placeholder @ placeholder.t()).pow(degree)
            # ref_kernel *= squared_prefactor.sqrt() * np.sqrt(squared_maclaurin_coefs[degree-1])

            ts = PolynomialSketch(
                d_in=placeholder.shape[1],
                d_features=int(D),
                degree=degree,
                bias=0,
                lengthscale=1.,
                var = 1.,
                ard = False,
                trainable_kernel=False,
                projection_type='countsketch_sparse',
                hierarchical=False,
                complex_weights=False,
                full_cov=False
            )

            # squared_errors = torch.zeros_like(ref_kernel)
            
            # for i in range(mc_samples):
            #     if i % 100 == 0:
            #         print('Sample {} / {}'.format(i+1, mc_samples))
            #     torch.manual_seed(i)
            #     np.random.seed(i)

            #     ts.resample()
            #     y = ts.forward(placeholder)
            #     # y = torch.cat([y.real, y.imag], dim=1)
            #     approx_kernel = y @ y.t()
            #     # approx_kernel *= squared_prefactor.sqrt() * np.sqrt(squared_maclaurin_coefs[degree-1])
            #     squared_errors += (approx_kernel - ref_kernel).pow(2)

            # squared_errors /= mc_samples

            # ts_vars.append(squared_errors.mean())

            # degree_var = var_gaussian_real(train_data, p=degree, D=D)
            # degree_var *= squared_prefactor * squared_maclaurin_coefs[degree-1]
            # gaussian_vars.append(degree_var.view(-1).numpy().mean())

            degree_var1 = var_rademacher_real(train_data, p=degree, D=D) # degree_var
            # degree_var *= squared_prefactor * squared_maclaurin_coefs[degree-1]
            # rad_vars.append(degree_var1.view(-1).numpy().mean())

            degree_var2 = var_rademacher_comp_real(train_data, p=degree, D=D//2.)
            # degree_var *= squared_prefactor * squared_maclaurin_coefs[degree-1]
            # comp_rad_vars.append(degree_var2.view(-1).numpy().mean())

            # degree_var1, _ = var_tensor_srht_real(placeholder, p=degree, D=D, full_cov=True)
            # degree_var *= squared_prefactor * squared_maclaurin_coefs[degree-1]
            # srht_vars.append(degree_var.view(-1).numpy().mean())

            # degree_var2, _ = var_tensor_srht_comp_real(placeholder, p=degree, D=D//2., full_cov=True)
            # degree_var *= squared_prefactor * squared_maclaurin_coefs[degree-1]
            # comp_srht_vars.append(degree_var.view(-1).numpy().mean())

            differences = degree_var2 / degree_var1 # squared_errors
            differences = differences[~(differences.isnan() | differences.isinf())]
            differences = differences.view(-1).sort(descending=False)[0]

            dataset_results.append((degree, differences))

            n = np.arange(1,len(differences)+1) / np.float(len(differences))

        results.append((dataset, power_2_pad, dataset_results))
    
    with open('saved_models/ecdf_plots/rademacher_ctr_r_norm_zero.pkl', 'wb') as handle:
        pickle.dump(results, handle)
