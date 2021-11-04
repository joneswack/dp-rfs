from numpy.core.shape_base import block
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import time

from scipy.special import factorial

from verify_variances import var_gaussian_comp_real, var_gaussian_real, var_rademacher_comp_real, var_rademacher_real, var_tensor_srht_comp_real, var_tensor_srht_real
from random_features.polynomial_sketch import PolynomialSketch

mc_samples = 100

a = 2.
bias = 1.-2./a**2
lengthscale = a / np.sqrt(2.)
#lengthscale = 1.
#bias = 1.

params = {
    'legend.fontsize': 'x-small',
    'figure.figsize': (16, 4), # 2.2*len(csvs)
    'axes.labelsize': 'medium',
    'axes.titlesize':'medium',
    'xtick.labelsize':'medium',
    'ytick.labelsize':'medium',
    'xtick.major.size': 7.0,
    'ytick.major.size': 3.0
}
pylab.rcParams.update(params)

if __name__ == "__main__":

    fig, axs = plt.subplots(1, 4, figsize=(14,4))

    for idx, dataset in enumerate([
        #('EEG', '../datasets/export/eeg/pytorch/eeg.pth'),
        #('Adult', '../datasets/export/adult/pytorch/train_adult.pth'),
        #('Drive', '../datasets/export/drive/pytorch/drive.pth'),
        #('CIFAR10 Conv', '../datasets/export/cifar10/pytorch/train_cifar10_resnet34_final.pth'),
        ('MNIST', '../datasets/export/mnist/pytorch/train_mnist.pth'),
        # ('Fashion MNIST', '../datasets/export/fashion_mnist/pytorch/train_fashion_mnist.pth'),
        # ('Gisette', '../datasets/export/gisette/pytorch/train_gisette.pth')
    ]):

        train_data, train_labels = torch.load(dataset[1])

        train_data = train_data.reshape(len(train_data), -1)

        torch.manual_seed(43)
        np.random.seed(43)

        indices = torch.randint(len(train_data), (1000,))
        train_data = train_data[indices].double()

        train_data = train_data / train_data.norm(dim=1, keepdim=True)

        power_2_pad = int(2**np.ceil(np.log2(train_data.shape[1])))

        placeholder = torch.zeros(len(train_data), power_2_pad)
        placeholder[:, :train_data.shape[1]] = train_data / lengthscale
        placeholder[:, -1] = np.sqrt(bias)
        train_data = placeholder

        ts_vars = []
        gaussian_vars = []
        rad_vars = []
        comp_rad_vars = []
        srht_vars = []
        comp_srht_vars = []

        degree = 5
        #D = int(2.*train_data.shape[1])
        D = 128
        # block_sizes = [1, int(np.sqrt(D))] + list(range(128, D+1, 128))
        #block_sizes = [int(np.sqrt(D))]
        # block_sizes = [4, 8, 12]
        block_sizes = [1, D//2, D] # ,2,3,4,5, int(np.sqrt(D)), D//2, D, 3*D//4, D

        for block_size in block_sizes:
            print('Degree', degree)
            # simulated TensorSketch
            ref_kernel = (train_data @ train_data.t()).pow(degree)

            ts = PolynomialSketch(
                d_in=train_data.shape[1],
                d_features=int(D),
                degree=degree,
                bias=0,
                lengthscale=1.,
                var = 1.,
                ard = False,
                trainable_kernel=False,
                projection_type='osnap_dense',
                hierarchical=False,
                complex_weights=False,
                full_cov=False,
                convolute_ts=False,
                block_size=block_size
            )

            squared_errors = torch.zeros_like(ref_kernel)
            
            for i in range(mc_samples):
                if i % 1 == 0:
                    print('Sample {} / {}'.format(i+1, mc_samples))
                torch.manual_seed(i)
                np.random.seed(i)

                tic = time.time()
                ts.resample()
                # print('Resampling time', time.time() - tic)
                tic = time.time()
                y = ts.forward(train_data)
                # print('Forward time', time.time() - tic)
                # y = torch.cat([y.real, y.imag], dim=1)
                approx_kernel = y @ y.t()
                # approx_kernel *= squared_prefactor.sqrt() * np.sqrt(squared_maclaurin_coefs[degree-1])
                squared_errors += (approx_kernel - ref_kernel).pow(2)

            squared_errors /= mc_samples

            # ts_vars.append(squared_errors.mean())

            # degree_var = var_gaussian_real(train_data, p=degree, D=D)
            # degree_var *= squared_prefactor * squared_maclaurin_coefs[degree-1]
            # gaussian_vars.append(degree_var.view(-1).numpy().mean())

            # degree_var = var_gaussian_comp_real(train_data, p=degree, D=D//2.)

            degree_var = var_rademacher_real(train_data, p=degree, D=D)
            # degree_var *= squared_prefactor * squared_maclaurin_coefs[degree-1]
            # rad_vars.append(degree_var.view(-1).numpy().mean())

            # degree_var = var_rademacher_comp_real(train_data, p=degree, D=D//2.)
            # degree_var *= squared_prefactor * squared_maclaurin_coefs[degree-1]
            # comp_rad_vars.append(degree_var.view(-1).numpy().mean())

            # degree_var, _ = var_tensor_srht_real(train_data, p=degree, D=D, full_cov=False)
            # degree_var *= squared_prefactor * squared_maclaurin_coefs[degree-1]
            # srht_vars.append(degree_var.view(-1).numpy().mean())

            # degree_var, _ = var_tensor_srht_comp_real(train_data, p=degree, D=D, full_cov=False)
            # degree_var *= squared_prefactor * squared_maclaurin_coefs[degree-1]
            # comp_srht_vars.append(degree_var.view(-1).numpy().mean())

            differences = squared_errors / degree_var
            differences = differences[~(differences.isnan() | differences.isinf())]
            differences = differences.view(-1).sort(descending=False)[0]
            n = np.arange(1,len(differences)+1) / np.float(len(differences))
            axs[idx].step(differences, n, label='BS={} (Frob. Ratio: {:.2f})'.format(block_size, squared_errors.sum().abs().sqrt() / degree_var.sum().abs().sqrt()))

            ## TODO: Change to ECDF plot instead of mean variance?

            #plt.plot(np.array(degrees), np.array(gaussian_vars), label='Gaussian') #  / np.sum(gaussian_vars)
            #plt.plot(np.array(degrees), np.array(ts_vars), label='TensorSketch')
            #plt.plot(np.array(degrees), np.array(rad_vars), label='Rademacher') # / np.sum(rad_vars)
            #plt.plot(np.array(degrees), np.array(comp_rad_vars), label='Compl. Rademacher')
            #plt.plot(np.array(degrees), np.array(srht_vars), label='TensorSRHT') #  / np.sum(srht_vars)
            #plt.plot(np.array(degrees), np.array(comp_srht_vars), label='Compl. TensorSRHT')
            #plt.boxplot(gaussian_vars)
            #means = [vars.mean() for vars in gaussian_vars]
            #stds = [vars.std() for vars in gaussian_vars]
            #plt.errorbar(np.array(degrees), means, yerr=stds)
            # plt.plot(np.array(degrees), squared_maclaurin_coefs / squared_maclaurin_coefs.sum(), label='Coefficient decay')
        axs[idx].set_title('{}, d={}'.format(dataset[0], power_2_pad))
        axs[idx].set_xlabel('Variance ratio')
        if idx == 0:
            axs[idx].set_ylabel('ECDF')
        axs[idx].set_xlim(-0.1, 2.0)
        axs[idx].vlines(x=1.0, ymin=0, ymax=1.0, colors='black', label='', linestyles='dashed')
        axs[idx].hlines(y=0.5, xmin=-0.1, xmax=2.0, colors='black', label='', linestyles='dashed')
        axs[idx].legend(ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.3))
    plt.tight_layout()
    plt.savefig('figures/binned_tensorsketch_real_rademacher.pdf', dpi=300, bbox_inches="tight")
    plt.show()