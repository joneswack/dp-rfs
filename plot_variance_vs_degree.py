import torch
import numpy as np
import matplotlib.pyplot as plt

from scipy.special import factorial

from verify_variances import var_gaussian_real, var_rademacher_comp_real, var_rademacher_real, var_tensor_srht_comp_real, var_tensor_srht_real
from random_features.polynomial_sketch import PolynomialSketch

def gaussian_kernel_coefs(n):
    return 1./factorial(n)

mc_samples = 100

a = 4.
bias = 1.-2./a**2
lengthscale = a / np.sqrt(2.)
#lengthscale = 1.
#bias = 0

if __name__ == "__main__":

    for idx, dataset in enumerate([
        #('EEG', '../datasets/export/boston/pytorch/boston.pth'),
        # ('CIFAR10 Conv', '../datasets/export/cifar10/pytorch/train_cifar10_resnet34_final.pth'),
        ('MNIST', '../datasets/export/mnist/pytorch/train_mnist.pth'),
        ('Fashion MNIST', '../datasets/export/fashion_mnist/pytorch/train_fashion_mnist.pth'),
        ('Gisette', '../datasets/export/gisette/pytorch/train_gisette.pth')
    ]):

        train_data, train_labels = torch.load(dataset[1])

        train_data = train_data.reshape(len(train_data), -1)

        if dataset[0] == 'MNIST':
            train_data += train_data.min().abs()

        # train_data = train_data - train_data.mean(dim=0)

        indices = torch.randint(len(train_data), (1000,))
        train_data = train_data[indices] #.double()

        # lengthscale = torch.cdist(train_data, train_data, p=2.).median()
        # train_data = train_data / lengthscale

        squared_prefactor_train = torch.exp(-train_data.pow(2).sum(dim=1))

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

        degrees = list(range(1, 11))

        squared_prefactor = squared_prefactor_train.unsqueeze(1) * squared_prefactor_train.unsqueeze(0)
        squared_maclaurin_coefs = gaussian_kernel_coefs(np.array(degrees))**2

        D = int(2.*train_data.shape[1])

        for degree in degrees:
            # simulated TensorSketch
            ref_kernel = (train_data @ train_data.t()).pow(degree)
            # ref_kernel *= squared_prefactor.sqrt() * np.sqrt(squared_maclaurin_coefs[degree-1])

            ts = PolynomialSketch(
                d_in=train_data.shape[1],
                d_features=D,
                degree=degree,
                bias=0,
                lengthscale=1.,
                var = 1.,
                ard = False,
                trainable_kernel=False,
                projection_type='countsketch_scatter',
                hierarchical=False,
                complex_weights=False,
                full_cov=False
            )

            squared_errors = torch.zeros_like(ref_kernel)
            
            for _ in range(mc_samples):
                ts.resample()
                y = ts.forward(train_data)
                #y = torch.cat([y.real, y.imag], dim=1)
                approx_kernel = y @ y.t()
                # approx_kernel *= squared_prefactor.sqrt() * np.sqrt(squared_maclaurin_coefs[degree-1])
                squared_errors += (approx_kernel - ref_kernel).pow(2)

            squared_errors /= mc_samples

            # ts_vars.append(squared_errors.mean())

            # degree_var = var_gaussian_real(train_data, p=degree, D=D)
            # degree_var *= squared_prefactor * squared_maclaurin_coefs[degree-1]
            # gaussian_vars.append(degree_var.view(-1).numpy().mean())

            # degree_var = var_rademacher_real(train_data, p=degree, D=D)
            # degree_var *= squared_prefactor * squared_maclaurin_coefs[degree-1]
            # rad_vars.append(degree_var.view(-1).numpy().mean())

            # degree_var = var_rademacher_comp_real(train_data, p=degree, D=D//2.)
            # degree_var *= squared_prefactor * squared_maclaurin_coefs[degree-1]
            # comp_rad_vars.append(degree_var.view(-1).numpy().mean())

            # degree_var, _ = var_tensor_srht_real(train_data, p=degree, D=D)
            # degree_var *= squared_prefactor * squared_maclaurin_coefs[degree-1]
            # srht_vars.append(degree_var.view(-1).numpy().mean())

            degree_var, _ = var_tensor_srht_comp_real(train_data, p=degree, D=D//2., full_cov=True)
            # degree_var *= squared_prefactor * squared_maclaurin_coefs[degree-1]
            # comp_srht_vars.append(degree_var.view(-1).numpy().mean())

            differences = degree_var / squared_errors
            differences = differences[~(differences.isnan() | differences.isinf())]
            differences = differences.view(-1).sort(descending=False)[0]
            n = np.arange(1,len(differences)+1) / np.float(len(differences))
            plt.step(differences, n, label='p={}'.format(degree))

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
        plt.xlim(-0.1, 2.0)
        plt.vlines(x=1.0, ymin=0, ymax=1.0, colors='black', label='', linestyles='dashed')
        plt.hlines(y=0.5, xmin=-0.1, xmax=2.0, colors='black', label='', linestyles='dashed')
        plt.legend()
        plt.show()