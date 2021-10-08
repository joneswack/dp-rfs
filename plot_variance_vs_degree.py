import torch
import numpy as np
import matplotlib.pyplot as plt

from scipy.special import factorial

from verify_variances import var_gaussian_real, var_rademacher_comp_real, var_rademacher_real, var_tensor_srht_comp_real, var_tensor_srht_real

def gaussian_kernel_coefs(n):
    return 1./factorial(n)

if __name__ == "__main__":

    for idx, dataset in enumerate([
        ('EEG', '../datasets/export/boston/pytorch/boston.pth'),
        ('CIFAR10 Conv', '../datasets/export/cifar10/pytorch/train_cifar10_resnet34_final.pth'),
        ('MNIST', '../datasets/export/mnist/pytorch/train_mnist.pth'),
        ('Fashion MNIST', '../datasets/export/fashion_mnist/pytorch/train_fashion_mnist.pth'),
        ('Gisette', '../datasets/export/gisette/pytorch/train_gisette.pth')
    ]):

        train_data, train_labels = torch.load(dataset[1])

        train_data = train_data.reshape(len(train_data), -1)

        # train_data = train_data - train_data.mean(dim=0)

        indices = torch.randint(len(train_data), (1000,))
        train_data = train_data[indices].double()

        lengthscale = torch.cdist(train_data, train_data, p=2.).median()
        train_data = train_data / lengthscale

        squared_prefactor_train = torch.exp(-train_data.pow(2).sum(dim=1))

        # train_data = train_data / train_data.norm(dim=1, keepdim=True)

        power_2_pad = int(2**np.ceil(np.log2(train_data.shape[1])))

        placeholder = torch.zeros(len(train_data), power_2_pad)
        placeholder[:, :train_data.shape[1]] = train_data
        train_data = placeholder

        gaussian_vars = []
        rad_vars = []
        comp_rad_vars = []
        srht_vars = []
        comp_srht_vars = []

        degrees = list(range(1, 11))

        squared_prefactor = squared_prefactor_train.unsqueeze(1) * squared_prefactor_train.unsqueeze(0)
        squared_maclaurin_coefs = gaussian_kernel_coefs(np.array(degrees))**2

        D = train_data.shape[1]//2.

        for degree in degrees:
            degree_var = var_gaussian_real(train_data, p=degree, D=D)
            degree_var *= squared_prefactor * squared_maclaurin_coefs[degree-1]
            gaussian_vars.append(degree_var.view(-1).numpy().mean())

            degree_var = var_rademacher_real(train_data, p=degree, D=D)
            degree_var *= squared_prefactor * squared_maclaurin_coefs[degree-1]
            rad_vars.append(degree_var.view(-1).numpy().mean())

            degree_var = var_rademacher_comp_real(train_data, p=degree, D=D//2.)
            degree_var *= squared_prefactor * squared_maclaurin_coefs[degree-1]
            comp_rad_vars.append(degree_var.view(-1).numpy().mean())

            degree_var, _ = var_tensor_srht_real(train_data, p=degree, D=D)
            degree_var *= squared_prefactor * squared_maclaurin_coefs[degree-1]
            srht_vars.append(degree_var.view(-1).numpy().mean())

            degree_var, _ = var_tensor_srht_comp_real(train_data, p=degree, D=D//2.)
            degree_var *= squared_prefactor * squared_maclaurin_coefs[degree-1]
            comp_srht_vars.append(degree_var.view(-1).numpy().mean())

        ## TODO: Change to ECDF plot instead of mean variance?

        #plt.plot(np.array(degrees), np.array(gaussian_vars), label='Gaussian') #  / np.sum(gaussian_vars)
        plt.plot(np.array(degrees), np.array(rad_vars), label='Rademacher') # / np.sum(rad_vars)
        plt.plot(np.array(degrees), np.array(comp_rad_vars), label='Compl. Rademacher')
        plt.plot(np.array(degrees), np.array(srht_vars), label='TensorSRHT') #  / np.sum(srht_vars)
        plt.plot(np.array(degrees), np.array(comp_srht_vars), label='Compl. TensorSRHT')
        #plt.boxplot(gaussian_vars)
        #means = [vars.mean() for vars in gaussian_vars]
        #stds = [vars.std() for vars in gaussian_vars]
        #plt.errorbar(np.array(degrees), means, yerr=stds)
        # plt.plot(np.array(degrees), squared_maclaurin_coefs / squared_maclaurin_coefs.sum(), label='Coefficient decay')
        plt.legend()
        plt.show()