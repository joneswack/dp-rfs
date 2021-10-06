from numpy.core.numeric import full
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
#from torch._C import DoubleTensor

from random_features.polynomial_sketch import PolynomialSketch

def get_pairwise_elements(X):
    # sum(x_i y_i)
    dot_product_squared = (X @ X.t())**2
    # sum(x_i^2 y_i^2)
    squared_dot_product = X.pow(2) @ X.pow(2).t()
    # ||x||^2 ||y||^2
    norms_squared = (X**2).sum(dim=1).unsqueeze(1) * (X**2).sum(dim=1).unsqueeze(0)

    return (dot_product_squared, squared_dot_product, norms_squared)

def var_gaussian_real(X, p=1., D=1.):
    dot_product_squared, _, norms_squared = get_pairwise_elements(X)
    moment_2 = (norms_squared + 2. * dot_product_squared)**p
    
    return (moment_2 - dot_product_squared**p) / D

def var_gaussian_comp_real(X, p=1., D=1.):
    dot_product_squared, _, norms_squared = get_pairwise_elements(X)
    moment_2_comp = (norms_squared + dot_product_squared)**p
    moment_2_real = (2. * dot_product_squared)**p

    return (0.5 * (moment_2_comp + moment_2_real) - dot_product_squared**p) / D

def var_rademacher_real(X, p=1., D=1.):
    dot_product_squared, squared_dot_product, norms_squared = get_pairwise_elements(X)
    moment_2 = (norms_squared + 2 * (dot_product_squared - squared_dot_product))**p
    
    return (moment_2 - dot_product_squared**p) / D

def var_rademacher_comp(X, p=1., D=1.):
    dot_product_squared, squared_dot_product, norms_squared = get_pairwise_elements(X)
    moment_2 = (norms_squared + (dot_product_squared - squared_dot_product))**p

    return (moment_2 - dot_product_squared**p) / D

def var_rademacher_comp_real(X, p=1., D=1.):
    dot_product_squared, squared_dot_product, norms_squared = get_pairwise_elements(X)
    moment_2_comp = (norms_squared + dot_product_squared - squared_dot_product)**p
    moment_2_real = (2. * dot_product_squared - squared_dot_product)**p

    return (0.5 * (moment_2_comp + moment_2_real) - dot_product_squared**p) / D

def var_tensor_srht_real(X, p=1., D=1.):
    dot_product_squared, squared_dot_product, norms_squared = get_pairwise_elements(X)
    d = X.shape[1]
    num_non_zero_pairs = np.floor(D / d)*d*(d-1) + (D % d) * (D % d - 1)

    var_term = var_rademacher_real(X, p, D)
    cov_term = (num_non_zero_pairs / D**2) * \
        ((dot_product_squared - 1./(d-1.) * (norms_squared + dot_product_squared - 2. * squared_dot_product))**p - dot_product_squared**p)

    vars = var_term + cov_term
    zero_vars = (dot_product_squared == 1) & (squared_dot_product == 1) & (norms_squared == 1)
    vars[zero_vars] = 0

    return vars, cov_term

def var_tensor_srht_comp(X, p=1., D=1.):
    dot_product_squared, squared_dot_product, norms_squared = get_pairwise_elements(X)
    d = X.shape[1]
    num_non_zero_pairs = np.floor(D / d)*d*(d-1) + (D % d) * (D % d - 1)

    var_term = var_rademacher_comp(X, p, D)
    cov_term = (num_non_zero_pairs / D**2) * \
        ((dot_product_squared - 1./(d-1.) * (norms_squared - squared_dot_product))**p - dot_product_squared**p)

    vars = var_term + cov_term
    zero_vars = (dot_product_squared == 1) & (squared_dot_product == 1) & (norms_squared == 1)
    vars[zero_vars] = 0

    return vars, cov_term

def var_tensor_srht_comp_real(X, p=1., D=1., full_cov=False):
    dot_product_squared, squared_dot_product, norms_squared = get_pairwise_elements(X)
    d = X.shape[1]

    if full_cov:
        num_non_zero_pairs = D*(D-1.)
    else:
        num_non_zero_pairs = np.floor(D / d)*d*(d-1) + (D % d) * (D % d - 1)

    if full_cov:
        block_coef = 1./(np.ceil(D / d)*d-1.)
    else:
        block_coef = 1./(d-1.)

    var_term = var_rademacher_comp_real(X, p, D)
    cov_term = num_non_zero_pairs / D**2 * \
        ((dot_product_squared - block_coef * (dot_product_squared - squared_dot_product))**p - dot_product_squared**p)
    cov_term += num_non_zero_pairs / D**2 * \
        ((dot_product_squared - block_coef * (norms_squared - squared_dot_product))**p - dot_product_squared**p)
    cov_term *= 0.5

    return var_term + cov_term, cov_term




if __name__ == "__main__":
    params = {
        'legend.fontsize': 'medium',
        'figure.figsize': (16, 4), # 2.2*len(csvs)
        'axes.labelsize': 'medium',
        'axes.titlesize':'medium',
        'xtick.labelsize':'medium',
        'ytick.labelsize':'medium',
        'xtick.major.size': 7.0,
        'ytick.major.size': 3.0
    }
    pylab.rcParams.update(params)

    a = 4.
    bias = 1.-2./a**2
    lengthscale = a / np.sqrt(2.)

    #var_function = lambda X, p, D: var_tensor_srht_comp_real(X, p, D)
    #real_var_function = lambda X, p, D: var_tensor_srht_real(X, p, D)
    #comp_var_function = lambda X, p, D: var_tensor_srht_comp(X, p, D)

    real_var_function = lambda X, p, D: var_rademacher_comp_real(X, p, 10.*D)
    comp_var_function = lambda X, p, D: var_tensor_srht_comp_real(X, p, 10.*D, full_cov=False)

    fig, axs = plt.subplots(1, 4, figsize=(12,3))

    #train_data = torch.rand(10000, 128, dtype=torch.float64)
    # train_data = torch.tensor([[0, 1], [1, 0]], dtype=torch.float64)
    # train_data = train_data / train_data.norm(dim=1, keepdim=True)

    #train_data, train_labels = torch.load('../datasets/export/fashion_mnist/pytorch/train_fashion_mnist.pth')
    # test_data, test_labels = torch.load('../datasets/export/fashion_mnist/pytorch/test_fashion_mnist.pth')

    #train_data, train_labels = torch.load('../datasets/export/mnist/pytorch/train_mnist.pth')
    # test_data, test_labels = torch.load('../datasets/export/mnist/pytorch/test_mnist.pth')

    #train_data, train_labels = torch.load('../datasets/export/adult/pytorch/train_adult.pth')
    # test_data, test_labels = torch.load('../datasets/export/adult/pytorch/test_adult.pth')


    for idx, dataset in enumerate([
        ('EEG', '../datasets/export/eeg/pytorch/eeg.pth'),
        ('CIFAR10 Conv', '../datasets/export/cifar10/pytorch/train_cifar10_resnet34_final.pth'),
        ('MNIST', '../datasets/export/mnist/pytorch/train_mnist.pth'),
        ('Gisette', '../datasets/export/gisette/pytorch/train_gisette.pth')
    ]):

        train_data, train_labels = torch.load(dataset[1])

        train_data = train_data.reshape(len(train_data), -1)
        #train_data = train_data / train_data.shape[1]

        train_data = train_data - train_data.mean(dim=0)

        # train_data = train_data / train_data.norm(dim=1, keepdim=True)

        indices = torch.randint(len(train_data), (1000,))
        train_data = train_data[indices].double()

        lengthscale = torch.cdist(train_data, train_data, p=2.).median()
        # train_data = train_data / lengthscale

        # train_data = train_data / train_data.norm(dim=1, keepdim=True)

        power_2_pad = int(2**np.ceil(np.log2(train_data.shape[1])))

        placeholder = torch.zeros(len(train_data), power_2_pad)
        placeholder[:, :train_data.shape[1]] = train_data # / lengthscale
        #placeholder[:, train_data.shape[1]] = np.sqrt(bias)
        train_data = placeholder

        for degree in range(3, 11, 2): # 11
            #var_comp_real, cov_comp_real = var_function(train_data, p=degree, D=train_data.shape[1])
            #var_real, cov_real = real_var_function(train_data, p=degree, D=2.*train_data.shape[1])

            #var_comp_real = var_function(train_data, p=degree, D=train_data.shape[1])
            var_comp, _ = comp_var_function(train_data, p=degree, D=power_2_pad)
            var_real = real_var_function(train_data, p=degree, D=power_2_pad)

            var_comp[var_comp < 0] = 0
            var_real[var_real < 0] = 0

            differences = var_comp / var_real
            differences = differences[~(differences.isnan() | differences.isinf())]
            differences = differences.view(-1).sort(descending=False)[0]
            if idx == 1:
                cutoff = 0.95
            else: cutoff = 0.8
            differences = differences[int(len(differences)*0):int(len(differences))] # *cutoff
            n = np.arange(1,len(differences)+1) / np.float(len(differences))
            axs[idx].step(differences,n, label='p={}'.format(degree))
            #axs[idx].hist(differences, bins=100, label='p={}'.format(degree), histtype='step', density=True)
            # cdf = differences.cumsum(dim=0) / differences[-1]
            #plt.hist(differences, 100, density=True, cumulative=True, histtype='step')
            #sns.ecdfplot(differences)
            #plt.xlim(right=0.5)
            
            sum_pos = float(((var_real - var_comp) > 0).sum().item())
            mse_advantage = (var_real - var_comp).sum().item()
            #sum_pos_cov = float(((cov_real - cov_comp_real) > 0).sum().item())

            #neg_cov_indices = ((cov_real - cov_comp_real) < 0).nonzero(as_tuple=True)


            print(
                'Degree:', degree,
                'Var advantage: {} / {} ({:.2f} %)'.format(sum_pos, len(train_data)**2, sum_pos / float(len(train_data)**2) * 100),
                'Mse advantage: {:.2f}'.format(mse_advantage)
                #'Cov advantage: {} / {} ({:.2f} %)'.format(sum_pos_cov, len(train_data)**2, sum_pos_cov / float(len(train_data)**2) * 100),
            )

        if idx==0:
            axs[idx].vlines(x=1.0, ymin=0, ymax=1.0, colors='black', label='Ratio=1', linestyles='dashed')
        else:
            axs[idx].vlines(x=1.0, ymin=0, ymax=1.0, colors='black', label='', linestyles='dashed')

        #axs[idx].xaxis.set_major_locator(plt.MaxNLocator(4))
        axs[idx].set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        axs[idx].set_xlim(left=0)
        if idx == 0:
            axs[idx].legend(loc='lower right')
        axs[idx].set_ylim(0, 1.05)
        #axs[idx].set_yscale('log')
        axs[idx].set_title('{}, d={}, {:.2f}% < 1'.format(dataset[0], power_2_pad, sum_pos / float(len(train_data)**2) * 100))
        axs[idx].set_xlabel('Variance ratio')
        axs[idx].set_ylabel('ECDF')

    #plt.yscale('log')

    # plt.show()

    # for D in [power_2_pad * i for i in range(1, 6)]: # 6
    #     #var_comp_real, cov_comp_real = var_function(train_data, p=degree, D=train_data.shape[1])
    #     #var_real, cov_real = real_var_function(train_data, p=degree, D=2.*train_data.shape[1])

    #     #var_comp_real = var_function(train_data, p=degree, D=train_data.shape[1])
    #     var_comp, _ = comp_var_function(train_data, p=3, D=D)
    #     var_real, _ = real_var_function(train_data, p=3, D=D)

    #     differences = torch.tril(var_real - var_comp).view(-1)
    #     differences = differences.sort(descending=False)[0]
    #     differences = differences[int(len(differences)*0):int(len(differences))] # *0.8
    #     # cdf = differences.cumsum(dim=0) / differences[-1]
    #     #plt.hist(differences, 100, density=True, cumulative=True, histtype='step')
    #     #sns.ecdfplot(differences)
    #     #plt.xlim(right=0.5)

    #     n = np.arange(1,len(differences)+1) / np.float(len(differences))
    #     axs[1].step(differences,n, label='D={}'.format(D))
    #     #axs[1].hist(differences, bins=100, label='D={}'.format(D), histtype='step', density=True)
        
    #     sum_pos = float(((var_real - var_comp) > 0).sum().item())
    #     mse_advantage = (var_real - var_comp).sum().item()
    #     #sum_pos_cov = float(((cov_real - cov_comp_real) > 0).sum().item())

    #     #neg_cov_indices = ((cov_real - cov_comp_real) < 0).nonzero(as_tuple=True)


    #     print(
    #         'D:', D,
    #         'Var advantage: {} / {} ({:.2f} %)'.format(sum_pos, len(train_data)**2, sum_pos / float(len(train_data)**2) * 100),
    #         'Mse advantage: {:.2f}'.format(mse_advantage)
    #         #'Cov advantage: {} / {} ({:.2f} %)'.format(sum_pos_cov, len(train_data)**2, sum_pos_cov / float(len(train_data)**2) * 100),
    #     )

    #plt.yscale('log')

    plt.tight_layout()
    #plt.savefig('figures/tensor_srht_ecdfs_2d.pdf', dpi=300)
    plt.show()
    exit()

    # Variance test to verify variance formulas
    n = 3
    d = 128
    D = 512
    n_points = 5
    data = torch.rand(n_points, d)
    data = data / data.norm(dim=1, keepdim=True)
    var_function = lambda X, p, D: var_gaussian_comp_real(X, p, D)

    def reference_kernel(data, k, c=0, lengthscale=1.):
        data = data / lengthscale
        # implement polynomial kernel and compare!
        return (data.mm(data.t()) + c)**k

    ts = PolynomialSketch(
        d_in=d,
        d_features=D,
        degree=n,
        bias=0,
        lengthscale=1.,
        var = 1.,
        ard = False,
        trainable_kernel=False,
        projection_type='gaussian',
        hierarchical=False,
        complex_weights=True
    )

    app_kernel_values = []
    scores = []
    # exact_kernel = dot_product**2

    ref_kernel = reference_kernel(data, n, c=0, lengthscale=1.)
    
    for _ in range(10000):
        ts.resample()
        y = ts.forward(data)
        y = torch.cat([y.real, y.imag], dim=1)
        approx_kernel = y @ y.t()
        app_kernel_values.append(approx_kernel)

        score = (approx_kernel - ref_kernel).pow(2).sum().sqrt() / ref_kernel.pow(2).sum().sqrt()
        scores.append(score.item())
    print(np.array(scores).mean())

    estimated_variance = torch.stack(app_kernel_values, dim=0).var(dim=0) #.sum(dim=-1)
    exact_variance = var_function(data, n, D)

    print('Estimated', estimated_variance)
    print('Exact', exact_variance)
    print('Rel. difference', (estimated_variance - exact_variance) / estimated_variance)

    print('Done!')