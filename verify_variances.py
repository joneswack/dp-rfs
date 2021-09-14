import torch
import numpy as np
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

    return var_term + cov_term, cov_term

def var_tensor_srht_comp_real(X, p=1., D=1.):
    dot_product_squared, squared_dot_product, norms_squared = get_pairwise_elements(X)
    d = X.shape[1]
    num_non_zero_pairs = np.floor(D / d)*d*(d-1) + (D % d) * (D % d - 1)

    var_term = var_rademacher_comp_real(X, p, D)
    cov_term = num_non_zero_pairs / D**2 * \
        ((dot_product_squared - 1./(d-1.) * (dot_product_squared - squared_dot_product))**p - dot_product_squared**p)
    cov_term += num_non_zero_pairs / D**2 * \
        ((dot_product_squared - 1./(d-1.) * (norms_squared - squared_dot_product))**p - dot_product_squared**p)
    cov_term *= 0.5

    return var_term + cov_term, cov_term




if __name__ == "__main__":
    a = 4.
    bias = 1.-2./a**2
    lengthscale = a / np.sqrt(2.)

    var_function = lambda X, p, D: var_gaussian_comp_real(X, p, D)
    real_var_function = lambda X, p, D: var_gaussian_real(X, p, D)

    train_data = torch.randn(10000, 128, dtype=torch.float64)
    # train_data = torch.tensor([[0, 1], [1, 0]], dtype=torch.float64)
    # train_data = train_data / train_data.norm(dim=1, keepdim=True)

    # train_data, train_labels = torch.load('../datasets/export/fashion_mnist/pytorch/train_fashion_mnist.pth')
    # test_data, test_labels = torch.load('../datasets/export/fashion_mnist/pytorch/test_fashion_mnist.pth')

    # train_data, train_labels = torch.load('../datasets/export/mnist/pytorch/train_mnist.pth')
    # test_data, test_labels = torch.load('../datasets/export/mnist/pytorch/test_mnist.pth')

    # train_data, train_labels = torch.load('../datasets/export/adult/pytorch/train_adult.pth')
    # test_data, test_labels = torch.load('../datasets/export/adult/pytorch/test_adult.pth')

    # train_data, train_labels = torch.load('../datasets/export/eeg/pytorch/eeg.pth')

    train_data = train_data.reshape(len(train_data), -1)

    train_data = train_data / train_data.norm(dim=1, keepdim=True)

    indices = torch.randint(len(train_data), (1000,))
    train_data = train_data[indices]

    power_2_pad = int(2**np.ceil(np.log2(train_data.shape[1])))

    # placeholder = torch.zeros(len(train_data), power_2_pad)
    # placeholder[:, :train_data.shape[1]] = train_data / lengthscale
    # placeholder[:, train_data.shape[1]] = np.sqrt(bias)
    # train_data = placeholder

    for degree in range(2, 10):
        #var_comp_real, cov_comp_real = var_function(train_data, p=degree, D=train_data.shape[1])
        #var_real, cov_real = real_var_function(train_data, p=degree, D=2.*train_data.shape[1])

        var_comp_real = var_function(train_data, p=degree, D=train_data.shape[1])
        var_real = real_var_function(train_data, p=degree, D=2.*train_data.shape[1])
        
        sum_pos = float(((var_real - var_comp_real) > 0).sum().item())
        #sum_pos_cov = float(((cov_real - cov_comp_real) > 0).sum().item())

        #neg_cov_indices = ((cov_real - cov_comp_real) < 0).nonzero(as_tuple=True)


        print(
            'Degree:', degree,
            'Var advantage: {} / {} ({:.2f} %)'.format(sum_pos, len(train_data)**2, sum_pos / float(len(train_data)**2) * 100),
            #'Cov advantage: {} / {} ({:.2f} %)'.format(sum_pos_cov, len(train_data)**2, sum_pos_cov / float(len(train_data)**2) * 100),
        )


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