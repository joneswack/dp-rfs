import torch
import torch.nn as nn
import numpy as np

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from random_features.maclaurin import Maclaurin
from random_features.polynomial_sketch import PolynomialSketch
from random_features.rff import RFF

from util.measures import Fixed_Measure, Exponential_Measure, P_Measure
from util.kernels import gaussian_kernel
import util.data


class GaussianApproximator(nn.Module):
    """
    This module is a wrapper to compute random features for the Gaussian kernel in numerous ways.
    """

    def __init__(self, d_in, d_features, approx_degree=4, lengthscale='auto', var=1.0,
                    ard=False, trainable_kernel=False, method='poly_sketch', dtype=torch.FloatTensor,
                    projection_type='srht', hierarchical=False, complex_weights=False):
        """
        d_in: Data input dimension
        d_features: Projection dimension
        approx_degree: The degree of the approximation
        lengthscale: Downscale of the inputs (1 if none)
        var: Scale of the final kernel
        ard: Automatic Relevance Determination = individual lengthscale/input dim.
        trainable_kernel: Learnable bias, lengthscales, scale
        method: Approximation technique (maclaurin/maclaurin_p/poly_sketch/rff)
            maclaurin is the optimized maclaurin method while maclarin_p the approach by Kar & Karnick 2012
        projection_type: rademacher/gaussian/srht/countsketch_sparse/countsketch_dense/countsketch_scatter
        hierarchical: Whether to use hierarchical sketches (overcomes exponential variances w.r.t. p but is not always better)
        complex_weights: Whether to use complex-valued weights (almost always lower variances but more expensive)
        """

        super(GaussianApproximator, self).__init__()

        if projection_type == 'srht':
            if method == 'poly_sketch' and not np.log2(d_in+1).is_integer():
                raise RuntimeError('poly_sketch in combination with srht requires d_in=2**n-1 because of bias!')
            elif method != 'poly_sketch' and not np.log2(d_in).is_integer():
                raise RuntimeError('srht requires d_in=2**n!')
        
        self.d_in = d_in
        self.d_features = d_features
        self.approx_degree = approx_degree
        
        if isinstance(lengthscale, str) and lengthscale == 'auto':
            # alternatively, we can simply set it to one
            lengthscale = np.sqrt(d_in)

        num_lengthscales = d_in if ard else 1
        self.log_lengthscale = torch.nn.Parameter(torch.ones(num_lengthscales).type(dtype) * np.log(lengthscale), requires_grad=trainable_kernel)
        self.log_var = torch.nn.Parameter(torch.ones(1).type(dtype) * np.log(var), requires_grad=trainable_kernel)

        self.method = method
        self.projection_type = projection_type
        self.hierarchical = hierarchical
        self.complex_weights = complex_weights

        if method == 'maclaurin':
            # maclaurin with optimized feature distribution
            # the maclaurin series for the exponential function
            kernel_coefs = lambda x: Exponential_Measure.coefs(x)
            # we initialize the distribution over degrees to be uniform (this will be overridden later)
            measure = Fixed_Measure(False, [1]*approx_degree, True)

            self.feature_encoder = Maclaurin(d_in, d_features, coef_fun=kernel_coefs, measure=measure,
                                module_args={
                                    'projection': projection_type,
                                    'hierarchical': hierarchical,
                                    'complex_weights': complex_weights
                                },
                                # the RF hyperparameters are fixed for this module
                                bias=0., lengthscale=1, trainable_kernel=False)
        elif method == 'maclaurin_p':
            # classical maclaurin with exponentially decaying measure (Kar & Karnick 2012)
            kernel_coefs = lambda x: Exponential_Measure.coefs(x)

            measure = P_Measure(2., False, np.inf)
            self.feature_encoder = Maclaurin(d_in, d_features, coef_fun=kernel_coefs, measure=measure,
                                module_args={
                                    'projection': projection_type,
                                    'hierarchical': hierarchical,
                                    'complex_weights': complex_weights
                                },
                                # the RF hyperparameters are fixed for this module
                                bias=0., lengthscale=1, trainable_kernel=False)
        elif method == 'poly_sketch':
            # exp(x.T y) is approximated through (x.T y / (approx_degree*lengthscale**2) + 1)^approx_degree
            # the lengthscale is set inside the forward pass of THIS module
            # the sqrt(approx_degree) scaling stays fixed on the other hand
            self.feature_encoder = PolynomialSketch(d_in, d_features, degree=approx_degree,
                                bias=1, lengthscale=np.sqrt(approx_degree),
                                trainable_kernel=False, projection_type=projection_type,
                                complex_weights=complex_weights)
        elif method == 'rff':
            self.feature_encoder = RFF(d_in, d_features, lengthscale=1, trainable_kernel=False,
                                complex_weights=complex_weights, projection_type=projection_type)
        else:
            raise RuntimeError('Method {} not available!'.format(method))

    def initialize_sampling_distribution(self, random_samples, min_sampling_degree=2):
        """
        Initialize random feature distribution of maclaurin method
        """
        with torch.no_grad():
            if self.method == 'maclaurin':
                # optimized maclaurin method
                target_kernel = gaussian_kernel(random_samples, random_samples, lengthscale=self.log_lengthscale.exp())
                exp_vars, exp_covs, exp_sq_biases = self.feature_encoder.expected_variances_and_biases(
                    random_samples / self.log_lengthscale.exp(), target_kernel, gaussian_kernel=True)
                self.feature_encoder.optimize_sampling_distribution(exp_vars, exp_covs, exp_sq_biases, min_degree=min_sampling_degree)
                print('Optimized distribution: {}'.format(self.feature_encoder.measure.distribution))

    def resample(self):
        self.feature_encoder.resample()

    def forward(self, x):
        x = x / self.log_lengthscale.exp()
        x_norm = x.norm(dim=1, keepdim=True)

        if self.method in ['poly_sketch', 'maclaurin', 'maclaurin_p']:
            prefactor = torch.exp(-x_norm.pow(2) / 2.)
        else:
            prefactor = torch.ones(len(x), 1, device=x.device, dtype=x.dtype)

        x = self.feature_encoder(x)

        if self.complex_weights:
            prefactor = prefactor.type(torch.complex64)
        
        x = x * prefactor

        x = x * torch.exp(self.log_var / 2.)

        return x

if __name__ == "__main__":
    torch.manual_seed(0)
    complex_weights = False
    projection_type = 'srht'
    method = 'maclaurin_p'
    degree = 10 # maximum maclaurin approx. degree
    d = 1024 # input dimension (power of 2 for srht, power of 2 - 1 for srht with poly_sketch)
    data = util.data.load_dataset('config/datasets/mnist.json', standardize=True, normalize=False)
    data_name, train_data, test_data, train_labels, test_labels = data

    data = train_data[torch.randperm(len(train_data))][:1000]
    data = util.data.pad_data_pow_2(data)[:, :d]
    lengthscale = torch.cdist(data, data, p=2.0).median()

    for d_features in [1024, 2048, 4096, 8192]:
        feature_encoder = GaussianApproximator(
            data.shape[1], d_features, approx_degree=degree, method=method, projection_type=projection_type,
            lengthscale=lengthscale, trainable_kernel=False, complex_weights=complex_weights)
        feature_encoder.initialize_sampling_distribution(data)
        feature_encoder.resample()
        features = feature_encoder.forward(data)

        ref_kernel = gaussian_kernel(data)

        approx_kernel = features @ features.conj().t()

        if complex_weights:
            approx_kernel = approx_kernel.real

        # score = torch.abs(approx_kernel - ref_kernel) / torch.abs(ref_kernel)
        score = (approx_kernel - ref_kernel).pow(2).sum().sqrt() / ref_kernel.pow(2).sum().sqrt()
        print(d_features, score.item())

    print('Done!')