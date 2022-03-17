import torch
import numpy as np

import time
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from util.helper_classes import LazyDict
from util.measures import Polynomial_Measure, P_Measure, Fixed_Measure

from random_features.polynomial_sketch import PolynomialSketch


class Maclaurin(torch.nn.Module):
    """
    A Maclaurin series approximation composed of polynomial sketches.
    """

    def __init__(self, d_in, d_features, coef_fun,
                        module_args={'projection': 'srht', 'hierarchical': False, 'complex_weights': False},
                        measure=P_Measure(2, False, 10), bias=0., lengthscale='auto', var=1.0, ard=False,
                        trainable_kernel=False, device='cpu'):
        """
        d_in: Data input dimension
        d_features: Projection dimension
        coef_fun: function yielding Maclaurin coefficients
        module_args: polynomial module arguments (projection, hierarchical, complex_weights)
        measure_args: measure arguments, e.g. (degree p, heuristic h01, max sample degree max_val)
        bias: The bias b (eventually added through input modifiction)
        lengthscale: Downscale of the inputs (1 if none)
        var: Scale of the final kernel
        ard: Automatic Relevance Determination = individual lengthscale/input dim.
        trainable_kernel: Learnable bias, lengthscales, scale
        device: cpu or cuda
        """
        super(Maclaurin, self).__init__()

        self.d_in = d_in
        self.d_features = d_features
        self.coef_fun = coef_fun
        self.module_args = module_args
        self.device = device

        self.measure = measure

        if measure.h01 and isinstance(measure, Polynomial_Measure):
            if measure.c == 0 or measure.p < 2:
                raise RuntimeError('h01 cannot be activated for linear/homogeneous polynomial kernels')

        if measure.has_constant:
            # we add degree 0 (constant) of the ML expansion as a random feature
            self.d_features -= 1
        if measure.h01:
            # we append the entire input to the random features
            self.d_features -= d_in
            if self.d_features < 0:
                self.d_features = 0

        # we initialize the kernel hyperparameters
        if bias != 0:
            self.log_bias = torch.nn.Parameter(torch.ones(1, device=device).float() * np.log(bias), requires_grad=trainable_kernel)
            self.d_in = self.d_in + 1
        else:
            self.log_bias = None
        
        if isinstance(lengthscale, str) and lengthscale == 'auto':
            lengthscale = np.sqrt(d_in)

        num_lengthscales = d_in if ard else 1
        self.log_lengthscale = torch.nn.Parameter(torch.ones(num_lengthscales, device=device).float() * np.log(lengthscale), requires_grad=trainable_kernel)
        self.log_var = torch.nn.Parameter(torch.ones(1, device=device).float() * np.log(var), requires_grad=trainable_kernel)

    def expected_variances_and_biases(self, training_data, target_kernel, gaussian_kernel=False):
        """
        This function serves to precompute expected variances and squared biases.

        training_data: Design matrix X used to estimate the biases and variances
            (the more data we provide the better the estimate, but scaling is quadratic)
        target_kernel: Target kernel computed on the training_data (pairwise evaluations)
        gaussian_kernel: Whether to obtain estimates for the Gaussian kernel
        """

        with torch.no_grad():
            if self.module_args['hierarchical']:
                raise RuntimeError('Expected variance for hierarchical random features not implemented yet!')

            # we first apply the lengthscale
            training_data = training_data / self.log_lengthscale.exp()
            if self.log_bias is not None:
                # we then append the bias
                training_data = torch.cat([(0.5 * self.log_bias).exp().repeat(len(training_data), 1), training_data], dim=1)

            # sum(x_i y_i)
            dot_product = training_data @ training_data.t()
            # sum(x_i^2 y_i^2)
            squared_dot_product = training_data.pow(2) @ training_data.pow(2).t()
            # ||x||^2 ||y||^2
            norms_squared = (training_data**2).sum(dim=1).unsqueeze(1) * (training_data**2).sum(dim=1).unsqueeze(0)

            if gaussian_kernel:
                # prefactor = exp(-||x||^2 / 2.)*exp(-||y||^2 / 2.) for all x, y in the training data
                squared_norm = (training_data**2).sum(dim=1)
                prefactor = torch.exp(-squared_norm / 2.).unsqueeze(1) * torch.exp(-squared_norm / 2.).unsqueeze(0)
            else:
                prefactor = 1.

            # compute biases of successively adding the first n degrees
            taylor_coefs = self.coef_fun(np.arange(0, self.measure.max_val+1))
            # we assume the bias and the linear term to be included (only for h1)
            approx_kernel = taylor_coefs[0]
            if self.measure.h01:
                approx_kernel += taylor_coefs[1] * dot_product
                start_index = 2
            else:
                start_index = 1

            expected_squared_biases = []
            for idx in range(start_index, len(taylor_coefs)):
                approx_kernel += taylor_coefs[idx] * dot_product**idx
                expected_squared_biases.append((target_kernel - prefactor * approx_kernel).pow(2).mean().item())

            second_moments = LazyDict({
                'gaussian': lambda:
                    norms_squared + dot_product**2 if self.module_args['complex_weights'] \
                    else norms_squared + 2.*dot_product**2,
                'rademacher': lambda:
                    norms_squared + dot_product**2 - squared_dot_product if self.module_args['complex_weights'] \
                    else norms_squared + 2.*(dot_product**2 - squared_dot_product),
                'srht': lambda:
                    norms_squared + dot_product**2 - squared_dot_product if self.module_args['complex_weights'] \
                    else norms_squared + 2.*(dot_product**2 - squared_dot_product)
            })

            def covariance(degree):
                if self.module_args['projection'] != 'srht':
                    return torch.zeros_like(dot_product)
                if self.module_args['complex_weights']:
                    return (dot_product**2 - (1./(self.d_in-1.)) \
                        * (norms_squared - squared_dot_product))**degree - dot_product**(2*degree)
                else:
                    return (dot_product**2 - (1./(self.d_in-1.)) \
                        * (norms_squared + dot_product**2 - 2*squared_dot_product))**degree - dot_product**(2*degree)

            # we only need degrees 1, ..., p since no randomness is required for the bias
            expected_vars = []
            expected_covs = []

            start_index = 2 if self.measure.h01 else 1

            for deg in range(start_index, self.measure.max_val+1):
                if self.module_args['projection'].split('_')[0] == 'countsketch':
                    var_term = norms_squared**deg + dot_product**(2*deg)
                else:
                    var_term = second_moments[self.module_args['projection']]**deg - dot_product**(2*deg)
                
                cov_term = covariance(deg)

                for estimator in [var_term, cov_term]:
                    expected_val = prefactor**2 * estimator
                    # take away the diagonal elements to have unbiased u-statistic
                    expected_val = torch.tril(expected_val, diagonal=-1)
                    # filter out infinity values
                    inf_indices = torch.isinf(expected_val)
                    nan_indices = torch.isnan(expected_val)
                    if inf_indices.float().mean() > 0.05:
                        raise RuntimeWarning('More than 5 percent of the expected variances are infinity. '
                            + 'The filtered variance may be largely underestimated!')
                    expected_val[inf_indices | nan_indices] = 0
                    # squared matrix - diagonal - subtracted indices
                    num_elements = (len(expected_val)**2 - len(expected_val)) / 2. - inf_indices.sum().item() - nan_indices.sum().item()
                    
                    if torch.equal(estimator, var_term):
                        expected_vars.append(expected_val.sum().item() / num_elements)
                    else:
                        expected_covs.append(expected_val.sum().item() / num_elements)

            return np.array(expected_vars), np.array(expected_covs), np.array(expected_squared_biases)

    def optimize_sampling_distribution(self, expected_vars, expected_covs, exp_sq_biases, min_degree=2):
        """
        Finds the optimal sampling distribution between the polynomial degrees.
        Moreover, the optimal degree between min_degree and self.degree is chosen.
        """

        with torch.no_grad():
            if not isinstance(self.measure, Fixed_Measure):
                raise RuntimeError('Optimizing the sampling distribution only makes sense for FixedMeasure!')

            def compute_variances(expected_vars, expected_covs, distribution):
                # these are the expected total variances
                num_non_zero_pairs = np.floor(distribution / self.d_in)*self.d_in*(self.d_in-1) \
                    + (distribution % self.d_in) * (distribution % self.d_in - 1)
                
                vars_type_1 = expected_vars/distribution \
                    + (num_non_zero_pairs/(distribution**2)) * expected_covs
                vars_type_2 = (expected_vars + (self.d_in - 1) * expected_covs) / distribution
                
                variance_terms = vars_type_2

                # mask for 1 <= D_n <= d, E[Cov] <= 0
                mask = (distribution <= self.d_in) & (expected_covs <= 0)
                variance_terms[mask] = vars_type_1[mask]

                start_index = 2 if self.measure.h01 else 1
                taylor_coefs = self.coef_fun(np.arange(start_index, len(variance_terms)+1))
                return (taylor_coefs**2 * variance_terms).sum()

            def optimize_distribution(expected_vars, expected_covs):
                # we start of with 1 feature for every coefficient
                distribution = np.ones(len(expected_vars))
                final_variance = compute_variances(expected_vars, expected_covs, distribution)

                while distribution.sum() < self.d_features:
                    # print(distribution)
                    best_variance = compute_variances(expected_vars, expected_covs, distribution)
                    best_distribution = None
                    for i in range(len(expected_vars)):
                        temp_distribution = distribution.copy()
                        temp_distribution[i] += 1
                        current_variance = compute_variances(expected_vars, expected_covs, temp_distribution)
                        if current_variance <= best_variance:
                            best_variance = current_variance
                            best_distribution = temp_distribution

                    if best_distribution is None:
                        distribution[0] += 1
                        final_variance = compute_variances(expected_vars, expected_covs, distribution)
                    else:
                        distribution = best_distribution
                        final_variance = best_variance

                return distribution, final_variance

            start_index = min_degree-1
            best_score = None
            best_distribution = None
            for i in range(start_index, min(self.d_features, len(expected_vars))):
                distribution, exp_variance = optimize_distribution(expected_vars[:i+1], expected_covs[:i+1])
                score = exp_variance + exp_sq_biases[i]
                if best_score is None or score <= best_score:
                    best_score = score
                    best_distribution = distribution

            self.measure.distribution = list(best_distribution.astype('int32'))

    def resample(self):
        if self.d_features == 0:
            return

        degrees = self.measure.rvs(size=self.d_features)
        # degrees are sorted from highest to lowest
        degrees, proj_dims = np.unique(np.array(degrees), return_counts=True)
        self.coefs = self.coef_fun(degrees)
        if isinstance(self.measure, P_Measure):
            # ensures unbiasedness of maclaurin estimator
            self.coefs /= self.measure._pmf(degrees)

        self.modules = []
        for degree, dim in zip(degrees, proj_dims):
            # we skip the constant
            # the bias and lengthscales will already be included in the data
            proj = self.module_args['projection']
            hier = self.module_args['hierarchical'] if degree >= 3 else False
            complex_weights = self.module_args['complex_weights']

            mod = PolynomialSketch(self.d_in, int(dim), degree=degree,
                                    bias=0, lengthscale=1.0, var=1.0, projection_type=proj,
                                    hierarchical=hier, complex_weights=complex_weights,
                                    trainable_kernel=False, device=self.device)
            mod.resample()
            self.modules.append(mod)

    def move_submodules_to_cuda(self):
        for mod in self.modules:
            mod = mod.cuda()
            mod.move_submodules_to_cuda()

    def forward(self, x):
        # we first apply the lengthscale
        x = x / self.log_lengthscale.exp()
        if self.log_bias is not None:
            # we then append the bias
            x = torch.cat([(0.5 * self.log_bias).exp().repeat(len(x), 1), x], dim=1)

        if self.d_features > 0:
            if not isinstance(self.measure, P_Measure):
                features = torch.cat([
                    self.modules[i].forward(x) * np.sqrt(self.coefs[i]) for i in range(len(self.modules))
                ], dim=1)
            else:
                # we need to adapt the scaling of the features per degree
                features = torch.cat([
                    self.modules[i].forward(x) * np.sqrt(self.coefs[i]) * np.sqrt(self.modules[i].d_features) for i in range(len(self.modules))
                ], dim=1)
                features = features / np.sqrt(self.d_features)

        # add degree 0 and 1 if desired
        add_features = None

        if self.measure.has_constant:
            add_features = torch.tensor(self.coef_fun(0)).float().sqrt().repeat(len(x), 1)

        if add_features is not None:
            if x.is_cuda:
                add_features = add_features.cuda()

            if self.measure.h01:
                # we need to append the linear features
                linear = torch.tensor(self.coef_fun(1)).float().sqrt() * x
                add_features = torch.cat([add_features, linear], dim=1)

            if self.module_args['complex_weights']:
                add_features = add_features.type(torch.complex64)
            
            features = torch.cat([add_features, features], dim=1)

        features = features * torch.exp(self.log_var / 2.)

        return features


if __name__ == "__main__":
    def reference_kernel(data, k, c, log_lengthscale='auto'):
        if isinstance(log_lengthscale, str) and log_lengthscale == 'auto':
            # lengthscale = sqrt(d_in)
            log_lengthscale = 0.5 * np.log(data.shape[1])

        data = data / np.exp(log_lengthscale)
        # implement polynomial kernel and compare!
        return (data.mm(data.t()) + c)**k

    torch.manual_seed(0)
    data = torch.randn(100, 8)
    data = data - data.mean(dim=0)
    data = data / data.norm(dim=1, keepdim=True)

    degree = 20
    a = 4.
    bias = 1.-2./a**2
    lengthscale = a / np.sqrt(2.)
    complex_weights = False
    hierarchical = False
    projection_type = 'countsketch_scatter'

    # the maclaurin series for the polynomial kernel function
    kernel_coefs = lambda x: Polynomial_Measure.coefs(x, degree, bias)
    # we initialize the distribution over degrees to be uniform (this will be overridden later)
    measure = Fixed_Measure(False, [1]*degree, True)

    ref_kernel = reference_kernel(data, degree, bias, log_lengthscale=np.log(lengthscale))

    dims = [1024 * i for i in range(1, 10)]

    for D in dims:
        scores = []
        for seed in np.arange(10):
            torch.manual_seed(seed)

            feature_encoder = Maclaurin(
                data.shape[1],
                D, kernel_coefs,
                module_args={
                    'projection': projection_type,
                    'hierarchical': hierarchical,
                    'complex_weights': complex_weights
                },
                measure=measure, bias=0,
                lengthscale=lengthscale,
                var=1., ard=False, trainable_kernel=False
            )

            exp_vars, exp_covs, exp_sq_biases = feature_encoder.expected_variances_and_biases(data, ref_kernel, gaussian_kernel=False)
            feature_encoder.optimize_sampling_distribution(exp_vars, exp_covs, exp_sq_biases, min_degree=2)
            print('Optimized distribution: {}'.format(feature_encoder.measure.distribution))

            feature_encoder.resample()
            # features = tensorsketch(data, 2, 0, num_features=10000)
            # features = ts.forward(data)

            projection = feature_encoder.forward(data)

            approx_kernel = projection @ projection.conj().t()
            if approx_kernel.dtype in [torch.complex32, torch.complex64, torch.complex128]:
                approx_kernel = approx_kernel.real

            # score = torch.abs(approx_kernel - ref_kernel) / torch.abs(ref_kernel)
            score = (approx_kernel - ref_kernel).pow(2).sum().sqrt() / ref_kernel.pow(2).sum().sqrt()
            scores.append(score.item())
        print(np.array(scores).mean())

    print('Done!')