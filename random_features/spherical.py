import torch
import numpy as np
from scipy import interpolate
import scipy.special
from mpmath import fp, mp, mpf

import time

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from util.hypergeometric_function import HypGeomInt

from util.transforms import softplus, softplus_inverse
from util.LBFGS import FullBatchLBFGS

from random_features.projections import SRHT, GaussianTransform

def reference_kernel_poly(X, Y=None, a=2, p=3, znorm=False):
    """
    Compute (1. - ||x-y||^2 / a^2) ^ p
    znorm: If true, ||z|| is passed as the input argument.

    if a=2, we get: k(x, y) = (0.5 + 0.5 x.T y)^p
    Then we can control the kernel through ARD and bias.
    They may
    """

    if a < 2:
        print('a lower than 2 leads to bad approximations.')

    if Y is None:
        Y = X

    if znorm:
        kernel = (1. - X**2 / a**2)**p
    else:
        kernel = (X.unsqueeze(1) - Y.unsqueeze(0)).pow(2).sum(dim=-1)
        kernel = (1. - kernel / a**2)**p
    
    return kernel


class Spherical(torch.nn.Module):
    """
    Spherical Random Features (Pennington et al., 2015)

    We need to learn the distribution p(norm_w) first s.t. the reference kernel is approximated well.
    The final random features do not have a bias parameter because the approximation is shift-invariant.
    """

    def __init__(self, d_in, d_features, lengthscale='auto', var=1.0, ard=False, discrete_pdf=False, num_pdf_components=10,
                    epsilon=1e-15, trainable_kernel=False, complex_weights=False, projection_type='gaussian', device='cpu'):
        """
        d_in: Data input dimension
        d_features: Projection dimension
        lengthscale: Downscale of the inputs (1 if none)
        var: Scale of the final kernel
        ard: Automatic Relevance Determination = individual lengthscale/input dim.
        discrete_pdf: whether to use a dirac delta pdf (different from original approach)
        num_pdf_components: The number of gaussian distributions/discrete terms used to approximate the kernel function
        epsilon: Used to determine cutoff value for support of p(norm_w), smaller eps = larger support
        trainable_kernel: Learnable lengthscales, scale
        complex_weights: Whether to use complex-valued weights (almost always lower variances but more expensive)
        projection_type: gaussian/srht
        """
        super(Spherical, self).__init__()
        
        self.d_in = d_in
        self.d_features = d_features
        if not complex_weights:
            self.d_features = self.d_features // 2
        self.num_pdf_components = num_pdf_components
        self.epsilon = epsilon
        self.device = device
        self.discrete_pdf = discrete_pdf
        self.projection_type = projection_type
        self.complex_weights = complex_weights
        
        # mixture weights are only for the Gaussian mixture and can be negative as well
        self.mixture_weights = torch.nn.Parameter(
            np.sqrt(1./num_pdf_components) * (torch.rand(num_pdf_components, device=device)),
            requires_grad=True
        )

        # gaussian scales in case of gaussian mixture / discrete scales in case of discrete pdf
        self.log_mixture_scales = torch.nn.Parameter(
            softplus_inverse(np.sqrt(1./num_pdf_components) * torch.rand(num_pdf_components, device=device)),
            requires_grad=True
        )

        # used for discrete pdf
        self.log_support = torch.nn.Parameter(
            softplus_inverse(torch.rand(num_pdf_components, device=device)*10),
            requires_grad=True
        )

        if isinstance(lengthscale, str) and lengthscale == 'auto':
            lengthscale = np.sqrt(d_in)
        
        # trainable lengthscales only make sense for ARD
        # otherwise they violate the unit-norm constraint
        # (we take care of it in the forward pass)
        num_lengthscales = d_in if ard else 1
        self.log_lengthscale = torch.nn.Parameter(
            torch.ones(num_lengthscales, device=device) * np.log(lengthscale),
            requires_grad=trainable_kernel
        )

        self.log_var = torch.nn.Parameter(
            torch.ones(1, device=device) * np.log(var),
            requires_grad=trainable_kernel
        )
        

    def save_model(self, path):
        print('Saving model...')
        torch.save(self.state_dict(), path, _use_new_zipfile_serialization=False)

    def load_model(self, path):
        model = torch.load(path, map_location=self.device)
        self.load_state_dict(model)
        # model.to(self.device)

    def resample(self, num_points_w=1000):
        with torch.no_grad():
            if self.discrete_pdf:
                choices = softplus(self.log_support)
                probs = softplus(self.log_mixture_scales)
                probs = probs / probs.sum()
                pdf_samples = np.random.choice(choices.numpy(), self.d_features, p=probs.numpy())
            else:
                w_vals, pdf = self.compute_norm_pdf(num_points_w=num_points_w)
                # since we only have samples from the pdf, they need to be normalized for the ecdf
                pdf = pdf / pdf.sum()
                cdf = np.cumsum(pdf.cpu().numpy())
                # to prevent out of range sampling
                cdf[-1] = 1.
                # we do inverse cdf sampling
                f = interpolate.interp1d(cdf, w_vals.cpu().numpy())
                pdf_samples = f(np.random.rand(self.d_features))

        if self.projection_type == 'srht':
            self.feature_encoder = SRHT(self.d_in, self.d_features,
                                        complex_weights=False, shuffle=False, k=3, device=self.device)
        else:
            self.feature_encoder = GaussianTransform(self.d_in, self.d_features, complex_weights=False, device=self.device)

        self.feature_encoder.resample()
        if self.projection_type != 'srht':
            self.feature_encoder.weights.data = self.feature_encoder.weights.data \
                / self.feature_encoder.weights.data.norm(dim=0)
        self.scales = torch.from_numpy(pdf_samples).float().to(self.device)

    def move_submodules_to_cuda(self):
        self.feature_encoder.cuda()
        self.scales = self.scales.cuda()

    def forward(self, x):
        # we first apply the lengthscale
        x = x / self.log_lengthscale.exp()
        # then we unit-normalize the data (requirement)
        x = x / x.norm(dim=1, keepdim=True)

        x = self.feature_encoder.forward(x)
        if self.projection_type == 'srht':
            # takes care that the product HD 1/d HD 1/d HD becomes orthogonal
            # (1/sqrt(d) HD).T (1/sqrt(d) HD) = D.T 1/sqrt(d)H.T 1/sqrt(d)H D = D.T I D = I.
            x = x / np.sqrt(self.d_in)
        x = x * self.scales

        x = torch.stack([torch.cos(x), torch.sin(x)], dim=-1)
        if self.complex_weights:
            x = torch.view_as_complex(x)
        else:
            x = x.view(len(x), -1)

        x = x / np.sqrt(self.d_features)

        x = x * torch.exp(self.log_var / 2.)

        return x

    def find_cutoff(self):
        """
        Simple approximate inversion function to find value
        beyond which the pdf has negligibly small mass.

        The pdf of our density is Nakagami distributed with m=d/2, Omega=d/2 * 4 * sigma_i^2:
        sum_i c_i * (0.5 * w)^(d-1) * exp(-w^2/(4*sigma_i^2)) / gamma(d/2) * (1/sigma_i)^d

        We want to obtain the value of w for which the pdf is below a certain threshold epsilon.
        For this purpose we need the lambert W function that solves x*exp(x) for x.
        Since the pdf is not injective, there are two solutions.
        We need to choose the lower (-1) branch which will give us the rightmost value of w.

        We compute this cutoff value for every mixture component i.
        c_i is changed to |c_i| here to have proper density components.
        """

        with torch.no_grad():

            # (eps*gamma(d/2))^(2/(d-1)) using intermediate high precision
            eps_gamma_power = np.array(
                mp.power(mpf(self.epsilon) * mp.gamma(mpf(self.d_in) / mpf(2)),
                mpf(2)/mpf(self.d_in-1)),
                dtype='float64')
            x = (2. / (self.d_in-1)) * eps_gamma_power \
                * (softplus(self.log_mixture_scales) / torch.abs(self.mixture_weights))**(2/(self.d_in-1))

            cutoff = softplus(self.log_mixture_scales) \
                * torch.from_numpy(np.sqrt(-2. * (self.d_in-1) * scipy.special.lambertw(x.cpu().numpy(), k=-1).real)).float().to(self.device)

            return cutoff

    def compute_norm_pdf(self, num_points_w=1000):
        """
        Computes the pdf of the mixture of chi-distributions.
        """

        cutoff = torch.max(self.find_cutoff())

        # cannot backprop through -inf from log(0)
        w_vals = torch.linspace(1e-9, cutoff, steps=num_points_w).to(self.device) # .type(self.dtype)

        # argument of the gaussians (n_gaussians x w_samples)
        gauss_arg = 0.5 * (1. / softplus(self.log_mixture_scales).unsqueeze(1)) * w_vals.unsqueeze(0)

        log_gamma = scipy.special.gammaln(self.d_in / 2)
        A = torch.exp(-gauss_arg.pow(2) + (self.d_in-1)*torch.log(gauss_arg) - log_gamma) \
            * 1. / softplus(self.log_mixture_scales).unsqueeze(1)
        pdf = A.t().mv(self.mixture_weights)
        # The density should be non-negative
        pdf[pdf < 0] = 0

        # the pdf needs to be manually normalized by its estimated area because of the transformation
        # otherwise it would be a Nakagami distribution with m = d/2 and Omega_i = d/2 * 4 sigma_i^2
        pdf = pdf / (pdf.sum() * w_vals[1])

        return w_vals, pdf

    def approximate_kernel(self, X, Y=None, znorm=False, num_points_w=1100):
        """
        Current kernel approximation according to saved pdf for norm_w.
        """

        if Y is None:
            Y = X

        if not znorm:
            X = (X.unsqueeze(1) - Y.unsqueeze(0)).norm(dim=-1).view(-1)

        if self.discrete_pdf:
            # adaptive discrete grid
            wx = (softplus(self.log_support).unsqueeze(1) * X.unsqueeze(0))

            # the difference between the approximate kernel and the real one
            # the integration needs the step size
            pdf = softplus(self.log_mixture_scales)
            pdf = pdf / pdf.sum()

        else:
            w_vals, pdf = self.compute_norm_pdf(num_points_w=num_points_w)

            # fixed discrete grid
            wx = (w_vals.unsqueeze(1) * X.unsqueeze(0))
            # the continuous integration needs the step size
            pdf = pdf * w_vals[1]

        hyp0f1 = HypGeomInt.apply(wx, self.d_in)
        approx_kernel = hyp0f1.t().mv(pdf)
        
        return approx_kernel
    
    def fit_approximation(self, reference_kernel, num_points_z=500, num_points_w=500, epsilon=1e-20,
                            lr=1e-3, iterations=100, save_model=True, save_name='spherical_model.torch'):
        """
        Fits a spectral density over the norm of the random feature distribution to approximate a target kernel.
        a let's us control the ratio of scale and bias
        We cannot change this value later on because the data needs to have unit-norm.
        Even when appending a bias to the input vectors it will vanish in z!
        => We can only do ARD!
        """

        z_vals = torch.linspace(0, 2., steps=num_points_z)
        # z_vals are the norm(x-y)
        reference_kernel = reference_kernel(z_vals, znorm=True)

        for i in range(iterations):
            print('### Iteration: {} ###'.format(i))

            # optimizer = torch.optim.SGD(self.parameters(), lr=lr)
            optimizer = FullBatchLBFGS(self.parameters(), lr=lr, history_size=10, line_search='Wolfe')

            def closure():
                optimizer.zero_grad()

                approx_kernel = self.approximate_kernel(z_vals, znorm=True, num_points_w=num_points_w)
                loss = (reference_kernel - approx_kernel).pow(2).mean()
                print('Loss: {}'.format(loss.item()))

                return loss

            loss = closure()
            loss.backward()
            options = {'closure': closure, 'current_loss': loss, 'max_ls': 10}
            loss, _, lr, _, F_eval, G_eval, _, _ = optimizer.step(options)
            
            # optimizer.step()

        if save_model:
            self.save_model('saved_models/' + save_name + '.torch')


if __name__ == '__main__':
    polynomial_kernel = lambda x, znorm=True: reference_kernel_poly(x, a=a, p=p, znorm=znorm)

    iterations = 200
    num_points_w = 500
    num_points_z = 500
    d_in = 1024
    d_out = 10000
    a = 2
    p = 10
    num_components = 10
    complex_weights = True
    projection_type = 'gaussian'
    
    reference_kernel = polynomial_kernel
    save_name='poly_a{:.2f}_p{}_d{}'.format(a, p, d_in)

    spherical_model = Spherical(
        d_in, d_out, discrete_pdf=False, num_pdf_components=num_components,
        complex_weights=complex_weights, projection_type=projection_type)
    # this does not use the lengthscale and kernel variance
    spherical_model.fit_approximation(reference_kernel, num_points_w=num_points_w, num_points_z=num_points_z,
                                        lr=1e-2, iterations=iterations, save_name=save_name)

    # spherical_model.load_model('saved_models/' + save_name + '.torch')

    # The target kernel only takes values in the range [0, 2]
    z_vals = torch.linspace(0, 2., steps=500)
    ref_kernel = reference_kernel(z_vals, znorm=True)

    with torch.no_grad():
        approx_kernel = spherical_model.approximate_kernel(z_vals, znorm=True, num_points_w=num_points_w*10)

        mean_abs_error = torch.abs(ref_kernel - approx_kernel).mean()
        max_abs_error = torch.abs(ref_kernel - approx_kernel).max()

    print('Mean absolute error: {}'.format(mean_abs_error))
    print('Max absolute error: {}'.format(max_abs_error))

    import matplotlib.pyplot as plt
    # plot kernel comparison
    plt.plot(z_vals, approx_kernel, label='Approximation')
    plt.plot(z_vals, ref_kernel, label='Exact')
    plt.ylim(0, 1.0)
    plt.xlim(0, 2.0)
    plt.legend(loc='upper right')
    plt.show()

    # plot pdf
    if spherical_model.discrete_pdf:
        plt.scatter(softplus(spherical_model.log_support).detach(), softplus(spherical_model.log_mixture_scales).detach())
    else:
        w_vals, pdf = spherical_model.compute_norm_pdf(num_points_w=num_points_w)
        plt.plot(w_vals, pdf.detach())
    plt.show()


    # random features
    data = torch.randn(10, d_in).float()
    data = data / data.norm(dim=1, keepdim=True)
    ref_kernel = reference_kernel(data, znorm=False)

    for D in [1024, 2048, 4096, 8192]:
        spherical_model.d_features = D
        spherical_model.resample(num_points_w=num_points_w*10)

        with torch.no_grad():
            features = spherical_model.forward(data)
            approx_kernel = features @ features.conj().t()
            if complex_weights:
                approx_kernel = approx_kernel.real

            difference = approx_kernel - ref_kernel
            score = (difference.pow(2).sum().sqrt() / ref_kernel.pow(2).sum().sqrt())

        print(score.item())
