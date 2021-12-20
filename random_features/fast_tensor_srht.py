import torch
import numpy as np
import math
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from random_features.projections import generate_rademacher_samples
from util.hadamard_cuda.fwht import FastWalshHadamardTransform


class FastTensorSRHT(torch.nn.Module):
    """ FastTensorSRHT. """

    def __init__(self, d_in, d_features, degree, bias=0, lengthscale='auto', var=1.0, ard=False,
                    complex_real=False, trainable_kernel=False, device='cpu'):
        """
        d_in: Data input dimension (needs to be a power of 2)
        d_features: Projection dimension
        degree: Degree of the polynomial kernel
        complex_weights: Whether to use complex-valued projections
        """
        super(FastTensorSRHT, self).__init__()
        self.d_in = d_in
        if complex_real:
            d_features = d_features // 2
        self.d_features = d_features
        self.degree = degree
        self.num_blocks = int(math.ceil(float(self.d_features)/float(self.d_in)))
        self.complex_weights = complex_real
        self.complex_real = complex_real
        self.device = device

        # we initialize the kernel hyperparameters
        self.log_bias = None
        if bias != 0:
            self.log_bias = torch.nn.Parameter(
                torch.ones(1, device=device).float() * np.log(bias),
                requires_grad=trainable_kernel
            )
            self.d_in = self.d_in + 1
        
        if isinstance(lengthscale, str) and lengthscale == 'auto':
            lengthscale = np.sqrt(d_in)

        num_lengthscales = d_in if ard else 1
        self.log_lengthscale = torch.nn.Parameter(
            torch.ones(num_lengthscales, device=device).float() * np.log(lengthscale),
            requires_grad=trainable_kernel
        )
        self.log_var = torch.nn.Parameter(
            torch.ones(1, device=device).float() * np.log(var),
            requires_grad=trainable_kernel
        )

        self.rad = torch.nn.Parameter(None, requires_grad=False)
        self.permutations = torch.nn.Parameter(None, requires_grad=False)

    def resample(self):
        # we copy D_i for different blocks
        self.rad.data = generate_rademacher_samples((self.degree, self.d_in), complex_weights=self.complex_weights, device=self.device)
        # generate an index permutation over B*d
        permutations = []
        for _ in range(self.degree):
            indices = torch.arange(self.d_in, device=self.device).repeat(self.num_blocks)
            permutation = indices.gather(0, torch.randperm(len(indices), device=self.device))
            # we only need d_features from the indices
            permutations.append(permutation[:self.d_features])
        self.permutations.data = torch.stack(permutations, dim=0)
    
    def forward(self, x):
        # we first apply the lengthscale
        x = x / self.log_lengthscale.exp()

        if self.log_bias is not None:
            # we then append the bias
            x = torch.cat([(0.5 * self.log_bias).exp().repeat(len(x), 1), x], dim=-1)

        # We add the product dimension
        x = x.unsqueeze(1)
        # obtain (n, p, d) scaled inputs
        x = x * self.rad

        # allocate target tensor (may be moved to resampling step!)
        if self.complex_weights:
            # Y = torch.zeros(len(x), self.degree, self.num_blocks*self.d_in, device=self.device) \
            #     + 1j * torch.zeros(len(x), self.degree, self.num_blocks*self.d_in, device=self.device)
            # FWHT
            x.real = FastWalshHadamardTransform.apply(x.real)
            x.imag = FastWalshHadamardTransform.apply(x.imag)
        else:
            # Y = torch.zeros(len(x), self.degree, self.num_blocks*self.d_in, device=self.device)
            x = FastWalshHadamardTransform.apply(x)

        # Y.scatter_(
        #     dim=2,
        #     index=self.permutations.unsqueeze(0).expand(len(x), self.degree, self.num_blocks*self.d_in),
        #     src=x.repeat(1, 1, self.num_blocks)
        # )
        x = x.gather(2, self.permutations[None, ...].expand(len(x), self.degree, self.d_features))
        # x = x.repeat(1, 1, self.num_blocks)
        x = x.prod(dim=1)

        x = x * torch.exp(self.log_var / 2.)

        if self.complex_real:
            x = torch.cat([x.real, x.imag], dim=-1)

        return x / np.sqrt(self.d_features)

