"""
Code for CtR-TensorSRHT
-----------------------
This code is an excerpt of our full code repository and contains the implementation
of Complex-to-Real (CtR) as well as normal TensorSRHT to help understanding the contents
of the paper.
The full code will be made available after the final paper has been made public.
"""

import torch
import numpy as np
import math

# This library allows for fast multiplications with the hadamard matrix H using cuda
from hadamard_cuda.fwht import FastWalshHadamardTransform

def generate_rademacher_samples(shape, complex_weights=False, device='cpu'):
    """ Draws uniformly from the (complex) Rademacher distribution. """
    if complex_weights:
        support = torch.tensor([1j, -1j, 1, -1], dtype=torch.complex64, device=device)
    else:
        support = torch.tensor([1, -1], dtype=torch.float32, device=device)

    indices = torch.randint(len(support), shape)
    return support[indices]


class TensorSRHT(torch.nn.Module):

    def __init__(self, d_in, d_features, degree, nu=0, gamma=1, complex_real=False, device='cpu'):
        """
        We approximate the kernel (nu + gamma * x.T y) through TensorSRHT
        d_in: Data input dimension (needs to be a power of 2)
        d_features: Projection dimension
        degree: Degree of the polynomial kernel
        complex_real: Whether to use a CtR-sketch
        device: 'cpu' or 'cuda' (gpu)
        """
        super(TensorSRHT, self).__init__()
        self.d_in = d_in
        if complex_real:
            # we divide D by 2 to yield a D-dimensional feature map for the CtR case
            d_features = d_features // 2
        self.d_features = d_features
        self.degree = degree
        self.num_blocks = int(math.ceil(float(self.d_features)/float(self.d_in)))
        self.complex_real = complex_real
        self.device = device

        # we initialize the kernel hyperparameters
        self.nu = 0
        if nu != 0:
            self.nu = nu
            self.d_in = self.d_in + 1
        
        self.gamma = gamma

        # this parameter will contain the p rademacher vectors {d_1, ..., d_p}
        self.rad = torch.nn.Parameter(None, requires_grad=False)
        # this parameter will contain the p index vectors {p_1, ..., p_p}
        self.permutations = torch.nn.Parameter(None, requires_grad=False)

    def resample(self):
        """
        Resample the rademacher and permutation vectors.
        Needs to be called before projecting any input data.
        """

        # we generate p different Rademacher vectors {d_1, ..., d_p}
        self.rad.data = generate_rademacher_samples(
            (self.degree, self.d_in), complex_weights=self.complex_real, device=self.device)
        
        # we generate p index vectors {p_1, ..., p_p}
        
        # 1) we build the base vector (1,...,d) x ceil(D/d)
        indices = torch.arange(self.d_in, device=self.device).repeat(self.num_blocks)

        p_vectors = []

        for _ in range(self.degree):
            # 2) we create p shuffled instances of the base vector
            p_vectors.append(
                indices.gather(
                    0, torch.randperm(len(indices), device=self.device)
                )[:self.d_features]
            )

        self.permutations.data = torch.vstack(p_vectors)
    
    def forward(self, x):
        """
        Projects a matrix x of shape (n inputs, d dimensions)
        Returns feature map (n inputs, D outputs)
        """

        # we first apply the gamma scaling
        x = x * np.sqrt(self.gamma)

        if self.nu > 0:
            # we then append the bias
            x = torch.cat([(torch.ones(1, device=self.device)*np.sqrt(self.nu)).repeat(len(x), 1), x], dim=-1)

        # We add the product dimension
        x = x.unsqueeze(1)
        # obtain (n, p, d) scaled inputs D_i x for all i=1,...,p
        x = x * self.rad

        # apply the FWHT to the result
        if self.complex_real:
            # for complex x we need to project the real and imaginary parts
            x.real = FastWalshHadamardTransform.apply(x.real)
            x.imag = FastWalshHadamardTransform.apply(x.imag)
        else:
            # otherwise we only have a simple matrix product
            x = FastWalshHadamardTransform.apply(x)

        # we collect the indices of HD_i x for all i=1,...,p, i.e., we compute P_i H D_i x
        x = x.gather(2, self.permutations[None, ...].expand(len(x), self.degree, self.d_features))

        # finally we multiply over the degrees, i.e., P_1 H D_1 x * ... * P_p H D_p x
        x = x.prod(dim=1)

        if self.complex_real:
            # for the CtR case we concatenate the real and imaginary parts
            x = torch.cat([x.real, x.imag], dim=-1)

        return x / np.sqrt(self.d_features)

