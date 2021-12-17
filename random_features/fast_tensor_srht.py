import torch
import numpy as np
import math
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from random_features.projections import generate_rademacher_samples
from util.hadamard_cuda.fwht import FastWalshHadamardTransform

N = 1000
d = 1024
D = 2048
num_blocks = int(math.ceil(float(D)/float(d)))
p = 3
complex_weights = True
device = 'cpu'

X = torch.randn(N, d) / np.sqrt(d)

# We add the product dimension
X = X.unsqueeze(1)

# generate (p, d) Rademacher samples
rad = generate_rademacher_samples((p, d), complex_weights=complex_weights, device=device)
# generate p index permutations over B*d
permutations = []
for _ in range(p):
    indices = torch.arange(d, device=device).repeat(num_blocks)
    permutation = indices.gather(0, torch.randperm(len(indices), device=device))
    permutations.append(permutation)
permutations = torch.stack(permutations, dim=0)
# allocate target tensor
Y = torch.zeros(N, p, num_blocks*d, device=device) + 1j * torch.zeros(N, p, num_blocks*d, device=device)

# obtain (n, p, d) scaled inputs
X = X * rad

# FWHT
X.real = FastWalshHadamardTransform.apply(X.real)
X.imag = FastWalshHadamardTransform.apply(X.imag)

# Scatter multiply the result
# Y = torch.gather(X, dim=2, index=permutations.unsqueeze(0).expand(len(X), p, num_blocks*d)) # , reduce='multiply'
Y.scatter_(dim=2, index=permutations.unsqueeze(0).expand(len(X), p, num_blocks*d), src=X.repeat(1, 1, num_blocks))
Y = Y.prod(dim=1)
print('Done!')

# keep d_features and convert x back to old shape
# return Y[:, :D]