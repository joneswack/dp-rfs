from numpy.core.shape_base import block
import torch
import numpy as np
import math
import time

from torch._C import device

from util.hadamard_cuda.fwht import FastWalshHadamardTransform


def generate_rademacher_samples(shape, complex_weights=False, device='cpu'):
    """ Draws uniformly from the (complex) Rademacher distribution. """
    if complex_weights:
        support = torch.tensor([1j, -1j, 1, -1], dtype=torch.complex64, device=device)
    else:
        support = torch.tensor([1, -1], dtype=torch.float32, device=device)
    samples = torch.index_select(support, 0, torch.randint(len(support), shape, device=device).view(-1))
    return samples.reshape(shape)
    #indices = torch.randint(len(support), shape)
    #return support[indices]


class CountSketch(torch.nn.Module):
    """ Computes the CountSketch Cx that can take advantage of data sparsity. """

    def __init__(self, d_in, d_features, sketch_type='sparse', complex_weights=False, full_complex=False, device='cpu'):
        """
        d_in: Data input dimension
        d_features: Projection dimension
        sketch_type: Type of numerical projection (sparse/dense/scatter)
        complex_weights: Whether to use complex Rademacher vectors for the sign sketch
        full_complex: Whether to also use complex index hashes
        block_size: If not None, sub block size < D for D/sub block_size CountSketches
            (currently not implemented)
        We recommend scatter_add for dense inputs and sparse for sparse inputs.
        Complex projections are not supported for this sketch.
        """
        super(CountSketch, self).__init__()

        self.d_in = d_in
        self.d_features = d_features
        self.sketch_type = sketch_type
        self.complex_weights = complex_weights
        self.full_complex = full_complex
        self.device = device

        self.i_hash = torch.nn.Parameter(None, requires_grad=False)

        self.s_hash = torch.nn.Parameter(None, requires_grad=False)

        dtype = torch.complex64 if self.complex_weights else torch.float32

        if sketch_type == 'sparse':
            self.P = torch.nn.Parameter(
                torch.zeros(self.d_features, self.d_in, dtype=dtype, device=device).to_sparse(),
                requires_grad=False)
        elif sketch_type == 'dense':
            self.P = torch.nn.Parameter(
                torch.zeros(self.d_features, self.d_in, dtype=dtype, device=device),
                requires_grad=False)

    def resample(self):
        with torch.no_grad():
            self.i_hash.data = torch.randint(low=0, high=self.d_features, size=(self.d_in,), device=self.device)

            if self.complex_weights:
                self.i_hash.data = self.i_hash.data \
                    + 1j*torch.randint(low=0, high=self.d_features, size=(self.d_in,), device=self.device)

            self.s_hash.data = generate_rademacher_samples((self.d_in,), complex_weights=self.complex_weights, device=self.device)

            if self.sketch_type == 'sparse' or self.sketch_type == 'dense':
                # if we use sparse or dense sketch_type, the matrices can be precomputed
                row = self.i_hash
                col = torch.arange(self.d_in, device=self.device)
                values = self.s_hash
                indices = torch.stack([row, col], dim=0)
                C = torch.sparse.FloatTensor(indices, values, torch.Size([self.d_features, self.d_in]))
                if self.sketch_type == 'dense':
                    C = C.to_dense()
                self.P.data = C

    def forward(self, x):
        # we convert x to a matrix
        original_shape = (*x.shape[:-1], self.d_features)
        x = x.reshape([-1, x.shape[-1]])
        
        if self.sketch_type == 'sparse':
            output = torch.sparse.mm(self.P, x.t()).t()
        elif self.sketch_type == 'dense':
            output = torch.mm(x, self.P.t())
        else:
            if self.complex_weights:
                output = torch.zeros(x.shape[0], self.d_features, 2, device=self.device).float()
            else:
                output = torch.zeros(x.shape[0], self.d_features, device=self.device).float()
            if x.is_cuda:
                output = output.cuda()

            y = x * self.s_hash

            if self.sketch_type == 'scatter':
                # for scatter_add_ x and h need to have the same shape
                # this might be a bit inefficient. scattering entire columns would be better
                # BUT: scatter_add uses atomicAdd on cuda, on cpu it is a c++ loop
                # CPU speedup is around x10 compared to a dense matrix vector product
                if self.complex_weights:
                    if self.full_complex:
                        # we use different hashes
                        output[..., 0].scatter_add_(dim=-1, index=self.i_hash.real.expand(*y.shape).type(torch.int64), src=y.real)
                        output[..., 1].scatter_add_(dim=-1, index=self.i_hash.imag.expand(*y.shape).type(torch.int64), src=y.imag)
                    else:
                        # we use the same hash twice
                        output[..., 0].scatter_add_(dim=-1, index=self.i_hash.real.expand(*y.shape).type(torch.int64), src=y.real)
                        output[..., 1].scatter_add_(dim=-1, index=self.i_hash.real.expand(*y.shape).type(torch.int64), src=y.imag)
                else:
                    output.scatter_add_(dim=-1, index=self.i_hash.expand(*y.shape), src=y)
            else:
                output.index_add_(dim=-1, index=self.i_hash, source=y)

            if self.complex_weights:
                output = torch.view_as_complex(output)

        return output.reshape(original_shape)

class OSNAP(torch.nn.Module):
    """ Computes the OSNAP Cx that can take advantage of data sparsity. """

    def __init__(self, d_in, d_features, s=1., sketch_type='sparse', complex_weights=False, full_complex=False, device='cpu'):
        """
        d_in: Data input dimension
        d_features: Projection dimension
        s: Number of non-zero entries per column of C
        sketch_type: Type of numerical projection (sparse/dense/scatter)
        complex_weights: Whether to use complex Rademacher vectors for the sign sketch
        full_complex: Whether to also use complex index hashes
        We recommend scatter_add for dense inputs and sparse for sparse inputs.
        Complex projections are not supported for this sketch.
        """
        super(OSNAP, self).__init__()

        self.d_in = d_in
        self.d_features = d_features
        self.s = s
        self.sketch_type = sketch_type
        self.complex_weights = complex_weights
        self.full_complex = full_complex
        self.device = device

        self.dtype = torch.complex64 if self.complex_weights else torch.float32

        if sketch_type == 'sparse':
            self.P = torch.nn.Parameter(
                torch.zeros(self.d_features, self.d_in, dtype=self.dtype, device=device).to_sparse(),
                requires_grad=False)
        else:
            self.P = torch.nn.Parameter(
                torch.zeros(self.d_features, self.d_in, dtype=self.dtype, device=device), requires_grad=False)

    def resample(self):
        with torch.no_grad():
            # random column-wise shuffling

            #tic = time.time()
            # argsort is slow
            permutation = torch.rand(self.d_features, self.d_in, device=self.device)
            permutation = torch.argsort(permutation, dim=0)
            #print('Perm creation time', time.time() - tic)

            C = torch.zeros(self.d_features, self.d_in, dtype=self.dtype, device=self.device)
            C[:self.s] = generate_rademacher_samples((self.s, self.d_in), complex_weights=self.complex_weights, device=self.device)
            C = C / np.sqrt(self.s)

            #tic = time.time()
            C = C[permutation, torch.arange(self.d_in, device=self.device)]
            #print('Permuting time', time.time() - tic)

            if self.sketch_type == 'sparse':
                C = C.to_sparse()
            
            self.P.data = C

    def forward(self, x):
        # we convert x to a matrix
        original_shape = (*x.shape[:-1], self.d_features)
        x = x.reshape([-1, x.shape[-1]])
        
        if self.sketch_type == 'sparse':
            output = torch.sparse.mm(self.P, x.t()).t()
        elif self.sketch_type == 'dense':
            output = torch.mm(x, self.P.t())

        return output.reshape(original_shape)


class SRHT(torch.nn.Module):
    """ Subsampled randomized Hadamard transform (SRHT) HDx. """

    def __init__(self, d_in, d_features, complex_weights=False, shuffle=True, k=1, full_cov=False, device='cpu'):
        """
        d_in: Data input dimension (needs to be a power of 2)
        d_features: Projection dimension
        complex_weights: Whether to use complex-valued projections
        shuffle: Whether to shuffle the projection rows per block (not needed for SORF)
        k: Number of subsequent projections (k=3 is used for RFF/SORF), otherwise not needed
            possible only when complex_weights=False
        full_cov: Whether to use correlated samples across projection blocks (faster)
        """
        super(SRHT, self).__init__()
        self.d_in = d_in
        self.d_features = d_features
        self.num_blocks = int(math.ceil(float(self.d_features)/float(self.d_in)))
        self.complex_weights = complex_weights
        self.shuffle = shuffle
        self.k = k
        self.full_cov = full_cov
        self.device = device

        self.rad = torch.nn.Parameter(None, requires_grad=False)
        self.permutations = torch.nn.Parameter(None, requires_grad=False)

    def resample(self):
        if self.full_cov:
            # we copy D_i for different blocks
            self.rad.data = generate_rademacher_samples(
                (self.k, 1, self.d_in), complex_weights=self.complex_weights, device=self.device)
            # generate an index permutation over B*d
            indices = torch.arange(self.d_in, device=self.device).repeat(self.num_blocks)
            self.permutations.data = indices.gather(
                0, torch.randperm(len(indices), device=self.device)
            )[:self.d_features]
        else:
            self.rad.data = generate_rademacher_samples(
                (self.k, self.num_blocks, self.d_in), complex_weights=self.complex_weights, device=self.device)
            # generate an index permutation per block
            self.permutations.data = torch.cat(
                [i*self.d_in + torch.randperm(self.d_in, device=self.device) for i in range(self.num_blocks)]
            , dim=0)[:self.d_features]
    
    def forward(self, x):
        # number of independent projections
        n_projections = self.rad.shape[1]

        x = x.unsqueeze(1).expand(x.shape[0], n_projections, self.d_in)

        for i in range(self.k):
            x = x * self.rad[i]
            
            if self.complex_weights:
                x.real = FastWalshHadamardTransform.apply(x.real)
                x.imag = FastWalshHadamardTransform.apply(x.imag)
            else:
                x = FastWalshHadamardTransform.apply(x)

            if i < (self.k-1):
                x = x / np.sqrt(self.d_in)

        x = x.view(-1, n_projections*self.d_in)

        if self.shuffle or self.full_cov:
            x = x.gather(1, self.permutations[None, :].expand(len(x), self.d_features))

        return x[:, :self.d_features]

class RademacherTransform(torch.nn.Module):
    def __init__(self, d_in, d_features, complex_weights=False, device='cpu'):
        """
        d_in: Data input dimension
        d_features: Projection dimension
        complex_weights: Whether to use complex-valued projections
        """
        super(RademacherTransform, self).__init__()
        self.d_in = d_in
        self.d_features = d_features
        self.complex_weights = complex_weights
        self.weights = torch.nn.Parameter(None, requires_grad=False)
        self.device = device

    def resample(self):
        self.weights.data = generate_rademacher_samples(
            (self.d_in, self.d_features), complex_weights=self.complex_weights, device=self.device)

    def forward(self, x):
        if self.complex_weights:
            x = x.type(torch.complex64)
        return torch.matmul(x, self.weights)


class GaussianTransform(torch.nn.Module):
    def __init__(self, d_in, d_features, complex_weights=False, device='cpu'):
        """
        d_in: Data input dimension
        d_features: Projection dimension
        complex_weights: Whether to use complex-valued projections
        """
        super(GaussianTransform, self).__init__()
        self.d_in = d_in
        self.d_features = d_features
        self.complex_weights = complex_weights
        self.weights = torch.nn.Parameter(None, requires_grad=False)
        self.device = device

    def resample(self):
        dtype = torch.cfloat if self.complex_weights else torch.float
        # the complex data type automatically uses the complex std normal CN(0, 1)
        # this implies var(weight.real) = var(weight.imag) = 0.5
        self.weights.data = torch.randn(self.d_in, self.d_features, dtype=dtype, device=self.device)

    def forward(self, x):
        if self.complex_weights:
            x = x.type(torch.complex64)
        return torch.matmul(x, self.weights)
