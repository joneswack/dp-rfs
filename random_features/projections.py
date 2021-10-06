import torch
import numpy as np
import math

#from torch._C import dtype, float32

if int(torch.__version__.split('.')[1]) > 1:
    # if number after first dot is larger than 1, use the new library
    from util.fwht.fwht import FastWalshHadamardTransform
else:
    from util.fwht_old.fwht import FastWalshHadamardTransform


def generate_rademacher_samples(shape, complex_weights=False):
    """ Draws uniformly from the (complex) Rademacher distribution. """
    if complex_weights:
        support = torch.tensor([1j, -1j, 1, -1], dtype=torch.complex64)
    else:
        support = torch.tensor([1, -1], dtype=torch.float32)
    #samples = torch.index_select(support, 0, torch.randint(len(support), shape).view(-1))
    #return samples.reshape(shape)
    indices = torch.randint(len(support), shape)
    return support[indices]


class CountSketch(torch.nn.Module):
    """ Computes the CountSketch Cx that can take advantage of data sparsity. """

    def __init__(self, d_in, d_features, sketch_type='sparse', complex_weights=False, full_complex=False):
        """
        d_in: Data input dimension
        d_features: Projection dimension
        sketch_type: Type of numerical projection (sparse/dense/scatter)
        We recommend scatter_add for dense inputs and sparse for sparse inputs.
        Complex projections are not supported for this sketch.
        """
        super(CountSketch, self).__init__()

        self.d_in = d_in
        self.d_features = d_features
        self.sketch_type = sketch_type
        self.complex_weights = complex_weights
        self.full_complex = full_complex

        self.i_hash = torch.nn.Parameter(None, requires_grad=False)
        self.s_hash = torch.nn.Parameter(None, requires_grad=False)
        self.P = torch.nn.Parameter(None, requires_grad=False)

    def resample(self):
        if self.complex_weights:
            self.i_hash.data = self.i_hash.data + 1j*torch.randint(low=0, high=self.d_features, size=(self.d_in,))
        else:
            self.i_hash.data = torch.randint(low=0, high=self.d_features, size=(self.d_in,))
        self.s_hash.data = generate_rademacher_samples((self.d_in,), complex_weights=self.complex_weights)

        if self.sketch_type == 'sparse' or self.sketch_type == 'dense':
            # if we use sparse or dense sketch_type, the matrices can be precomputed
            row = self.i_hash
            col = torch.arange(self.d_in)
            values = self.s_hash
            indices = torch.stack([row, col], dim=0)

            self.P.data = torch.sparse.FloatTensor(indices, values, torch.Size([self.d_features, self.d_in]))
            if self.sketch_type == 'dense':
                self.P.data = self.P.data.to_dense()

    def forward(self, x):
        # we convert x to a matrix
        original_shape = (*x.shape[:-1], self.d_features)
        x = x.reshape([-1, x.shape[-1]])
        
        if self.sketch_type == 'sparse':
            output = torch.sparse.mm(self.P, x.t()).t()
        elif self.sketch_type == 'dense':
            output = torch.mm(x, self.P.t())
        else:
            dtype = torch.complex64 if self.complex_weights else torch.float32
            # output = torch.zeros(x.shape[0], self.d_features).type(dtype)
            if self.complex_weights:
                output = torch.zeros(x.shape[0], self.d_features, 2).type(torch.float32)
            else:
                output = torch.zeros(x.shape[0], self.d_features).type(torch.float32)
            if x.is_cuda:
                output = output.cuda()
            x = x * self.s_hash

            if self.sketch_type == 'scatter':
                # for scatter_add_ x and h need to have the same shape
                # this might be a bit inefficient. scattering entire columns would be better
                # BUT: scatter_add uses atomicAdd on cuda, on cpu it is a c++ loop
                # CPU speedup is around x10 compared to a dense matrix vector product
                if self.complex_weights:
                    if self.full_complex:
                        # we use different hashes
                        output[..., 0].scatter_add_(dim=-1, index=self.i_hash.real.expand(*x.shape).type(torch.int64), src=x.real)
                        output[..., 1].scatter_add_(dim=-1, index=self.i_hash.imag.expand(*x.shape).type(torch.int64), src=x.imag)
                    else:
                        # we use the same hash twice
                        output[..., 0].scatter_add_(dim=-1, index=self.i_hash.real.expand(*x.shape).type(torch.int64), src=x.real)
                        output[..., 1].scatter_add_(dim=-1, index=self.i_hash.real.expand(*x.shape).type(torch.int64), src=x.imag)
                    output = torch.view_as_complex(output)
                else:
                    output.scatter_add_(dim=-1, index=self.i_hash.expand(*x.shape), src=x)
            else:
                output.index_add_(dim=-1, index=self.i_hash, source=x)

        return output.reshape(original_shape)


class SRHT(torch.nn.Module):
    """ Subsampled randomized Hadamard transform (SRHT) HDx. """

    def __init__(self, d_in, d_features, complex_weights=False, shuffle=True, k=1, full_cov=False):
        """
        d_in: Data input dimension (needs to be a power of 2)
        d_features: Projection dimension
        complex_weights: Whether to use complex-valued projections
        shuffle: Whether to shuffle the projection rows per block (not needed for SORF)
        k: Number of subsequent projections (k=3 is used for SORF), otherwise not needed
            possible only when complex_weights=False
        """
        super(SRHT, self).__init__()
        self.d_in = d_in
        self.d_features = d_features
        self.num_blocks = int(math.ceil(float(self.d_features)/float(self.d_in)))
        self.complex_weights = complex_weights
        self.shuffle = shuffle
        self.k = k
        self.full_cov = full_cov

        self.rad = torch.nn.Parameter(None, requires_grad=False)
        self.permutations = torch.nn.Parameter(None, requires_grad=False)

    def resample(self):
        if self.full_cov:
            # we copy D_i for different blocks
            self.rad.data = generate_rademacher_samples(
                (self.k, 1, self.d_in), complex_weights=self.complex_weights)
            self.rad.data = self.rad.data.expand(self.k, self.num_blocks, self.d_in)
            # generate an index permutation over B*d
            self.permutations.data = torch.randperm(self.num_blocks * self.d_in)
        else:
            self.rad.data = generate_rademacher_samples(
                (self.k, self.num_blocks, self.d_in), complex_weights=self.complex_weights)
            # generate an index permutation per block
            self.permutations.data = torch.cat(
                [i*self.d_in + torch.randperm(self.d_in) for i in range(self.num_blocks)], dim=0)
    
    def forward(self, x):
        # we convert x to a matrix
        original_shape = (*x.shape[:-1], self.d_features)
        x = x.reshape([-1, x.shape[-1]])
        # we copy x num_block times
        x = x.unsqueeze(1).expand(x.shape[0], self.num_blocks, x.shape[1])

        # move input to CPU first because CUDA FWHT is buggy for large D
        cpu_mode = x.shape[2] > 4096 and x.is_cuda

        for i in range(self.k):
            x = x * self.rad[i]

            if cpu_mode:
                x = x.cpu()
            
            if self.complex_weights:
                x.real = FastWalshHadamardTransform.apply(x.real)
                x.imag = FastWalshHadamardTransform.apply(x.imag)
            else:
                x = FastWalshHadamardTransform.apply(x)
            
            if cpu_mode:
                x = x.cuda()

            if i < (self.k-1):
                x = x / np.sqrt(self.d_in)
    
        # obtain one large projection tensor
        x = x.view(-1, self.num_blocks*self.d_in)
    
        if self.shuffle:
            # this implementation is faster than x[:, self.permutations]
            x = x.gather(1, self.permutations[None, :].expand(*x.shape))

        # keep d_features and convert x back to old shape
        return x[:, :self.d_features].reshape(original_shape)

class RademacherTransform(torch.nn.Module):
    def __init__(self, d_in, d_features, complex_weights=False):
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

    def resample(self):
        self.weights.data = generate_rademacher_samples(
            (self.d_in, self.d_features), complex_weights=self.complex_weights)

    def forward(self, x):
        return torch.matmul(x, self.weights)


class GaussianTransform(torch.nn.Module):
    def __init__(self, d_in, d_features, complex_weights=False):
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

    def resample(self):
        dtype = torch.cfloat if self.complex_weights else torch.float
        # the complex data type automatically uses the complex std normal CN(0, 1)
        # this implies var(weight.real) = var(weight.imag) = 0.5
        self.weights.data = torch.randn(self.d_in, self.d_features, dtype=dtype)

    def forward(self, x):
        return torch.matmul(x, self.weights)
