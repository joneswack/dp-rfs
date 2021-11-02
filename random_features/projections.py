from numpy.core.shape_base import block
import torch
import numpy as np
import math
import time

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

    def __init__(self, d_in, d_features, sketch_type='sparse', complex_weights=False, full_complex=False, block_size=None):
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

        self.block_size = d_features if not block_size else block_size
        self.num_blocks = math.ceil(d_features / self.block_size)
        self.block_sizes = [
            self.block_size if i < (self.num_blocks-1) else (d_features - (self.num_blocks-1)*self.block_size)
            for i in range(self.num_blocks)
        ]

        self.i_hash = torch.nn.ParameterList(
            [torch.nn.Parameter(None, requires_grad=False) for _ in range(self.num_blocks)]
        )

        self.s_hash = torch.nn.ParameterList(
            [torch.nn.Parameter(None, requires_grad=False) for _ in range(self.num_blocks)]
        )

        self.P = torch.nn.Parameter(torch.zeros(self.d_features, self.d_in), requires_grad=False)
        if sketch_type == 'sparse':
            self.P_sparse = torch.nn.Parameter(torch.zeros(self.d_features, self.d_in).to_sparse(), requires_grad=False)

        # self.i_hash = torch.nn.Parameter(None, requires_grad=False)
        # self.s_hash = torch.nn.Parameter(None, requires_grad=False)
        # self.P = torch.nn.Parameter(None, requires_grad=False)

    def resample(self):
        with torch.no_grad():
            # tic = time.time()

            # bs = torch.tensor([1,2,3])
            # row = torch.tensor([1,2,3])
            # col = torch.tensor([1,2,3])

            # indices = torch.stack([bs, row, col], dim=0)

            # torch.sparse.FloatTensor(indices, torch.ones(3), torch.Size((4, 4, 4))).to_dense()

            # works!!

            if self.num_blocks > 1:
                # perm = torch.rand(self.num_blocks-1, self.block_size, self.d_in)
                # perm = perm.argsort(dim=1)

                # C = torch.zeros(self.num_blocks-1, self.block_size, self.d_in)
                # C[:, 0, :] = generate_rademacher_samples((self.num_blocks-1, self.d_in), complex_weights=self.complex_weights)
                # C = C.gather(1, perm)

                # self.P.data[:(self.num_blocks-1)*self.block_size] = C.reshape(-1, self.d_in)

                # alternative create (num_blocks, block_size) sparse tensor
                # seems to be fast for larger blocks
                self.i_hash.data = torch.randint(low=0, high=self.block_size, size=(self.num_blocks-1, self.d_in))
                self.s_hash.data = generate_rademacher_samples((self.num_blocks-1, self.d_in), complex_weights=self.complex_weights)
                # block_dim
                bs = torch.stack(self.d_in * [torch.arange(self.num_blocks-1)], dim=0).t().reshape(-1)
                row = self.i_hash.data.reshape(-1)
                col = torch.arange(self.d_in).repeat(self.num_blocks-1)
                values = self.s_hash.data.reshape(-1)
                indices = torch.stack([bs, row, col], dim=0)
                C = torch.sparse.FloatTensor(indices, values, torch.Size([self.num_blocks-1, self.block_size, self.d_in])).to_dense()

                self.P.data[:(self.num_blocks-1)*self.block_size] = C.reshape(-1, self.d_in)


            # last_perm = torch.rand(1, self.block_sizes[-1], self.d_in)
            # last_perm = last_perm.argsort(dim=1)

            #print('Perm creation time', time.time() - tic)

            # tic = time.time()

            # C_last = torch.zeros(1, self.block_sizes[-1], self.d_in)
            # C_last[0, 0, :] = generate_rademacher_samples((1, self.d_in,), complex_weights=self.complex_weights)
            # C_last = C_last.gather(1, last_perm)

            #print('Gather time', time.time() - tic)

            # self.P.data[(self.num_blocks-1)*self.block_size:] = C_last[0]
            
            # for i in range(self.num_blocks):
            #     # the permutation generation is too slow
            #     # perm = torch.vstack([torch.randperm(self.block_sizes[i]) for _ in range(self.d_in)]).t()
            #     # C = torch.zeros(self.block_sizes[i], self.d_in)
            #     start_index = i*self.block_size
            #     end_index = i*self.block_size+self.block_sizes[i]

            #     C[start_index] = generate_rademacher_samples((self.d_in,), complex_weights=self.complex_weights)
            #     # C[0] = generate_rademacher_samples((self.d_in,), complex_weights=self.complex_weights)

            #     if i == self.num_blocks-1:
            #         # last block
            #         C[start_index:end_index] = C[start_index:end_index].gather(0, last_perm)
            #     else:
            #         C[start_index:end_index] = C[start_index:end_index].gather(0, perm[i])
                
            #     self.P.data[start_index:end_index] = C[start_index:end_index]

            # for i in range(self.num_blocks):
            #     # self.i_hash.data = torch.randint(low=0, high=self.d_features, size=(self.d_in,))
            #     self.i_hash[i].data = torch.randint(low=0, high=self.block_sizes[i], size=(self.d_in,))
            #     self.P.data[i*self.block_size:i*self.block_size+self.block_sizes[i]] = 

            #     # this implementation is faster than x[:, self.permutations]
            #     x = x.gather(1, self.permutations[:, None].expand(*x.shape))

            #     if self.complex_weights:
            #         # self.i_hash.data = self.i_hash.data + 1j*torch.randint(low=0, high=self.d_features, size=(self.d_in,))
            #         self.i_hash[i].data = self.i_hash[i].data + 1j*torch.randint(low=0, high=self.block_sizes[i], size=(self.d_in,))

            #     self.s_hash[i].data = generate_rademacher_samples((self.d_in,), complex_weights=self.complex_weights)

            #     # TODO: adapt sparse and dense sketch to parameter list
            #     # create the countsketch matrix for subsketch i

            #     if self.sketch_type == 'sparse' or self.sketch_type == 'dense':
            #         # if we use sparse or dense sketch_type, the matrices can be precomputed
            #         row = self.i_hash[i]
            #         col = torch.arange(self.d_in)
            #         values = self.s_hash[i]
            #         indices = torch.stack([row, col], dim=0)
            #         self.P.data[i*self.block_size:i*self.block_size+self.block_sizes[i]] = torch.sparse.FloatTensor(
            #             indices, values, torch.Size([self.block_sizes[i], self.d_in])).to_dense()

            if self.sketch_type == 'sparse':
                self.P_sparse.data = self.P.data.to_sparse()

    def forward(self, x):
        # we convert x to a matrix
        original_shape = (*x.shape[:-1], self.d_features)
        x = x.reshape([-1, x.shape[-1]])
        
        if self.sketch_type == 'sparse':
            output = torch.sparse.mm(self.P_sparse, x.t()).t()
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

            for i in range(self.num_blocks):
                y = x * self.s_hash[i]
                start_index = i * self.block_size
                end_index = (i+1) * self.block_size

                if self.sketch_type == 'scatter':
                    # for scatter_add_ x and h need to have the same shape
                    # this might be a bit inefficient. scattering entire columns would be better
                    # BUT: scatter_add uses atomicAdd on cuda, on cpu it is a c++ loop
                    # CPU speedup is around x10 compared to a dense matrix vector product
                    if self.complex_weights:
                        if self.full_complex:
                            # we use different hashes
                            output[:, start_index:end_index, 0].scatter_add_(dim=-1, index=self.i_hash[i].real.expand(*y.shape).type(torch.int64), src=y.real)
                            output[:, start_index:end_index, 1].scatter_add_(dim=-1, index=self.i_hash[i].imag.expand(*y.shape).type(torch.int64), src=y.imag)
                        else:
                            # we use the same hash twice
                            output[:, start_index:end_index, 0].scatter_add_(dim=-1, index=self.i_hash[i].real.expand(*y.shape).type(torch.int64), src=y.real)
                            output[:, start_index:end_index, 1].scatter_add_(dim=-1, index=self.i_hash[i].real.expand(*y.shape).type(torch.int64), src=y.imag)
                    else:
                        output[:, start_index:end_index].scatter_add_(dim=-1, index=self.i_hash[i].expand(*y.shape), src=y)
                else:
                    output[:, start_index:end_index].index_add_(dim=-1, index=self.i_hash[i], source=y)

            if self.complex_weights:
                output = torch.view_as_complex(output)

        return output.reshape(original_shape) / np.sqrt(self.num_blocks)


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
