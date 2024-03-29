import torch
import numpy as np
import sys, os
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from random_features.projections import CountSketch, OSNAP, SRHT, RademacherTransform, GaussianTransform
from util.data import pad_data_pow_2


class SketchNode:
    """
    A node of the recursive sketch tree presented in: https://arxiv.org/pdf/1909.01410.pdf
    This is only used for hierarchical sketches.
    """

    def __init__(self, left, right, projection, d_features, sup_leaf=False):
        """
        left: Left child node
        right: Right child node
        projection: Projection object
        d_features: Projection dimension
        sup_leaf: Whether current node is a supplementary node to make #leafs a power of 2
        """
        self.left = left
        self.right = right
        self.projection = projection
        self.d_features = d_features
        self.sup_leaf = sup_leaf

    def _is_leaf(self):
        return (self.left is None) and (self.right is None)

    def recursive_resample(self):
        self.projection.resample()
        if self.left is not None:
            self.left.recursive_resample()
        if self.right is not None:
            self.right.recursive_resample()

    def recursive_cuda(self):
        self.projection.cuda()
        if self.left is not None:
            self.left.recursive_cuda()
        if self.right is not None:
            self.right.recursive_cuda()

    def join_children(self, data):
        """ Computes and joins two child sketches. """
        output = None

        for child in [self.left, self.right]:

            if child._is_leaf():
                if child.sup_leaf:
                    e1 = torch.zeros_like(data)
                    e1[:, 0] = 1.0
                    c1 = child.projection.forward(e1)
                else:
                    c1 = child.projection.forward(data)

                # if the child is a leaf, we do not combine its projection
                # moreover, we immediately return this projection
                # only if using additional countsketch
                return c1

            # combine both children if they are not the leaf nodes
            else:
                c1 = child.projection.forward(child.join_children(data))

            if isinstance(child.projection, CountSketch):
                # CountSketches need to be convolved
                c1 = torch.fft.rfft(c1)
                if output is None:
                    output = c1
                else:
                    return torch.fft.irfft(output * c1, n=self.d_features)
            else:
                if output is None:
                    output = c1
                else:
                    output = c1 * output / np.sqrt(self.d_features)
                    output[:, self.d_features:] = 0

                    return output

def construct_sketch_tree(node_projection, leaf_projection, degree, d_in, d_features, srht=False):
    """
    Constructs a binary tree composed of SketchNodes (only used for hierarchical sketches).

    node_projection (callable): The constructor of the base sketch of choice
    leaf_projection (callable): The constructor of the leaf sketch of choice
    degree: The degree of the polynomial kernel
    d_in: Data input dimension
    d_features: Projection dimension
    """

    q = int(2.**np.ceil(np.log2(degree))) # e.g. degree 5 -> 8
    q = max(q, 2) # even for degree 1, we construct a sketch tree

    if srht:
        # in the case of srht, we need to project to the power of 2
        # because the output sketch becomes an input sketch
        proj_dim = int(2**np.ceil(np.log2(d_features)))
    else:
        proj_dim = d_features

    # start with the leafs
    current_layer = [SketchNode(None, None, leaf_projection(d_in, proj_dim),
                d_features, sup_leaf=False) for _ in range(degree)]
    current_layer += [SketchNode(None, None, leaf_projection(d_in, proj_dim),
                d_features, sup_leaf=True) for _ in range(q - degree)]
    previous_layer = current_layer
    current_layer = [
        SketchNode(previous_layer[i], None,
                    node_projection(proj_dim, proj_dim),
                    d_features, sup_leaf=False)
        for i in range(0, len(previous_layer))
    ]

    # current_layer = [SketchNode(None, None, node_projection(d_in, proj_dim // downscale),
    #             d_features // downscale, sup_leaf=False, ctr=ctr) for _ in range(degree)]
    # current_layer += [SketchNode(None, None, node_projection(d_in, proj_dim // downscale),
    #             d_features // downscale, sup_leaf=True, ctr=ctr) for _ in range(q - degree)]

    # and go up layer by layer
    for _ in range(int(np.log2(q))):
        # we join an intermediate layer
        previous_layer = current_layer
        current_layer = [
            SketchNode(previous_layer[i], previous_layer[i+1],
                        node_projection(proj_dim, proj_dim),
                        d_features, sup_leaf=False)
            for i in range(0, len(previous_layer), 2)
        ]
    
    # return tree root
    return current_layer[0]


class AhleEtAl(torch.nn.Module):
    """
    Ahle et al.'s implementation
    """

    def __init__(self, projection, degree, d_in, d_features, tree=False, srht=False):
        super(AhleEtAl, self).__init__()
        
        # in the case of srht, we need to project to the power of 2
        # because the output sketch becomes an input sketch
        proj_dim = int(2**np.ceil(np.log2(d_features))) if srht else d_features
        
        self.base_projections = torch.nn.ModuleList(
            [projection(d_in, proj_dim) for _ in range(degree)]
        )
        
        self.node_projections = torch.nn.ModuleList(
            [projection(proj_dim, proj_dim) for _ in range(2*degree)]
        )
        
        self.degree = degree
        self.tree = tree
        self.srht = srht
        self.d_features = d_features
        
    def resample(self):
        for node in self.base_projections:
            node.resample()
        for node in self.node_projections:
            node.resample()
            
    def forward(self, x):
        bases = [proj.forward(x) for proj in self.base_projections]
        for base in bases:
            # for srht, we pad the output with zeros
            base[:, self.d_features:] = 0
        
        if self.tree:
            m = int(np.ceil(np.log2(len(bases))))
            for _ in range((1 << m) - len(bases)):
                bases.append(torch.ones_like(bases[0], device=bases[0].device))
            assert len(bases) == 1 << m
            
            r = 0
            for _ in range(m):
                new_bases = []
                for i in range(0, len(bases), 2):
                    s = bases[i] * bases[i+1]
                    s = self.node_projections[r].forward(s) / (self.d_features ** 0.5)
                    # for srht, we pad the output with zeros
                    s[:, self.d_features:] = 0
                    new_bases.append(s)
                    r += 1
                bases = new_bases
            assert len(bases) == 1
            sketch = bases[0]
        else:
            sketch = bases[0]
            for i in range(1, self.degree):
                sketch = sketch * bases[i]
                sketch = self.node_projections[i-1].forward(sketch) / (self.d_features ** 0.5)
                # for srht, we pad the output with zeros
                sketch[:, self.d_features:] = 0
                
        return sketch[:, :self.d_features] / np.sqrt(self.d_features)


class PolynomialSketch(torch.nn.Module):
    """
    The basic polynomial sketch (xTy/l^2 + b)^p with lengthscale l, bias b and degree p.
    """

    def __init__(self, d_in, d_features, degree=2, bias=0, lengthscale='auto', var=1.0, ard=False, trainable_kernel=False,
                    device='cpu', projection_type='countsketch_sparse', ahle=False, tree=False, complex_weights=False,
                    complex_real=False, full_cov=False, num_osnap_samples=10):
        """
        d_in: Data input dimension
        d_features: Projection dimension
        degree: The degree of the polynomial kernel
        bias: The bias b (eventually added through input modifiction)
        lengthscale: Downscale of the inputs (1 if none)
        var: Scale of the final kernel
        ard: Automatic Relevance Determination = individual lengthscale per input dimension
        trainable_kernel: Learnable bias, lengthscales, kernel variance
        projection_type: rademacher/gaussian/srht/countsketch_sparse/countsketch_dense/countsketch_scatter
        ahle: Whether to use the construction by Ahle et al. (overcomes exponential variances w.r.t. p but is not always better)
        tree: Whether to use Ahle et al. with a tree construction, otherwise sequential
        complex_weights: Whether to use complex-valued weights (almost always lower variances but more expensive)
        complex_real: Whether to use Complex-to-Real (CtR) sketches, the same as complex_weights but with a real transformation in the end
        num_osnap_samples: Only for projection_type='osnap' - Number of times each input coordinate is allocated to a random index (out of d_features)
        device: cpu or cuda
        """
        super(PolynomialSketch, self).__init__()
        self.d_in = d_in
        if complex_real:
            d_features = d_features // 2
        self.d_features = d_features
        self.degree = degree
        self.device = device
        self.projection_type = projection_type
        self.ahle = ahle
        self.tree = tree
        self.complex_weights = (complex_weights or complex_real)
        self.convolute_ts = True if self.projection_type.startswith('countsketch') else False
        self.complex_real = complex_real
        self.num_osnap_samples = num_osnap_samples

        # we initialize the kernel hyperparameters
        self.log_bias = None
        if bias != 0:
            self.log_bias = torch.nn.Parameter(torch.ones(1, device=device).float() * np.log(bias), requires_grad=trainable_kernel)
            self.d_in = self.d_in + 1
        
        if isinstance(lengthscale, str) and lengthscale == 'auto':
            # we use the same lengthscale heuristic as for the Gaussian kernel
            # alternatively, we can simply set it to one
            lengthscale = np.sqrt(d_in)

        num_lengthscales = d_in if ard else 1
        self.log_lengthscale = torch.nn.Parameter(torch.ones(num_lengthscales, device=device).float() * np.log(lengthscale), requires_grad=trainable_kernel)
        self.log_var = torch.nn.Parameter(torch.ones(1, device=device).float() * np.log(var), requires_grad=trainable_kernel)

        if projection_type == 'srht':
            node_projection = lambda d_in, d_out: SRHT(d_in, d_out, complex_weights=self.complex_weights, full_cov=full_cov, device=device)
        elif projection_type == 'rademacher':
            node_projection = lambda d_in, d_out: RademacherTransform(d_in, d_out, complex_weights=self.complex_weights, device=device)
        elif projection_type == 'gaussian':
            node_projection = lambda d_in, d_out: GaussianTransform(d_in, d_out, complex_weights=self.complex_weights, device=device)
        elif projection_type.split('_')[0] == 'countsketch':
            node_projection = lambda d_in, d_out: CountSketch(d_in, d_out, sketch_type=projection_type.split('_')[1],
                complex_weights=self.complex_weights, device=device)
        elif projection_type.split('_')[0] == 'osnap':
            node_projection = lambda d_in, d_out: OSNAP(d_in, d_out, s=num_osnap_samples, sketch_type=projection_type.split('_')[1],
                complex_weights=self.complex_weights, device=device)

        leaf_projection = lambda d_in, d_out: CountSketch(d_in, d_out, sketch_type='scatter',
                complex_weights=False, device=device)

        # the number of leaf nodes is p
        if self.ahle:
            # self.root = construct_sketch_tree(node_projection, leaf_projection, degree, self.d_in, d_features, srht=(projection_type=='srht'))
            self.sketch = AhleEtAl(node_projection, degree, self.d_in, d_features, tree=tree, srht=(projection_type=='srht'))
        else:
            self.sketch_list = torch.nn.ModuleList(
                [node_projection(self.d_in, self.d_features) for _ in range(degree)]
            )

    def resample(self):
        # seeding is handled globally!
        if self.ahle:
            self.sketch.resample()
        else:
            for node in self.sketch_list:
                node.resample()

    def plain_forward(self, x):
        # non-hierarchical polynomial sketch

        output = None

        for i, ls in enumerate(self.sketch_list):
            current_output = ls.forward(x)

            if self.convolute_ts:
                if self.complex_weights:
                    current_output = torch.fft.fft(current_output, n=self.d_features)
                else:
                    current_output = torch.fft.rfft(current_output, n=self.d_features)

                if (not isinstance(ls, CountSketch)) and (not isinstance(ls, OSNAP)):
                    # we need to scale down the sketch
                    current_output = current_output / np.sqrt(self.d_features)
            elif isinstance(ls, OSNAP) or isinstance(ls, CountSketch):
                # OSNAP and CountSketch without convolution require upscaling
                current_output = current_output * np.sqrt(self.d_features)
            
            if i == 0:
                output = current_output
            else:
                output = output * current_output

        if self.convolute_ts:
            if self.complex_weights:
                output = torch.fft.ifft(output, n=self.d_features)
            else:
                output = torch.fft.irfft(output, n=self.d_features)

        else:
            output = output / np.sqrt(self.d_features)
        
        return output

    def forward(self, x):
        # (hierarchical) random feature construction
        # if self.complex_weights and self.projection_type != 'srht':
        #     x = x.type(torch.complex64)

        # we first apply the lengthscale
        x = x / self.log_lengthscale.exp()

        if self.log_bias is not None:
            # we then append the bias
            x = torch.cat([(0.5 * self.log_bias).exp().repeat(len(x), 1), x], dim=-1)

        if self.ahle:
            #  recursive sketch
            # x = self.root.join_children(x)[:, :self.d_features]
            x = self.sketch.forward(x)
        else:
            # standard sketch
            x = self.plain_forward(x)

        x = x * torch.exp(self.log_var / 2.)

        if self.complex_real:
            x = torch.cat([x.real, x.imag], dim=-1)

        return x


### Launch this script directly in order to verify correct implementation of the sketches
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_gpu', dest='use_gpu', action='store_true')
    parser.set_defaults(use_gpu=False)

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_args()

    def reference_kernel(data, k, c, log_lengthscale='auto'):
        if isinstance(log_lengthscale, str) and log_lengthscale == 'auto':
            # lengthscale = sqrt(d_in)
            log_lengthscale = 0.5 * np.log(data.shape[1])

        data = data / np.exp(log_lengthscale)
        # implement polynomial kernel and compare!
        return (data.mm(data.t()) + c)**k

    torch.manual_seed(0)
    #data = torch.rand(100, 127)
    #data = data - data.mean(dim=0)
    # data, train_labels = torch.load('../datasets/export/cifar10/pytorch/train_cifar10_resnet34_final.pth')
    data, train_labels = torch.load('../datasets/export/mnist/pytorch/train_mnist.pth')
    # data, train_labels = torch.load('../datasets/export/fashion_mnist/pytorch/train_fashion_mnist.pth')
    # data, train_labels = torch.load('../datasets/export/eeg/pytorch/eeg.pth')
    data = data.view(len(data), -1)
    data = data - data.min()
    data = pad_data_pow_2(data)[:,:-1]
    
    #data = data - data.mean(dim=0)
    indices = torch.randint(len(data), (1000,))
    data = data[indices]

    data = data / data.norm(dim=1, keepdim=True)

    if args.use_gpu:
        data = data.cuda()

    degree = 7
    a = 2.
    bias = 1.-2./a**2
    lengthscale = a / np.sqrt(2.)
    complex_weights = True
    complex_real = True
    hierarchical = False
    projection_type = 'srht'

    ref_kernel = reference_kernel(data, degree, bias, log_lengthscale=np.log(lengthscale))

    # dims = [1024, 2048, 4096, 8192, 2*8192]
    # dims = [2*8000]
    # dims = [512 * i // 2 for i in range(1, 6)]
    up_dim = 2*32768
    down_dims = [2048, 4096, 6144, 8192, 10240]
    # dims = [20*1024]

    for D in down_dims:
        # D = D // 2
        scores = []
        for seed in np.arange(100):
            torch.manual_seed(seed)

            ts = PolynomialSketch(
                d_in=data.shape[1],
                d_features=up_dim, # D
                degree=degree,
                bias=bias,
                lengthscale=lengthscale,
                var = 1.,
                ard = False,
                trainable_kernel=False,
                projection_type=projection_type,
                hierarchical=hierarchical,
                complex_weights=complex_weights,
                complex_real=complex_real,
                device=('cuda' if args.use_gpu else 'cpu')
            )
            ts.resample()
            # features = tensorsketch(data, 2, 0, num_features=10000)
            # features = ts.forward(data)

            projection = ts.forward(data)

            ts = PolynomialSketch(
                d_in=up_dim,
                d_features=D, # D
                degree=1,
                bias=0,
                lengthscale=1.,
                var = 1.,
                ard = False,
                trainable_kernel=False,
                projection_type=projection_type,
                hierarchical=hierarchical,
                complex_weights=False,
                complex_real=False,
                device=('cuda' if args.use_gpu else 'cpu')
            )
            ts.resample()
            # features = tensorsketch(data, 2, 0, num_features=10000)
            # features = ts.forward(data)

            projection = ts.forward(projection)            

            approx_kernel = projection @ projection.conj().t()
            if approx_kernel.dtype in [torch.complex64, torch.complex128]: # torch.complex32
                approx_kernel = approx_kernel.real

            # score = torch.abs(approx_kernel - ref_kernel) / torch.abs(ref_kernel)
            score = (approx_kernel - ref_kernel).pow(2).sum().sqrt() / ref_kernel.pow(2).sum().sqrt()
            scores.append(score.item())
        print(np.array(scores).mean())

    print('Done!')
    exit()

    # Variance test to verify variance formulas
    n = 3
    d = 4
    D = 8
    n_points = 10
    data = torch.randn(n_points, d)
    ts = PolynomialSketch(
        d_in=d,
        d_features=D,
        degree=n,
        bias=0,
        lengthscale=1.,
        var = 1.,
        ard = False,
        trainable_kernel=False,
        projection_type='srht',
        hierarchical=False,
        complex_weights=True
    )

    norms_squared = (data**2).sum(dim=1).unsqueeze(1) * (data**2).sum(dim=1).unsqueeze(0)
    squared_dot_product = data**2 @ data.t()**2
    dot_product = data @ data.t()

    # regular
    # second_moment = norms_squared + 2.*(dot_product**2 - squared_dot_product)
    # cov_term = (dot_product**2 - (1./(d-1.)) * (norms_squared + dot_product**2 - 2*squared_dot_product))**n - dot_product**(2*n)

    # complex
    second_moment = norms_squared + dot_product**2 - squared_dot_product
    cov_term = (dot_product**2 - (1./(d-1.)) *(norms_squared - squared_dot_product))**n - dot_product**(2*n)

    variance_term = second_moment**n - dot_product**(2*n)
    # both prefactors are correct!
    # even better: denominator: D-1, enumerator: (K-1)*(d-1) + K*d-D-1
    # variance = 1./D * variance_term + (1-1./D) * ((d-1.)/(n_blocks*d-1.)) * cov_term
    # variance = 1./D * variance_term + ((n_blocks*d*(d-1)) / D**2) * cov_term
    variance = 1./D * variance_term + (np.floor(D/d)*d*(d-1) + (D%d)*(D%d-1)) / D**2 * cov_term

    app_kernel_values = []
    exact_kernel = dot_product**2
    
    for _ in range(10000):
        ts.resample()
        app_kernel_values.append(torch.stack(ts.approximate_kernel(data, return_complex=True), dim=-1))

    estimated_variance = torch.stack(app_kernel_values, dim=0).var(dim=0).sum(dim=-1)

    print('Estimated', estimated_variance)
    print('Exact', variance)
    print('Rel. difference', (estimated_variance - variance) / estimated_variance)

    print('Done!')
