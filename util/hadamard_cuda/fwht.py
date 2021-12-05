import torch
from fwht_py_new import hadamard_transform

class FastWalshHadamardTransform(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u):
        original_shape = list(u.shape)
        # u = u.contiguous().view(-1, original_shape[-1])
        results = hadamard_transform(u.view(-1, original_shape[-1]))
        results = results.view(*original_shape)
        return results
    @staticmethod
    def backward(ctx, grad):
        original_shape = list(grad.shape)
        # grad = grad.view(-1, original_shape[-1]).contiguous()
        grad_out = FastWalshHadamardTransform.apply(grad.view(-1, original_shape[-1]))
        # grad_out = forward(grad)
        grad_out = grad_out.view(*original_shape)
        return grad_out