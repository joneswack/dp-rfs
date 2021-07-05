import numpy as np
import scipy.special
import torch

from mpmath import fp, mp, mpf

"""Confluent hypergeometric limit function 0F1."""

class HypGeomInt(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, d_in):
        input = input.detach()
        ctx.save_for_backward(input)
        ctx.d_in = d_in

        # hyp0f1_vec = np.vectorize(fp.hyp0f1)
        # output = hyp0f1_vec(d_in/2., (-0.25 * input.numpy()**2).reshape(-1)) \
        #     .reshape(input.shape)
        
        if d_in > 128:
            # scipy becomes numerically unstable
            # in the limit 0F1(;d/2;-0.25*(wx)^2) becomes exp(-(wx)^2/(2d))
            output = np.exp(-input.numpy()**2 / (2*d_in))
        else:
            output = scipy.special.hyp0f1(d_in/2., -0.25 * input.numpy()**2)
        
        return torch.as_tensor(output, dtype=input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        d_in = ctx.d_in

        # hyp0f1_vec = np.vectorize(fp.hyp0f1)
        # output = hyp0f1_vec(d_in/2., (-0.25 * input.numpy()**2).reshape(-1)) \
        #     .reshape(input.shape)
        # dydx = -0.5 * input.numpy() / (d_in / 2.) * output

        if d_in > 128:
            # scipy becomes numerically unstable
            # in the limit 0F1(;d/2;-0.25*(wx)^2) becomes exp(-(wx)^2/(2d))
            # h0f1 captures the outer derivative which is equal to the function itself
            h0f1 = np.exp(-input.numpy()**2 / (2*d_in))
        else:
            h0f1 = scipy.special.hyp0f1(d_in/2.+1., -0.25 * input.numpy()**2)
        
        dydx = -input.numpy() / d_in * h0f1
        
        # we return as many gradients as there were arguments
        return grad_output * torch.from_numpy(dydx).to(input.dtype), None
