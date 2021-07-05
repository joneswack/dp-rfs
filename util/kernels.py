import numpy as np
import torch

def gaussian_kernel(data_x, data_y=None, lengthscale='auto'):
    if isinstance(lengthscale, str) and lengthscale == 'auto':
        # lengthscale = sqrt(d_in)
        lengthscale = np.sqrt(data_x.shape[1])
    if data_y is None:
        data_y = data_x.clone()

    data_x = data_x / lengthscale
    data_y = data_y / lengthscale
    distances = torch.cdist(data_x, data_y, p=2.0)

    return torch.exp(- distances**2 / 2.)

def polynomial_kernel(data_x, data_y=None, lengthscale='auto', k=2., c=0):
    if isinstance(lengthscale, str) and lengthscale == 'auto':
        # lengthscale = sqrt(d_in)
        lengthscale = np.sqrt(data_x.shape[1])
    if data_y is None:
        data_y = data_x.clone()

    data_x = data_x / lengthscale
    data_y = data_y / lengthscale
    # implement polynomial kernel and compare!
    return (data_x.mm(data_y.t()) + c)**k
