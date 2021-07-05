import torch
import numpy as np

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import util.data
from random_features.projections import SRHT, GaussianTransform


class RFF(torch.nn.Module):
    """ Classical random Fourier features (optional SORF) """
    def __init__(self, d_in, d_features, lengthscale='auto', trainable_kernel=False,
                    dtype=torch.FloatTensor, complex_weights=False, projection_type='srht'):
        super(RFF, self).__init__()

        self.d_in = d_in
        self.d_features = d_features
        if not complex_weights:
            self.d_features = self.d_features // 2
        self.dtype = dtype
        self.complex_weights = complex_weights
        self.projection_type = projection_type

        if isinstance(lengthscale, str) and lengthscale == 'auto':
            lengthscale = np.sqrt(d_in)
        # we activate ARD
        self.log_lengthscale = torch.nn.Parameter(
            torch.ones(d_in).type(dtype) * np.log(lengthscale), requires_grad=trainable_kernel)

        if self.projection_type == 'srht':
            self.feature_encoder = SRHT(self.d_in, self.d_features,
                                        complex_weights=False, shuffle=False, k=3)
        else:
            self.feature_encoder = GaussianTransform(self.d_in, self.d_features, complex_weights=False)

    def resample(self):
        self.feature_encoder.resample()
    
    def forward(self, x):
        x = x / self.log_lengthscale.exp()

        x = self.feature_encoder.forward(x)

        x = torch.stack([torch.cos(x), torch.sin(x)], dim=-1)
        if self.complex_weights:
            x = torch.view_as_complex(x)
        else:
            x = x.view(len(x), -1)
        
        x = x / np.sqrt(self.d_features)

        return x

    def reference_kernel(self, data_x, data_y=None):
        """
        Reference RBF kernel.
        """
        if data_y is None:
            data_y = data_x.clone()

        data_x = data_x / self.log_lengthscale.exp()
        data_y = data_y / self.log_lengthscale.exp()
        distances = torch.cdist(data_x, data_y, p=2.0)

        return torch.exp(-distances**2 / 2.)


if __name__ == "__main__":
    complex_weights = True
    projection_type = 'gaussian'
    # data = torch.randn(128, 2)
    # print('Loading dataset: {}'.format(args.dataset_config))
    data = util.data.load_dataset('config/datasets/mnist.json', standardize=True, normalize=False)
    data_name, train_data, test_data, train_labels, test_labels = data

    data = train_data[torch.randperm(len(train_data))][:1000]
    data = util.data.pad_data_pow_2(data)
    lengthscale = torch.cdist(data, data, p=2.0).median()

    for d_features in [1024, 2048, 4096, 8192]:
        ts = RFF(
            data.shape[1], d_features,
            lengthscale=lengthscale, trainable_kernel=False,
            dtype=torch.FloatTensor, complex_weights=complex_weights, projection_type=projection_type)
        ts.resample()
        features = ts.forward(data)

        ref_kernel = ts.reference_kernel(data)

        approx_kernel = features @ features.conj().t()

        if complex_weights:
            approx_kernel = approx_kernel.real

        # score = torch.abs(approx_kernel - ref_kernel) / torch.abs(ref_kernel)
        mse = (ref_kernel - approx_kernel).pow(2).mean().item()
        print(d_features, mse)

    print('Done!')

