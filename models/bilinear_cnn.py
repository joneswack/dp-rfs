import numpy as np

import torch
import torch.nn as nn

def extract_random_patches(conv_features, num_samples):
    with torch.no_grad():
        # the conv features have shape (samples x channels x height x width)
        spatial_dim = conv_features.shape[2] * conv_features.shape[3]
        num_patches = len(conv_features) * spatial_dim
        indices = np.random.randint(0, high=num_patches, size=num_samples)

        conv_features = torch.nn.functional.relu(conv_features)
        conv_features = conv_features.permute(0, 2, 3, 1)
        # treat conv output as huge mini batch
        conv_features = conv_features.reshape(-1, conv_features.shape[-1])
        # we subsample the patches
        return conv_features[indices]


class CNNKernelPooling(nn.Module):
    def __init__(self, model, D=10240, n_classes=10,
                    estimate_lengthscale=True, feature_encoder=None,
                    sqrt=True, norm=True, finetuning=False, device='cpu'):
        """
        estimate_lengthscale: whether to estimate the lengthscale on a data subsample
        """
        super(CNNKernelPooling, self).__init__()
        
        self.estimate_lengthscale = estimate_lengthscale
        self.sqrt = sqrt
        self.norm = norm
        self.finetuning = finetuning
        self.D = D

        # last conv activation (drop last MaxPool2d)
        # self.features = model.features[:-1]
        self.features = torch.nn.Sequential(*list(model.features.children())[:-2]).to(device)

        self.relu5_3 = torch.nn.ReLU(inplace=False).to(device)

        self.pool_sketch = feature_encoder

        self.classifier = nn.Linear(D, n_classes).to(device) # 512
        # self.classifier = torch.nn.Linear(in_features=512 * 512, out_features=n_classes, bias=True)

        # may not be necessary!
        torch.nn.init.constant_(self.classifier.bias, val=0.0).to(device)

    def estimate_lengthscale_and_features(self, random_patches):
        """
        x: conv_features
        """
        with torch.no_grad():
            if self.estimate_lengthscale:
                log_lengthscale = torch.cdist(random_patches, random_patches).median().log()
                self.pool_sketch.log_lengthscale.data = log_lengthscale * \
                    torch.ones_like(self.pool_sketch.log_lengthscale.data)

    def extract_pool_features(self, x):
        """
        Extract last conv features.
        Return: Tensor of size (n x D), D ~ 12K
        """

        x = self.features(x)

        return x
    
    def forward(self, x, finetuning=False):
        # https://github.com/HaoMood/blinear-cnn-faster/blob/master/src/model.py
        # for orientation
        bs = len(x)

        if finetuning:
            x = self.extract_pool_features(x)

        # we drop the last channel for srht
        # if self.pool_sketch.projection_type == 'srht' and self.pool_sketch.log_bias is not None:
            # x = x[:, :-1, :, :]
            # x = self.relu5_3(x[:, :-1, :, :])
        # else:
            # pass
            # x = self.relu5_3(x)
            # for fairness, we provide the same data as for srht
        x = self.relu5_3(x[:, :-1, :, :])

        # x = torch.reshape(x, (bs, 512, 28 * 28))

        # tic = time.time()
        # x = torch.bmm(x, torch.transpose(x, 1, 2)) / (28 * 28)
        # print(time.time() - tic)

        # x = torch.reshape(x, (bs, 512 * 512))

        # move channels to the back
        x = x.permute(0, 2, 3, 1)
        # print(x.shape)
        # treat conv output as huge mini batch
        x = x.reshape(-1, x.shape[-1])
        # construct sketch vector (H01)
        # x = torch.cat([x, self.pool_sketch(x)], dim=-1)
        # only required, if model has own lengthscale parameter (already done for feature encoder)
        # x = x / self.log_lengthscale.exp()
        # print(x.shape)

        x = self.pool_sketch(x)
        # print(x.shape)
        # print('Poolsketch time: {}'.format(time.time() - tic))
        
        # reshape minibatch to (bs x patches x D)
        x = x.view(bs, -1, self.D)
        # average pooling over patches (can be interchanged with coefficient mult.)
        x = x.mean(dim=1)

        # we apply sign(x)*sqrt(|x|). This may only be necessary to bound polynomials!
        # sqrt improves accuracy by 5%!
        if self.sqrt:
            x = torch.sign(x + 1e-5) * torch.sqrt(torch.abs(x + 1e-5))
        # unit normalization
        if self.norm:
            x = x / x.norm(dim=-1, keepdim=True)
        # we apply the classifier
        # print(x.shape)
        # x = self.classifier(x)

        return x