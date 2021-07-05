import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm

from util.helper_functions import cholesky_solve
from util.LBFGS import FullBatchLBFGS


def predictive_dist_exact(train_data, test_data, train_labels, noise_vars, kernel_fun):
    """
    Heteroskedastic GP predictive distribution with the exact kernel.
    This function is used to compute the ground truth for KL divergence computation.

    train_data: Real-valued training inputs
    test_data: Real-valued test inputs
    train_labels: Real-valued training labels
    noise_vars: Heteroskedastic noise variances
    kernel_fun: callable k(X, Y)
    """

    with torch.no_grad():
        k_xx = kernel_fun(train_data, train_data)
        k_xt = kernel_fun(train_data, test_data)
        k_tt = kernel_fun(test_data, test_data)

        f_test_mean_full = torch.zeros(len(test_data), train_labels.shape[1], device=train_labels.device)
        f_test_stds_full = torch.zeros(len(test_data), train_labels.shape[1], device=train_labels.device)

        for out_dim in tqdm(range(train_labels.shape[1])):
            train_labels_tmp = train_labels[:, out_dim].unsqueeze(1)
            c_xx = k_xx + noise_vars[:, out_dim].diag()
            # c_xx = (c_xx + c_xx.t()) / 2.
            L_xx = torch.linalg.cholesky(c_xx)
            L_X_t = torch.triangular_solve(k_xt, L_xx, upper=False)[0]

            f_test_mean_full[:, out_dim] = (k_xt.t() @ cholesky_solve(train_labels_tmp, L_xx)).view(-1)
            f_test_stds_full[:, out_dim] = (k_tt.diagonal() - (L_X_t**2).sum(dim=0)).sqrt()

    return f_test_mean_full, f_test_stds_full


class HeteroskedasticGP(nn.Module):
    """
    Heteroskedastic Gaussian Process Regression with feature encoder.
    """

    def __init__(self, feature_encoder):
        """
        feature_encoder: random feature encoder
            If None, the inputs will not be transformed before solving the model
        """
        super(HeteroskedasticGP, self).__init__()
        self.feature_encoder = feature_encoder

    def predictive_dist(self, train_data, test_data, train_labels, noise_vars):
        with torch.no_grad():
            # step 1: extract random features (optional)
            if self.feature_encoder is not None:
                train_features = self.feature_encoder.forward(train_data)
                test_features = self.feature_encoder.forward(test_data)
            else:
                train_features = train_data
                test_features = test_data

            if train_features.dtype in [torch.complex32, torch.complex64, torch.complex128]:
                train_labels = train_labels.type(train_features.dtype)
                noise_vars = noise_vars.type(train_features.dtype)
            
            f_test_mean_full = torch.zeros(len(test_data), train_labels.shape[1], device=train_labels.device)
            f_test_stds_full = torch.zeros(len(test_data), train_labels.shape[1], device=train_labels.device)

            for out_dim in tqdm(range(train_labels.shape[1])):
                y_tmp = train_labels[:, out_dim].unsqueeze(1)
                noise_var_tmp = noise_vars[:, out_dim].unsqueeze(1)
                # bottleneck (cannot be accelerated without using a lot of memory)
                sigma_inv = train_features.conj().t() @ (train_features / noise_var_tmp)
                sigma_inv = sigma_inv + torch.eye(len(sigma_inv), device=train_labels.device)
                xTy = train_features.conj().t() @ (y_tmp / noise_var_tmp)
                L_sigma_inv = torch.linalg.cholesky(sigma_inv)
                alpha = cholesky_solve(xTy, L_sigma_inv)

                f_test_mean = test_features @ alpha
                if f_test_mean.dtype in [torch.complex32, torch.complex64, torch.complex128]:
                    f_test_mean = f_test_mean.real
                # bottleneck (cannot be accelerated without using a lot of memory)
                L_sigma_X_t = torch.triangular_solve(test_features.conj().t(), L_sigma_inv, upper=False)[0]
                f_test_stds = L_sigma_X_t.t().norm(dim=-1)

                del sigma_inv, L_sigma_inv, xTy, L_sigma_X_t

                f_test_mean_full[:, out_dim] = f_test_mean.view(-1)
                f_test_stds_full[:, out_dim] = f_test_stds

            return f_test_mean_full, f_test_stds_full

    def predictive_sample(self, train_data, test_data, train_labels, noise_vars, num_samples=100):
        """
        Draws num_samples samples from the predictive distribution on test_data.

        train_data: training inputs
        test_data: test inputs
        train_labels: training labels
        noise_vars: heteroskedastic noise variances
        num_samples: number of samples from the predictive distribution
        """
        with torch.no_grad():
            f_test_mean_full, f_test_stds_full = self.predictive_dist(
                train_data, test_data, train_labels, noise_vars)

            epsilon = torch.randn(num_samples, *f_test_mean_full.shape, device=train_data.device)
            y_predict_test = f_test_mean_full + f_test_stds_full * epsilon

            return y_predict_test

    def log_marginal_likelihood(self, train_data, train_labels, log_noise_vars, heteroskedastic=False):
        """
        Computes the log marginal likelihood on train_data.
        We assume 0 covariance between the individual heteroskedastic GPs.

        This function is only implemented for real-valued random features for now!

        train_data: training inputs
        train_labels: training labels
        log_noise_vars: (heteroskedastic) log noise variances
        heteroskedastic: whether the noise variances are heteroskedastic
        """

        # step 1: feature extraction (to store gradients for kernel parameters)
        if self.feature_encoder is not None:
            train_features = self.feature_encoder.forward(train_data)
        else:
            train_features = train_data

        # step 2: compute marginal log likelihood for every gp
        lml = 0
        for out_dim in range(train_labels.shape[1]):
            y_tmp = train_labels[:, out_dim].unsqueeze(1)
            if heteroskedastic:
                noise_var_tmp = log_noise_vars[:, out_dim].exp().unsqueeze(1)
            else:
                noise_var_tmp = log_noise_vars.exp() * torch.ones_like(y_tmp)

            sigma_inv = train_features.t() @ (train_features / noise_var_tmp)
            sigma_inv = sigma_inv + torch.eye(len(sigma_inv), device=sigma_inv.device)
            L_sigma_inv = torch.linalg.cholesky(sigma_inv)
            xTy = train_features.t() @ (y_tmp / noise_var_tmp)
            q_inv_xTy = torch.triangular_solve(xTy, L_sigma_inv, upper=False)[0]

            lml += -0.5 * noise_var_tmp.log().sum() - L_sigma_inv.diagonal().log().sum()
            lml += -0.5 * (y_tmp * (y_tmp / noise_var_tmp)).sum()
            lml += 0.5 * q_inv_xTy.pow(2).sum()
            lml += -len(y_tmp) / 2 * np.log(2. * np.pi)

        return lml

    def optimize_marginal_likelihood(self, train_data, train_labels, log_noise_vars, fit_noise=False, heteroskedastic=False, num_iterations=10, lr=1e-3):
        """
        Optimizes the log marginal likelihood w.r.t. the random feature hyperparameters and noise variances.

        train_data: training inputs
        train_labels: training labels
        log_noise_vars: log noise variances
        fit_noise: whether to optimize the noise variances
        heteroskedastic: whether to use heteroskedastic noise variances
        num_iterations: number of iterations for LBFGS
        lr: learning rate for LBFGS
        """

        if self.feature_encoder is None:
            raise RuntimeError('There are not parameters to optimize without a feature encoder!')

        trainable_params = [self.feature_encoder.log_lengthscale, self.feature_encoder.log_var]
        if fit_noise:
            trainable_params += [log_noise_vars]

        for iteration in range(num_iterations):

            print('### Iteration {} ###'.format(iteration))
            optimizer = FullBatchLBFGS(trainable_params, lr=lr, history_size=10, line_search='Wolfe')

            def closure():
                optimizer.zero_grad()

                loss = - self.log_marginal_likelihood(train_data, train_labels, log_noise_vars, heteroskedastic=heteroskedastic)
                print('Loss: {}'.format(loss.item()))

                return loss

            loss = closure()
            loss.backward()
            options = {'closure': closure, 'current_loss': loss, 'max_ls': 10}
            loss, _, lr, _, F_eval, G_eval, _, _ = optimizer.step(options)
