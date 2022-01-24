import torch
import torch.nn as nn

from tensorboardX import SummaryWriter

import numpy as np

from time import time
from datetime import datetime
from timeit import default_timer as timer

from likelihoods.softmax_likelihood import Softmax


class VariationalGP(nn.Module):
    def __init__(self, d_hidden, d_out, feature_encoder, trainable_noise=False,
                    trainable_vars=True, covariance='factorized', use_gpu=False, dtype=torch.FloatTensor):
        """
        d_hidden: random feature output dimension
        d_out: output dimension

        log_var: log noise variance for gaussian likelihood
        q_mu: mean of Q
        q_log_vars: log variances for Q distribution. Also serves as positive diagonal of L.
        q_L: lower triangular of covariance of Q excluding the diagonal

        covariance: 0 - MAP, factorized - ~L2 reg, full - Gaussian GP approx
        """
        super(VariationalGP, self).__init__()

        self.dtype = dtype
        self.use_gpu = use_gpu

        self.d_hidden = d_hidden
        self.d_out = d_out
        self.covariance = covariance

        # we could think about adding a jitter to the variances whenever dividing by them or taking their log
        self.log_noise_var = nn.Parameter(torch.zeros(1).type(self.dtype), requires_grad=trainable_noise)

        self.p_mu = nn.Parameter(torch.zeros(d_hidden).type(self.dtype), requires_grad=False)
        # the log vars (amplitudes) are initialized to log(kernel_var / sqrt(D))
        self.p_logvars = nn.Parameter(torch.zeros(1).type(self.dtype), requires_grad=False)

        # we initialize each Q_j, j in 1,...,d_out to the prior of p.
        # this will ensure that we start off with a KL divergence equal to 0.
        # q_mu has the shape (d_out, d_hidden) - d_out instances always at index 0
        self.q_mu = nn.Parameter(self.p_mu.data.clone().repeat(d_out, 1), requires_grad=True)

        if covariance == 'full':
            # For the initialization, we have to store sqrt(sigma**2) because we take L L.T where L diagonal
            # log sigma = log (sigma**2) / 2
            self.log_q_L_diag = nn.Parameter(0.5 * self.p_logvars.data.clone().repeat(d_out, d_hidden), requires_grad=True)
            self.q_L = nn.Parameter(torch.zeros(d_out, d_hidden, d_hidden).type(self.dtype), requires_grad=True)
        else:
            self.q_logvars = nn.Parameter(self.p_logvars.data.clone().repeat(d_out, d_hidden), requires_grad=True)

        # the feature encoder will take care of all kernel hyperparameters except the kernel variance
        self.feature_encoder = feature_encoder
        self.log_feature_scale = nn.Parameter(torch.zeros(1).type(self.dtype), requires_grad=trainable_vars)

        self.likelihood = Softmax()
    
    def reconstruct_covar(self):
        """
        Returns L_sigma and sigma
        """

        if self.covariance == 'full':
            lower_triang_q = torch.tril(self.q_L, -1)
            # the following only guarantees non-negativity. we may need to add noise if numerical issues occur!
            lower_triang_q += torch.diag_embed(torch.exp(self.log_q_L_diag))
            # permute allows transposing matrices inside tensors (we only change axis 1 and 2)
            q_sigma = torch.matmul(lower_triang_q, lower_triang_q.permute(0, 2, 1))
        else:
            lower_triang_q = torch.diag_embed(torch.exp(0.5 * self.q_logvars))
            q_sigma = torch.diag_embed(torch.exp(self.q_logvars))

        return lower_triang_q, q_sigma

    def forward(self, x, lower_triang_q, mc_samples=10):
        """
        Samples from XW with (local) reparameterization.

        lower_triang_q: The lower cholesky factor of the weight covariance.
        We pass it as an argument so that it only needs to be built once during each update.
        """

        if self.feature_encoder is not None:
            x = self.feature_encoder(x)

        x = self.log_feature_scale.exp() * x

        mu_output = torch.mm(x, self.q_mu.t())

        if self.covariance == 'zero':
            # for the MAP estimate, we only return the means
            return mu_output.unsqueeze(0)
        elif self.covariance == 'full':
            # now we need to collect the variances

            # X[samples, d] @ L[out_dim, d, d] = [out_dim, samples, d]
            std_output = torch.matmul(x, lower_triang_q)
            std_output = std_output.norm(dim=-1).t()

            # std_output = torch.zeros(len(x), self.d_out).type(self.dtype)
            # for dim in range(self.d_out):
            #     # V[y] = x.T Sigma x = || L.T x ||^2
            #     # In row format: V[y] = || x.T L ||^2 => norm(X L, dim=1)**2
            #     # Norm gives the standard deviation
            #     # we may need to add some jitter to the variances
            #     std_output[:, dim] = torch.mm(x, lower_triang_q[dim]).norm(dim=1)
        else:
            std_output = torch.mm(x, torch.exp(0.5 * self.q_logvars.t()))

        # now we sample from the mini-batch of y
        if self.use_gpu:
            epsilon = torch.randn(mc_samples, *mu_output.shape, device='cuda')
        else:
            epsilon = torch.randn(mc_samples, *mu_output.shape, device='cpu')
        y = mu_output + std_output * epsilon

        return y

    def kl_divergence_full_vs_factorized(self, q_mu, q_sigma, log_q_L_diag, p_mu, p_logvars):
        """
        Numerically stable implementation of the KL divergence between the full and factorized gaussian.
        KL(N_0 || N_1) = 0.5 * (A + B - k + C)

        L L.T = Sigma_0
        A = tr (Sigma_1^(-1) Sigma_0) = sum_i(Sigma_0_ii / Sigma_1_ii)
        B = (mu_1 - mu_0).T Sigma_1^(-1) (mu_1 - mu_0) = sum_i((mu_1_i - mu_0_i)**2 / Sigma_1_ii)
        C = log (det(Sigma_1) / det(Sigma_0)) = sum_i(log Sigma_1_ii) - 2*sum_i(log L_ii)
        
        L_Sigma_0_ii > 0 ensures positive definiteness of Sigma and is crucial for the log in C!
        Moreover, Sigma needs to be p.d. to define a proper multivariate normal distribution!
        """

        # we need to turn p_logvars into a diagonal matrix
        p_logvars = p_logvars * torch.ones_like(p_mu) #.type(self.dtype)

        A = torch.sum((-p_logvars).exp() * torch.diagonal(q_sigma, dim1=1, dim2=2), dim=1)
        B = torch.sum((-p_logvars).exp() * (p_mu - q_mu)**2, dim=1)
        # we can substitute q_logvars for log L_Sigma_0_ii since they are used in the construction
        C = torch.sum(p_logvars) - 2.0 * torch.sum(log_q_L_diag, dim=1)

        k = len(p_logvars)
        # we get D sub KL divergences that are summed up
        return 0.5 * (A + B - k + C).sum()

    def kl_divergence_factorized(self, q_mu, q_logvars, p_mu, p_logvars):

        # we need to turn p_logvars into a diagonal matrix
        p_logvars = p_logvars * torch.ones_like(p_mu) # .type(self.dtype)

        kl = 0.5 * torch.sum(
            p_logvars - q_logvars
            + (torch.pow(q_mu - p_mu, 2) * torch.exp(-p_logvars))
            + torch.exp(q_logvars - p_logvars) - 1.0
        )
        return kl

    def negative_lower_bound(self, X, Y, num_points, kl_weight=1.0, mc_samples=10):
        """
        Returns the negative lower bound to the marginal likelihood.
        NELL = KL - ELL
        """

        if self.covariance == 'full':
            lower_triang_q, q_sigma = self.reconstruct_covar()
            latent_val = self.forward(X, lower_triang_q, mc_samples=mc_samples)
            kl = self.kl_divergence_full_vs_factorized(
                self.q_mu, q_sigma, self.log_q_L_diag,
                self.p_mu, self.p_logvars
            )
        else:
            latent_val = self.forward(X, None, mc_samples=mc_samples)
            kl = self.kl_divergence_factorized(
                self.q_mu, self.q_logvars,
                self.p_mu, self.p_logvars
            )

        # the log probs should be of shape (NMC x BS)
        # we take the monte carlo average and sum over the batch
        batch_ell = self.likelihood.log_cond_prob(Y, latent_val).mean(dim=0).sum()

        ell = batch_ell * float(num_points) / len(X)
        loss = kl_weight * kl - ell

        # MC average
        probs = self.likelihood.predict(latent_val).mean(dim=0)
        batch_nll = -(Y * probs).sum(dim=1).log().sum()
        # prediction
        predictions = probs.argmax(dim=1)

        target = torch.argmax(Y, dim=1)
        n_correct = torch.sum(predictions.data == target.data)

        # error = 1. - correct.float() / (Y.shape[0])

        return loss, batch_nll, kl, n_correct

    def compute_error(self, X, Y, mc_samples=100):
        if self.covariance == 'full':
            lower_triang_q, _ = self.reconstruct_covar()
            latent_val = self.forward(X, lower_triang_q, mc_samples=mc_samples)
        else:
            latent_val = self.forward(X, None, mc_samples=mc_samples)

        # MC average
        probs = self.likelihood.predict(latent_val).mean(dim=0)
        # prediction
        predictions = probs.argmax(dim=1)

        target = torch.argmax(Y, dim=1)
        correct = torch.sum(predictions.data == target.data)

        return 1. - correct.float() / (Y.shape[0])

    def test_epoch(self, test_loader, kl_weight):

        with torch.no_grad():
            test_loss = 0
            test_nll = 0
            test_kl = 0
            test_correct = 0

            for _, batch_data in enumerate(test_loader):
                if self.use_gpu:
                    batch_data[0] = batch_data[0].cuda()
                    batch_data[1] = batch_data[1].cuda()

                loss, batch_nll, kl, n_correct = self.negative_lower_bound(batch_data[0], batch_data[1], len(test_loader.dataset), kl_weight, mc_samples=200)
                # error = self.compute_error(batch_data[0], batch_data[1], mc_samples=10)

                test_loss += loss.item()
                test_nll += batch_nll.item()
                test_kl += kl.item()
                test_correct += n_correct.item()

            test_loss /= len(test_loader)
            test_mnll = test_nll / len(test_loader.dataset)
            test_kl /= len(test_loader)
            test_error = 1. - test_correct / len(test_loader.dataset)

        return test_loss, test_mnll, test_kl, test_error

    def optimize_lower_bound(self, model_name, train_loader, test_loader,
                            num_epochs=10, lr=1e-3, a=0.5, b=50, gamma=1.0):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
        # optimizer = torch.optim.LBFGS(self.parameters(), lr=lr, max_iter=20, history_size=100)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, mode='min', factor=0.1, patience=8, verbose=True, threshold=1e-4)

        # we should have one log dir per run
        # otherwise tensorboard will have overlapping graphs
        log_dir = 'tensorboard_logs/{}_lr_{}_epochs_{}/{}'.format(
            model_name, lr, num_epochs, datetime.now().strftime("%Y%m%d-%H%M%S"))
        summary_writer = SummaryWriter(log_dir)

        total_training_time = 0

        for iteration in range(num_epochs):
            print('### Epoch: {} ###'.format(iteration))

            # kl_weight += delta_kl
            # kl_weight = 1. / (1. + np.exp(-a*(iteration - b)))
            kl_weight = gamma

            test_loss, test_mnll, test_kl, test_error = self.test_epoch(test_loader, kl_weight)
            
            print('Test Loss:', test_loss)
            print('Test MNLL:', test_mnll)
            print('Test KL:', test_kl)
            print('Test error:', test_error)
            summary_writer.add_scalar('test_loss', test_loss, iteration)
            summary_writer.add_scalar('test_mnll', test_mnll, iteration)
            summary_writer.add_scalar('test_kl', test_kl, iteration)
            summary_writer.add_scalar('test_error', test_error, iteration)
            summary_writer.add_scalar('training_time', total_training_time, iteration)

            train_loss = 0
            train_nll = 0
            train_kl = 0
            train_correct = 0

            if self.use_gpu:
                torch.cuda.synchronize()
            start_time = time()

            for _, batch_data in enumerate(train_loader):
                if self.use_gpu:
                    batch_data[0] = batch_data[0].cuda()
                    batch_data[1] = batch_data[1].cuda()


                # def closure():
                #     optimizer.zero_grad()
                #     loss, _, _, _ = self.negative_lower_bound(
                #         batch_data[0], batch_data[1],
                #         len(train_loader.dataset), kl_weight, mc_samples=200
                #     )
                #     loss.backward()
                #     # print(loss.item())
                #     return loss
                
                # Train statistics
                # with torch.no_grad():
                optimizer.zero_grad()
                loss, batch_nll, kl, n_correct = self.negative_lower_bound(
                    batch_data[0], batch_data[1],
                    len(train_loader.dataset), kl_weight, mc_samples=1
                )
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_nll += batch_nll.item()
                train_kl += kl.item()
                train_correct += n_correct.item()




                # optimizer.zero_grad()

                # loss, neg_batch_ell, kl, n_correct = self.negative_lower_bound(
                #     batch_data[0], batch_data[1],
                #     len(train_loader.dataset), kl_weight, mc_samples=100
                # )
                # train_loss += loss.item()
                # train_nell += neg_batch_ell.item()
                # train_kl += kl.item()
                # train_correct += n_correct.item()

                # loss.backward()

                # optimizer.step(closure)

            if self.use_gpu:
                torch.cuda.synchronize()
            elapsed = time() - start_time
            print('Epoch time: {}'.format(elapsed))

            total_training_time += elapsed

            train_loss /= len(train_loader)
            train_mnll = train_nll / len(train_loader.dataset)
            train_kl /= len(train_loader)
            train_error = 1. - train_correct / len(train_loader.dataset)

            # scheduler.step(train_mnll)
            # scheduler.step()

            print('Loss:', train_loss)
            print('MNLL:', train_mnll)
            print('KL:', train_kl)
            print('Train error:', train_error)
            summary_writer.add_scalar('train_loss', train_loss, iteration)
            summary_writer.add_scalar('train_mnll', train_mnll, iteration)
            summary_writer.add_scalar('train_kl', train_kl, iteration)
            summary_writer.add_scalar('train_error', train_error, iteration)
            summary_writer.add_scalar('feature_scale', np.exp(self.log_feature_scale.item()), iteration)
            summary_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], iteration)
            summary_writer.add_scalar('kl_weight', kl_weight, iteration)
            # summary_writer.add_scalar('lengthscale', np.median(np.exp(self.feature_encoder.log_lengthscale.detach().cpu())), iteration)
            # summary_writer.add_scalar('p_std', self.p_logvars.item(), iteration)
            summary_writer.add_histogram('q_mean', self.q_mu.detach().cpu().numpy(), iteration)
            # summary_writer.add_histogram('q_stds', self.q_logvars.detach().cpu().exp().sqrt().numpy(), iteration)