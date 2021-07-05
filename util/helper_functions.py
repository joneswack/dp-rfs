import numpy as np
import torch

from likelihoods.softmax_likelihood import Softmax
from likelihoods.gaussian_likelihood import Gaussian


def cholesky_solve(y, L):
    # L: lower triangular cholesky
    return torch.triangular_solve(
        torch.triangular_solve(y, L, upper=False)[0],
        L.conj().t(), transpose=False, upper=True
    )[0]

def kl_factorized_gaussian(mu0, mu1, std0, std1, eps=1e-6):
    # print('Percentage 0: {}'.format((std0 == 0).float().mean()))
    # print('Percentage 1: {}'.format((std1 == 0).float().mean()))
    std0[std0 == 0] = eps
    std1[std1 == 0] = eps

    kl = (std0/std1)**2
    kl += (mu1 - mu0)**2/std1**2 - 1.
    kl += 2. * torch.log(std1 / std0)
    return 0.5*kl

def classification_scores(test_predictions, test_labels):
    """
    Returns MNLL and error rate for classification.
    """

    likelihood = Softmax()
    test_probs = likelihood.predict(test_predictions).mean(dim=0)
    test_mnll = -(test_probs * test_labels).sum(dim=1).log()
    test_mnll[torch.isinf(test_mnll)] = -750
    test_mnll = test_mnll.mean().item()
    test_predictions = test_probs.argmax(dim=1)
    test_target = torch.argmax(test_labels, dim=1)
    test_error = (test_predictions != test_target).float().mean().item()

    # print('Test error: {}'.format(test_error))
    # print('Test MNLL Loss: {}'.format(test_mnll))

    return test_error, test_mnll

def regression_scores(test_mean, test_vars, test_labels):
    likelihood = Gaussian()
    likelihood.log_noise_var.data = test_vars.log()
    # the sum is over the number of outputs
    # since we see two outputs as a diagonal 2d gaussian
    test_mnll = -likelihood.log_cond_prob(test_labels, test_mean).sum(dim=1).mean().item()
    # average is taken also over outputs
    test_mse = (test_mean - test_labels).pow(2).mean().item()

    # print('Test mse: {}'.format(test_mse))
    # print('Test MNLL Loss: {}'.format(test_mnll))

    return test_mse, test_mnll