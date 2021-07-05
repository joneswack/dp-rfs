import torch

def softplus(x):
    return torch.nn.functional.softplus(x, beta=1, threshold=20)

def softplus_inverse(x):
    return torch.log(torch.expm1(x))
