
import math
import torch
import torch.utils
import torch.utils.data
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init
import numpy as np
from torch.distributions import Normal

torch.manual_seed(0)
np.random.seed(0)


def log_standard_gaussian(x):
    """
    Evaluates the log pdf of a standard normal distribution at x. (Univariate distribution)
    :param x: point to evaluate
    :return: log N(x|0,I)
    """
    return torch.sum(-0.5 * math.log(2 * math.pi) - x ** 2 / 2, dim=-1)

def log_gaussian(x, mu, log_var):
    """
    Returns the log pdf of a normal distribution parametrised
    by mu and log_var evaluated at x.
    :param x: point to evaluate
    :param mu: mean of distribution
    :param log_var: log variance of distribution
    :return: log N(x|µ,σ)
    """
    log_pdf = - 0.5 * math.log(2 * math.pi) - log_var / 2 - (x - mu)**2 / (2 * torch.exp(log_var))
    return torch.sum(log_pdf, dim=-1)

def log_gaussian2(x, mu, var):
    """
    Returns the log pdf of a normal distribution parametrised
    by mu and log_var evaluated at x. (Univariate distribution)
    :param x: point to evaluate
    :param mu: mean of distribution
    :param log_var: log variance of distribution
    :return: log N(x|µ,σ)
    """
    log_pdf = - 0.5 * math.log(2 * math.pi) - torch.log(var) / 2 - (x - mu)**2 / (2 * var)
    # print('Size log_pdf:', log_pdf.shape)
    return torch.sum(log_pdf, dim=-1)

print(log_gaussian(torch.Tensor([3]), torch.Tensor([0.0]), torch.log(torch.Tensor([1.0]))))

print(log_gaussian2(torch.Tensor([3]), torch.Tensor([0.0]), torch.Tensor([1.0])))

prior = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
print(prior.log_prob(torch.Tensor([3])))