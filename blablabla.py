
import math
import torch
import torch.utils
import torch.utils.data
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import LogNormal
#
# neurons = [12,13,12]
# c = [1,2,3]
#
# # self.hidden_layers = nn.ModuleList([nn.Linear(neurons[i - 1], neurons[i]), nn.BatchNorm1d([neurons[i]]) for i in range(1, len(neurons))])
# # c = [[i, i+1] for i in range(1, len(neurons))]
# # print(c)
# for i, n in zip(neurons,c):
#     print(i,n)

# z = torch.randn((10, 2), dtype=torch.float)
# print(z)

# d = {'{}'.format(i+1) : [] for i in range(5)}
# print(d)
# d['2'].append(4)
# print(d)
#
# hidden_dims = [512, 256, 128, 64, 32]
# z_dims = [64, 32, 16, 8, 4]

# for i in range(0, len(hidden_dims)-1):
#     print('--- layer {} ---- '.format(i+1))
#     print(z_dims[i+1])
#     print(hidden_dims[i+1])
#     print(z_dims[i])



# for i in range(1, len(hidden_dims)):
#     print('--- layer {} ---- '.format(i+1))
#     print(z_dims[i])
#     print(hidden_dims[i])
#     print(z_dims[i-1])

from LADDER_VAE.utils import DeterministicWarmup


beta = DeterministicWarmup(n_steps=5, t_max=1)
i = 0
while i < 10:
    print(next(beta))
    i +=1