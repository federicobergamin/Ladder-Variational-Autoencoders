'''
Personal implementation og Ladder Variational AutoEncoder by Sønderby C. [2016]

arxiv papers: https://arxiv.org/abs/1602.02282
'''

import math
import torch
import torch.utils
import torch.utils.data
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

## what do we need? We need some operational blocks:
## 1a) MLP block that returns the last hidden layer and then mu and std (or log_var)
## 2a) MLP block that returns only the mu and std
##
## Then we have to be able to do:
## 1b) Reparam trick
## 2b) Merge two gaussian

def reparametrization_trick(mu, log_var):
    ## TODO: BE CAREFUL OF LOG_VAR AND VAR
    '''
    Function that given the mean (mu) and the logarithmic variance (log_var) compute
    the latent variables using the reparametrization trick.
        z = mu + sigma * noise, where the noise is sample

    :param mu: mean of the z_variables
    :param log_var: variance of the latent variables
    :return: z = mu + sigma * noise
    '''
    # we should get the std from the log_var
    # log_std = 0.5 * log_var (use the logarithm properties)
    # std = exp(log_std)
    std = torch.exp(log_var * 0.5)

    # we have to sample the noise (we do not have to keep the gradient wrt the noise)
    eps = Variable(torch.randn_like(std), requires_grad=False)
    z = mu.addcmul(std, eps)

    return z

def merge_gaussian(mu1, log_var1, mu2, log_var2):
    # we have to compute the precision 1/variance
    precision1 = 1 / torch.exp(log_var1)
    precision2 = 1 / torch.exp(log_var2)

    # now we have to compute the new mu = (mu1*prec1 + mu2*prec2)/(prec1+prec2)
    new_mu = (mu1 * precision1 + mu2 * precision2) / (precision1 + precision2)

    # and the new variance var = 1/(prec1 + prec2)
    new_var = 1 / (precision1 + precision2)

    # we have to transform the new var into log_var
    new_log_var = torch.log(new_var + 1e-8)

    # TODO: maybe returning also new_var??
    return new_mu, new_log_var

## now we create our building block
class EncoderMLPBlock(nn.Module):
    def __init__(self, input_dim, hidden_dims, z_dim):
        '''
        This is a single block that takes x or d as input and return
        the last hidden layer and mu and log_var
        (Substantially this is a small MLP as the encoder we in the original VAE)

        :param input_dim:
        :param hidden_dims:
        :param latent_dim:
        '''

        super(EncoderMLPBlock, self).__init__()
        ## now we have to create the architecture
        neurons = [input_dim, *hidden_dims]
        ## common part of the architecture
        self.hidden_layers = nn.ModuleList([nn.Linear(neurons[i - 1], neurons[i]) for i in range(1, len(neurons))])

        ## we have two output: mu and log(sigma^2) #TODO: we can create a specific gaussian layer
        self.mu = nn.Linear(hidden_dims[-1], z_dim)
        self._var = nn.Linear(hidden_dims[-1], z_dim)


    def forward(self, d):

        for layer in self.hidden_layers:
            d = layer(d)
            d = F.leaky_relu(nn.BatchNorm1d(d))

        _mu = self.mu(d)
        _var = F.softplus(self._var(d))


        return d, _mu, _var

class DecoderMLPBlock(nn.Module):
    def __init__(self, z1_dim, hidden_dims, z2_dim):
        '''
        This is also substantially a MLP, it takes the z obtained from the
        reparametrization trick and it computes the mu and log_var of the z of the layer
        below, which, during the inference, has to be merged with the mu and log_var obtained
        by at the EncoderMLPBlock at the same level.

        :param z1_dim:
        :param hidden_dims:
        :param z2_dim:
        '''
        super(DecoderMLPBlock, self).__init__()
        ## now we have to create the architecture
        neurons = [z1_dim, *hidden_dims]
        ## common part of the architecture
        self.hidden_layers = nn.ModuleList([nn.Linear(neurons[i - 1], neurons[i]) for i in range(1, len(neurons))])

        ## we have two output: mu and log(sigma^2) #TODO: we can create a specific gaussian layer
        self.mu = nn.Linear(hidden_dims[-1], z2_dim)
        self._var = nn.Linear(hidden_dims[-1], z2_dim)

    def forward(self, d):

        for layer in self.hidden_layers:
            d = layer(d)
            d = F.leaky_relu(nn.BatchNorm1d(d))

        _mu = self.mu(d)
        _var = F.softplus(self._var(d))


        return _mu, _var


class FinalDecoder(nn.Module):
    def __init__(self, z_final, hidden_dims, input_dim):
        '''
        This is the final decoder, the one that is used only in the generation process. It takes the z_L
        and then it learn to reconstruct the original x.

        :param z_final:
        :param hidden_dims:
        :param input_dim:
        '''
        super(FinalDecoder, self).__init__()
        ## now we have to create the architecture
        neurons = [z_final, *hidden_dims]
        ## common part of the architecture
        self.hidden_layers = nn.ModuleList([nn.Linear(neurons[i - 1], neurons[i]) for i in range(1, len(neurons))])
        # test_set_reconstruction layer
        self.reconstruction = nn.Linear(hidden_dims[-1], input_dim)
        self.output_activation = nn.Sigmoid()

    def forward(self, z):

        for layer in self.hidden_layers:
            z = F.relu(layer(z))
        # print(self.test_set_reconstruction(x).shape)
        return self.output_activation(self.reconstruction(z))


## now we have to put all these things together, to create the Ladder Variational Autoencoder
class LadderVariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, z_dims):
        '''

        :param input_dim: dimension of the input
        :param hidden_dims: an array containing thedimension of the hidden layer for each MLP block.
                            As described in the paper each block is made up of only one hidden layer TODO: allow the MLP to be bigger
        :param z_dims: array that contains the dimension of the different z considered
        '''

        self.input_dimension = input_dim
        self.hidden_dims = hidden_dims
        self.z_dims = z_dims
        self.n_layers = len(z_dims)

        ## todo: we should use the n_layers here if you wnat to allow the MLP to have more hidden layers
        neurons = [input_dim, *hidden_dims]
        encoder_layers = [EncoderMLPBlock(neurons[i-1], neurons[i], z_dims[i-1]) for i in range(1, len(neurons))]

        ## this part of the decoder would be used also for inference, to map the input to the hidden space
        decoder_layers = [DecoderMLPBlock([z_dims[i - 1], hidden_dims[i - 1], z_dims[i]]) for i in range(2, len(hidden_dims))][::-1]

        self.encoder = nn.ModuleList(encoder_layers)
        self.decoder = nn.ModuleList(decoder_layers)

        ## the final deocder used in the generative process
        self.reconstruction = FinalDecoder(z_dims[0], hidden_dims[0], input_dim)

        ## TODO: add initialization

    def forward(self, x):
        ## during the infrence we are passing our input through the encoder and
        ## for each layer we have to compute the variance and the mean and store them

        mu_and_var_from_layers = []
        for block in self.encoder:
            d, mu, var = block(x)











