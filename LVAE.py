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
from torch.distributions import LogNormal
from torch.nn import init

## what do we need? We need some operational blocks:
## 1a) MLP block that returns the last hidden layer and then mu and std (or log_var)
## 2a) MLP block that returns only the mu and std
##
## Then we have to be able to do:
## 1b) Reparam trick
## 2b) Merge two gaussian

## TODO: MAYBE ADD A + 1e-8 IN THE LOG TO GET SOMETHING MORE STABLE AND AVOID NAN GET BY LOG(0)
def log_standard_gaussian(x):
    """
    Evaluates the log pdf of a standard normal distribution at x. (Univariate distribution)
    :param x: point to evaluate
    :return: log N(x|0,I)
    """
    return torch.sum(-0.5 * math.log(2 * math.pi ) - x ** 2 / 2, dim=-1)


def log_gaussian(x, mu, var):
    """
    Returns the log pdf of a normal distribution parametrised
    by mu and var evaluated at x.
    :param x: point to evaluate
    :param mu: mean of distribution
    :param var: variance of distribution
    :return: log N(x|µ,σ)
    """
    log_pdf = - 0.5 * math.log(2 * math.pi) - torch.log(var + 1e-8) / 2 - (x - mu)**2 / (2 * var + 1e-8)
    # print('Size log_pdf:', log_pdf.shape)
    return torch.sum(log_pdf, dim=-1)


def reparametrization_trick(mu, var):
    '''
    Function that given the mean (mu) and the logarithmic variance (log_var) compute
    the latent variables using the reparametrization trick.
        z = mu + sigma * noise, where the noise is sample

    :param mu: mean of the z_variables
    :param var: variance of the latent variables (as in the paper)
    :return: z = mu + sigma * noise
    '''
    # compute the standard deviation from the variance
    std = torch.sqrt(var)

    # we have to sample the noise (we do not have to keep the gradient wrt the noise)
    eps = Variable(torch.randn_like(std), requires_grad=False)
    z = mu.addcmul(std, eps)

    return z

def merge_gaussian(mu1, var1, mu2, var2):
    # we have to compute the precision 1/variance
    precision1 = 1 / (var1 + 1e-8)
    precision2 = 1 / (var2 + 1e-8)

    # now we have to compute the new mu = (mu1*prec1 + mu2*prec2)/(prec1+prec2)
    new_mu = (mu1 * precision1 + mu2 * precision2) / (precision1 + precision2)

    # and the new variance var = 1/(prec1 + prec2)
    new_var = 1 / (precision1 + precision2)

    # we have to transform the new var into log_var
    # new_log_var = torch.log(new_var + 1e-8)
    return new_mu, new_var

# ## now we create our building block ### TODO: FIX THIS IN A WAY THAT WE CAN HAVE MLP WITH DIFFERENT LAYERS
# class EncoderMLPBlock(nn.Module):
#     def __init__(self, input_dim, hidden_dims, z_dim):
#         '''
#         This is a single block that takes x or d as input and return
#         the last hidden layer and mu and log_var
#         (Substantially this is a small MLP as the encoder we in the original VAE)
#
#         :param input_dim:
#         :param hidden_dims:
#         :param latent_dim:
#         '''
#
#         super(EncoderMLPBlock, self).__init__()
#         ## now we have to create the architecture
#         if isinstance(input_dim, list):
#             neurons = [input_dim[-1], *hidden_dims]
#         else:
#             neurons = [input_dim, *hidden_dims]
#         print(neurons)
#         ## common part of the architecture
#         self.hidden_layers = nn.ModuleList([nn.Linear(neurons[i - 1], neurons[i]) for i in range(1, len(neurons))])
#
#         batchnorms_layer = []
#         for i in range(len(hidden_dims)):
#             batchnorms_layer.append(nn.BatchNorm1d(hidden_dims[i]))
#
#         self.batchnorms = nn.ModuleList(batchnorms_layer)
#
#
#         ## we have two output: mu and log(sigma^2) #TODO: we can create a specific gaussian layer
#         if len(hidden_dims) == 1:
#             self.mu = nn.Linear(hidden_dims[0], z_dim[0])
#             self._var = nn.Linear(hidden_dims[0], z_dim[0])
#         else:
#             self.mu = nn.Linear(hidden_dims[-1], z_dim[0])
#             self._var = nn.Linear(hidden_dims[-1], z_dim[0])
#
#
#     def forward(self, d):
#
#         for layer, batchnorm in zip(self.hidden_layers, self.batchnorms):
#             d = layer(d)
#             d = F.leaky_relu(batchnorm(d))
#
#         _mu = self.mu(d)
#         _var = F.softplus(self._var(d))
#
#
#         return d, _mu, _var
#
# class DecoderMLPBlock(nn.Module):
#     def __init__(self, z1_dim, hidden_dims, z2_dim):
#         '''
#         This is also substantially a MLP, it takes the z obtained from the
#         reparametrization trick and it computes the mu and log_var of the z of the layer
#         below, which, during the inference, has to be merged with the mu and log_var obtained
#         by at the EncoderMLPBlock at the same level.
#
#         :param z1_dim:
#         :param hidden_dims:
#         :param z2_dim:
#         '''
#         super(DecoderMLPBlock, self).__init__()
#         print(hidden_dims)
#         ## now we have to create the architecture
#         if isinstance(z1_dim, list):
#             neurons = [z1_dim[-1], *hidden_dims]
#         else:
#             neurons = [z1_dim, *hidden_dims]
#         print(neurons)
#         # neurons = [z1_dim, *hidden_dims]
#         ## common part of the architecture
#         self.hidden_layers = nn.ModuleList([nn.Linear(neurons[i - 1], neurons[i]) for i in range(1, len(neurons))])
#         batchnorms_layer = []
#         for i in range(len(hidden_dims)):
#             batchnorms_layer.append(nn.BatchNorm1d(hidden_dims[i]))
#
#         self.batchnorms = nn.ModuleList(batchnorms_layer)
#         ## we have two output: mu and log(sigma^2)
#         self.mu = nn.Linear(hidden_dims[-1], z2_dim[0])
#         self._var = nn.Linear(hidden_dims[-1], z2_dim[0])
#
#     def forward(self, d):
#
#         for layer, batchnorm in zip(self.hidden_layers, self.batchnorms):
#             d = layer(d)
#             d = F.leaky_relu(self.batchnorms(d), 0.1)
#
#         _mu = self.mu(d)
#         _var = F.softplus(self._var(d))
#
#
#         return _mu, _var
#
#
# class FinalDecoder(nn.Module):
#     def __init__(self, z_final, hidden_dims, input_dim):
#         '''
#         This is the final decoder, the one that is used only in the generation process. It takes the z_L
#         and then it learn to reconstruct the original x.
#
#         :param z_final:
#         :param hidden_dims:
#         :param input_dim:
#         '''
#         super(FinalDecoder, self).__init__()
#         ## now we have to create the architecture
#         if isinstance(z_final, list):
#             neurons = [z_final[-1], *hidden_dims]
#         else:
#             neurons = [z_final, *hidden_dims]
#         print(neurons)
#         # neurons = [z_final, *hidden_dims]
#         ## common part of the architecture
#         self.hidden_layers = nn.ModuleList([nn.Linear(neurons[i - 1], neurons[i]) for i in range(1, len(neurons))])
#         # test_set_reconstruction layer
#         self.test_set_reconstruction = nn.Linear(hidden_dims[-1], input_dim)
#         self.output_activation = nn.Sigmoid()
#
#     def forward(self, z):
#
#         for layer in self.hidden_layers:
#             z = F.relu(layer(z))
#         # print(self.test_set_reconstruction(x).shape)
#         return self.output_activation(self.test_set_reconstruction(z))


## the simplest way is to assume that each block has only 1 hidden dimension
## now we create our building block
class EncoderMLPBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim):
        '''
        This is a single block that takes x or d as input and return
        the last hidden layer and mu and _var
        (Substantially this is a small MLP as the encoder we in the original VAE)

        :param input_dim:
        :param hidden_dims:
        :param latent_dim:
        '''

        super(EncoderMLPBlock, self).__init__()
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.batchnorm = nn.BatchNorm1d(hidden_dim)

        self.mu = nn.Linear(hidden_dim, z_dim)
        self._var = nn.Linear(hidden_dim, z_dim)


    def forward(self, d):
        d = self.hidden_layer(d)
        d = F.leaky_relu(self.batchnorm(d))

        _mu = self.mu(d)
        _var = F.softplus(self._var(d))


        return d, _mu, _var

class DecoderMLPBlock(nn.Module):
    def __init__(self, z1_dim, hidden_dim, z2_dim):
        '''
        This is also substantially a MLP, it takes the z obtained from the
        reparametrization trick and it computes the mu and var of the _z of the layer
        below, which, during the inference, has to be merged with the mu and _var obtained
        by at the EncoderMLPBlock at the same level.

        :param z1_dim:
        :param hidden_dims:
        :param z2_dim:
        '''
        super(DecoderMLPBlock, self).__init__()

        self.hidden_layer =nn.Linear(z1_dim, hidden_dim)
        self.batchnorm = nn.BatchNorm1d(hidden_dim)
        ## we have two output: mu and # sigma^2
        self.mu = nn.Linear(hidden_dim, z2_dim)
        self._var = nn.Linear(hidden_dim, z2_dim)

    def forward(self, d):


        d = self.hidden_layer(d)
        d = F.leaky_relu(self.batchnorm(d), 0.1)

        _mu = self.mu(d)
        _var = F.softplus(self._var(d))

        return _mu, _var


class FinalDecoder(nn.Module):
    def __init__(self, z_final, hidden_dim, input_dim):
        '''
        This is the final decoder, the one that is used only in the generation process. It takes the z_L
        and then it learn to reconstruct the original x.

        :param z_final:
        :param hidden_dims:
        :param input_dim:
        '''
        super(FinalDecoder, self).__init__()
        ## now we have to create the architecture
        # neurons = [z_final, *hidden_dims]
        ## common part of the architecture
        self.hidden_layer = nn.Linear(z_final, hidden_dim)
        # test_set_reconstruction layer
        self.reconstruction = nn.Linear(hidden_dim, input_dim)
        self.output_activation = nn.Sigmoid()

    def forward(self, z):

        z = F.relu(self.hidden_layer(z))
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

        super(LadderVariationalAutoencoder, self).__init__()
        self.input_dimension = input_dim
        self.hidden_dims = hidden_dims
        self.z_dims = z_dims
        # print('z_dims:', z_dims)
        self.n_layers = len(z_dims)

        ## todo: we should use the n_layers here if you waNt to allow the MLP to have more hidden layers
        neurons = [input_dim, *hidden_dims]
        # print(neurons)
        encoder_layers = [EncoderMLPBlock(neurons[i-1], neurons[i], z_dims[i-1]) for i in range(1, len(neurons))]

        ## this part of the decoder would be used also for inference, to map the input to the hidden space
        #decoder_layers = [DecoderMLPBlock(z_dims[i], hidden_dims[i - 1], z_dims[i-1]) for i in range(1, len(hidden_dims))][::-1]
        decoder_layers = [DecoderMLPBlock(z_dims[i], hidden_dims[i], z_dims[i - 1]) for i in range(1, len(hidden_dims))][::-1]
        # print('decoder layers')
        # print(decoder_layers)
        # print('----')

        self.encoder = nn.ModuleList(encoder_layers)
        self.decoder = nn.ModuleList(decoder_layers)

        ## the final deocder used in the generative process
        self.reconstruction = FinalDecoder(z_dims[0], hidden_dims[0], input_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


    def _approximate_kl(self, z, q_params, p_params = None):
        '''
                The function compute the KL divergence between the distribution q_phi(z|x) and the prior p_theta(z)
                of a sample z.

                KL(q_phi(z|x) || p_theta(z))  = -∫ q_phi(z|x) log [ p_theta(z) / q_phi(z|x) ]
                                              = -E[log p_theta(z) - log q_phi(z|x)]

                :param z: sample from the distribution q_phi(z|x)
                :param q_params: (mu, log_var) of the q_phi(z|x)
                :param p_params: (mu, log_var) of the p_theta(z)
                :return: the kl divergence KL(q_phi(z|x) || p_theta(z)) computed in z
        '''

        ## we have to compute the pdf of z wrt q_phi(z|x)
        (mu, var) = q_params
        qz = log_gaussian(z, mu, var)
        # print('size qz:', qz.shape)
        ## we should do the same with p
        if p_params is None:
            pz = log_standard_gaussian(z)
        else:
            (mu, log_var) = p_params
            pz = log_gaussian(z, mu, log_var)
            # print('size pz:', pz.shape)

        kl = qz - pz

        return kl

    def forward(self, x):
        ## during the infrence we are passing our input through the encoder and
        ## for each layer we have to compute the variance and the mean and store them

        d = x
        mu_and_var_from_layers = []
        latents = []
        for block in self.encoder:
            # we forward the input throug each Encoder block,
            # this gives us the hidden activation, and the mu and var
            d, mu, var = block(d)
            mu_and_var_from_layers.append((mu, var))

        ## now we have store all the mu and var we get from each block
        ## since we are at the top of the encoder, we have to sample the initial z from the last mu and var
        mu, var = mu_and_var_from_layers[-1]

        z = reparametrization_trick(mu, var)
        latents.append(z)
        # print(z.shape)
        # and then use the decoder blocks to get a mu and var to merge with the one obtain from the previous process

        # we will start from the last mu and var (maybe we should consider a stack and use pop)
        mu_and_var_from_layers = list(reversed(mu_and_var_from_layers))

        # we should also start computing the kl divergence
        self.kl_divergence = 0

        ## we can also track the kl of each layer
        # self.kl_divergence_per_layer = {'{}'.format(i+1) : [] for i in range(self.n_layers)}
        self.kl_divergence_per_layer = []

        for i, decoder in enumerate([-1, *self.decoder]):

            mu_d, var_d = mu_and_var_from_layers[i]

            if i == 0:
                # we are at the top, we have to compute the kl
                # self.kl_divergence += self._approximate_kl(z, (mu_d, var_d))
                layer_kl = self._approximate_kl(z, (mu_d, var_d))
                self.kl_divergence += layer_kl
                # print('Info layer_kl', layer_kl)
                # print('Shape layer_kl', layer_kl.shape)
                # self.kl_divergence_per_layer['{}'.format(i+1)].append(torch.sum(layer_kl))
                self.kl_divergence_per_layer.append(torch.sum(layer_kl))

            else:
                # otherwise we have to pass the z through the decoder
                # get the mu and var and merge them with the one we get in the previous step
                mu_t, var_t = decoder(z)

                # we have to merge those
                merged_mu, merged_var = merge_gaussian(mu_d, var_d, mu_t, var_t)

                # and now we can sample them
                z = reparametrization_trick(merged_mu, merged_var)
                latents.append(z)

                # and compute the kl
                layer_kl = self._approximate_kl(z, (merged_mu, merged_var), (mu_t, var_t))
                self.kl_divergence += layer_kl
                # self.kl_divergence_per_layer['{}'.format(i + 1)].append(torch.sum(layer_kl))
                self.kl_divergence_per_layer.append(torch.sum(layer_kl))


        # at the end of the cycle we have the z before the test_set_reconstruction
        pixels_mean = self.reconstruction(z)

        return pixels_mean, latents


    def sample(self, n_images):
        '''
        Method to sample from our generative model

        :return: a sample starting from z ~ N(0,1)
        '''

        z = torch.randn((n_images, self.z_dims[-1]), dtype = torch.float)
        # print(z)
        for decoder in self.decoder:
            mu, var = decoder(z)
            z = reparametrization_trick(mu, var)


        return self.reconstruction(z)











