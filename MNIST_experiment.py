'''
We are going to learn a latent space and a generative model for the MNIST dataset.

'''

import math
import numpy as np
import torch
import torch.utils
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms, utils
from torch.autograd import Variable
from LADDER_VAE.LVAE import LadderVariationalAutoencoder
from LADDER_VAE.utils import DeterministicWarmup
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

def show_images(images, title=None, path=None):
    images = utils.make_grid(images)
    show_image(images[0], title, path)

def show_image(img, title = "", path = None):
    plt.imshow(img, cmap='gray')
    plt.title(title)
    if path is not None:
        plt.savefig(path)
    plt.show()

# We use this custom binary cross entropy
def binary_cross_entropy(r, x):
    return -torch.sum(x * torch.log(r + 1e-8) + (1 - x) * torch.log(1 - r + 1e-8), dim=-1)

# Writer will output to ./runs/ directory by default
writer = SummaryWriter()

use_cuda = torch.cuda.is_available()
print('Do we get access to a CUDA? - ', use_cuda)
device = torch.device("cuda" if use_cuda else "cpu")

BATCH_SIZE = 256
BATCH_SIZE_TEST = 64
# HIDDEN_LAYERS = [[256], [128], [64]]
# Z_DIM = [[32], [16], [8]]
HIDDEN_LAYERS = [512, 256, 128, 64, 32]
Z_DIM = [64, 32, 16, 8, 4]

N_EPOCHS = 1
LEARNING_RATE = 0.001 #1e-3 #(PAPER ORIGINAL)
WEIGHT_DECAY = -1
N_WARM_UP = 15

N_SAMPLE = 64

N_LAYERS = len(HIDDEN_LAYERS)

SAVE_MODEL_EPOCH = N_EPOCHS - 10

PATH = 'saved_models/'


beta = DeterministicWarmup(n_steps=N_WARM_UP, t_max=1) # Linear warm-up from 0 to 1 over 50 epochs

## we have the binarized MNIST
## TRAIN SET
training_set = datasets.MNIST('../MNIST_dataset', train=True, download=True,
                   transform=transforms.ToTensor())
print('Number of examples in the training set:', len(training_set))
print('Size of the image:', training_set[0][0].shape)
## we plot an example only to check it
idx_ex = 1000
x, y = training_set[idx_ex] # x is now a torch.Tensor
plt.imshow(x.numpy()[0], cmap='gray')
plt.title('Example n {}, label: {}'.format(idx_ex, y))
plt.show()

### we only check if it is binarized
input_dim = x.numpy().size
print('Size of the image:', input_dim)

flatten_bernoulli = lambda x: transforms.ToTensor()(x).view(-1).bernoulli()

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../MNIST_dataset', train=True, transform=flatten_bernoulli),
    batch_size=BATCH_SIZE, shuffle=True)

## TEST SET
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../MNIST_dataset', train=False, transform=flatten_bernoulli),
batch_size=BATCH_SIZE_TEST, shuffle=True)

## another way to plot some images from the dataset
dataiter = iter(train_loader)
images, labels = dataiter.next() ## next return a complete batch --> BATCH_SIZE images
show_images(images.view(BATCH_SIZE,1,28,28))


## now we have our train and test set
## we can create our model and try to train it
model = LadderVariationalAutoencoder(input_dim, HIDDEN_LAYERS, Z_DIM)
print('Model overview and recap\n')
print(model)
print('\n')

## optimization
if WEIGHT_DECAY > 0:
    # we add small L2 reg as in the original paper
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), weight_decay=WEIGHT_DECAY)
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))

## training loop
training_loss = []
approx_kl = []

## we have also to retrieve and store the mean of the kl for each layer
kl_per_layer_per_epoch = {'{}'.format(i+1) : [] for i in range(N_LAYERS)}
print('.....Starting trianing')
_beta = 0
for epoch in range(N_EPOCHS):
    kl_per_batch = np.zeros(N_LAYERS)
    tmp_elbo = 0
    tmp_kl = 0
    tmp_recon = 0
    n_batch = 0
    for i, data in enumerate(train_loader, 0):
        n_batch += 1
        images, labels = data
        images = images.to(device)

        reconstruction, _ = model(images)

        likelihood = -binary_cross_entropy(reconstruction, images)
        elbo = torch.sum(likelihood) - _beta * torch.sum(model.kl_divergence)
        approx_kl.append(torch.sum(model.kl_divergence)/ len(images))


        L = - elbo / len(images)
        L.backward()
        optimizer.step()
        optimizer.zero_grad()
        training_loss.append(elbo/len(images))
        tmp_elbo += L.item() * len(images)
        tmp_recon += torch.sum(likelihood)
        tmp_kl += torch.sum(model.kl_divergence)

        ## we have to add the kl per layer per batch
        layers_kls = model.kl_divergence_per_layer
        for i in range(N_LAYERS):
            kl_per_batch[i] += layers_kls[i]

    ## at the end of each epoch we can store some samples and reconstructions
    with torch.no_grad():
        for r, data in enumerate(test_loader, 0):
            images, labels = data
            images = images.to(device)
            reconstruction, _ = model(images)
            # print(conditional_reconstruction.shape)
            recon_image_ = reconstruction.view(reconstruction.shape[0], 1, 28, 28)
            images = images.view(images.shape[0], 1, 28, 28)
            if r % 100 == 0:
                # show_images(images, 'original')
                # show_images(recon_image_, 'conditional_reconstruction')
                grid1 = torchvision.utils.make_grid(images)
                writer.add_image('orig images', grid1, 0)
                grid2 = torchvision.utils.make_grid(recon_image_)
                writer.add_image('recon images', grid2)
                writer.close()
                ## maybe we just store the conditional_reconstruction
                images = utils.make_grid(images)
                recon_image_ = utils.make_grid(recon_image_)
                plt.imshow(images[0], cmap='gray')
                plt.title('Original from epoch {}'.format(epoch + 1))
                plt.savefig('reconstruction_during_training/originals_epoch_{}_example_{}'.format(epoch + 1, r))
                plt.imshow(recon_image_[0], cmap='gray')
                plt.title('Reconstruction from epoch {}'.format(epoch + 1))
                plt.savefig('reconstruction_during_training/reconstruction_epoch_{}_example_{}'.format(epoch + 1, r))


        ## we want also to sample something from the model during training
        rendom_samples = model.sample(N_SAMPLE)
        samples = rendom_samples.view(rendom_samples.shape[0], 1, 28, 28)
        samples = utils.make_grid(samples)
        plt.imshow(samples[0], cmap='gray')
        plt.title('Samples from epoch {}'.format(epoch+1))
        plt.savefig('samples_during_training/samples_epoch_{}'.format(epoch+1))





    ## we should add the kl per layer in the epoch
    for i in range(N_LAYERS):
        kl_per_layer_per_epoch['{}'.format(i+1)].append(kl_per_batch[i]/ len(train_loader.dataset))

    print('Epoch: {}, Elbo: {}, recon_error: {}, KL: {}'.format(epoch+1, tmp_elbo/ len(train_loader.dataset), -tmp_recon/ len(train_loader.dataset), tmp_kl/ len(train_loader.dataset) ))
    if epoch + 1 > SAVE_MODEL_EPOCH:
        ## we have to store the model
        torch.save(model.state_dict(), PATH + 'nlayer_{}_epoch_{}_elbo_{}_learnrate_{}'.format(N_LAYERS, epoch+1, tmp_elbo/ len(train_loader.dataset), LEARNING_RATE))

    ## update the beta
    _beta = next(beta)


print('....Training ended')
plt.plot(training_loss, label='Elbo mean per batch')
plt.legend()
plt.show()

plt.plot(approx_kl, label='Approximated KL (mean)')
# plt.plot(anal_kl, label='Analitycal KL (mean)')
plt.legend()
plt.show()

## we should plot the kl per layer

for i in range(N_LAYERS):
    plt.plot(kl_per_layer_per_epoch['{}'.format(i+1)], label='KL layer {}'.format(i+1))

plt.legend()
plt.show()


## at this point I want to take the test set and compute the latent code
## for each example and then run PCA or TSNE and plot it
model.eval()
latent_representation = {'{}'.format(i+1) : [] for i in range(N_LAYERS)}
all_labels = []
with torch.no_grad():
    for i, data in enumerate(test_loader, 0):
        images, labels = data
        labels = labels.numpy()
        images = images.to(device)
        for k in range(len(images)):
            # print('Info about images k', images[k].shape)
            _, latent_repr = model(images[k].unsqueeze(0))
            for i in range(len(latent_repr)):
                # print('info latent', latent_repr[i].numpy()[0].shape)
                latent_representation['{}'.format(i+1)].append(latent_repr[i].numpy()[0])
            all_labels.append(labels[k])

    # at this point the two sets contain what we want
    # we can do PCA and plot the 2 components results
    ## in this case we have N_layers representation, so we
    ## will have to do N_layers PCA

    for i in range(N_LAYERS):
        layer_latent_representation = np.array(latent_representation['{}'.format(i+1)])
        # print(layer_latent_representation.shape)
        pca = PCA(2)
        pca.fit(layer_latent_representation)
        feat = pca.fit_transform(layer_latent_representation)
        features_pca = np.array(feat)
        # print(features_pca.shape)

        colors = ['#0165fc', '#02ab2e', '#fdaa48', '#fffe7a', '#6a79f7', '#db4bda', '#0ffef9', '#bd6c48', '#fea993', '#1e9167']

        COLORS = ["#0072BD",
                  "#D95319",
                  "#006450",
                  "#7E2F8E",
                  "#77AC30",
                  "#EDB120",
                  "#4DBEEE",
                  "#A2142F",
                  "#191970",
                  "#A0522D"]

        # print(all_labels)
        all_labels = np.array(all_labels)
        fig = plt.figure()
        for j in range(10):
            idxs = np.where(all_labels == j)
            # print(idxs)
            plt.scatter(features_pca[idxs,0], features_pca[idxs,1], c = colors[j], label = j)

        # plt.scatter(features_pca[:,0], features_pca[:,1], c = all_labels)
        plt.title('PCA on the latent dimension from layer {}'.format(5-i)) # it was i+1
        plt.legend()
        plt.savefig('PCA/PCA_latent_repr_layer_{}'.format(5-i)) # it was i+1
        plt.show()

    ## CONDITIONAL RECONSTRUCTION
    for i, data in enumerate(test_loader, 0):
        images, labels = data
        images = images.to(device)
        reconstruction, _ = model(images)
        # print(conditional_reconstruction.shape)
        recon_image_ = reconstruction.view(reconstruction.shape[0], 1, 28, 28)
        images = images.view(images.shape[0], 1, 28, 28)
        if i % 100 == 0:
            show_images(images, 'original', 'conditional_reconstruction/original_images_{}.png'.format(i))
            show_images(recon_image_, 'conditional_reconstruction', 'conditional_reconstruction/conditional_reconstruction_{}.png'.format(i))

    # we can randomly sample from the prior with the final model
    for i in range(5):
        images_from_random = model.sample(N_SAMPLE)
        sampled_ima = images_from_random.view(images_from_random.shape[0], 1, 28, 28)
        show_images(sampled_ima, 'Random sampled images', 'random_samples/samples_prova_{}.png'.format(i))
