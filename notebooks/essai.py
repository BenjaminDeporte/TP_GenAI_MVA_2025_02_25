import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.set_default_dtype(torch.float64)

if torch.cuda.is_available():
    device = torch.device('cuda')
    dtype = torch.cuda.FloatTensor
else:
    device = torch.device('cpu')
    dtype = torch.FloatTensor

# If you don't want to bother with the device, stay on cpu:
# device = torch.device('cpu')

print(f"Using {device}")

# !nvidia-smi

# set seeds for reproductibility
random_seed = 42
rng = np.random.default_rng(seed=random_seed)
torch.manual_seed(random_seed)

import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms
import torchvision.utils as vutils
import torch.utils.data as data
from torch.utils.data import Subset

import os

# here, we normalize between 0 and 1

# MNIST Dataset
batch_size = 128
datapath = '/home/benjamin.deporte/MVA/TP_GenAI_MVA_2025_02_25/data'  # setup IRT

train_dataset2 = MNIST(datapath, train=True, transform=transforms.ToTensor(), download=True)
test_dataset2 = MNIST(datapath, train=False, transform=transforms.ToTensor(), download=False)

# Data Loader (Input Pipeline)
train_loader2 = torch.utils.data.DataLoader(dataset=train_dataset2, batch_size=batch_size, shuffle=True)
test_loader2 = torch.utils.data.DataLoader(dataset=test_dataset2, batch_size=batch_size, shuffle=False)

latent_dim = 2

class EncoderMLP2(nn.Module):
        def __init__(self, latent_dim=latent_dim):
                
                super(EncoderMLP2, self).__init__()
                
                # self.fc1 = nn.Linear(1*28*28, n_neurons)
                # self.fc2 = nn.Linear(n_neurons, n_neurons)
                # # self.fc3 = nn.Linear(n_neurons, n_neurons)
                # self.fc_mu = nn.Linear(n_neurons, latent_dim)
                # self.fc_logvar = nn.Linear(n_neurons, latent_dim)
                
                # encoder part
                self.fc1 = nn.Linear(784, 512)
                self.fc2 = nn.Linear(512, 256)
                self.fc31 = nn.Linear(256, latent_dim)
                self.fc32 = nn.Linear(256, latent_dim)
                
        def forward(self,inputs):
                # input is B x 1 x 28 x 28
                # x = inputs.view(-1, 784)                # x = self.deconv1(x) # out : B x 32 x 3 x 3
                # x = self.deconv2(x) # out : B x 32 x 6 x 6
                # x = self.deconv3(x) # out : B x 32 x 12 x 12
                # x = self.deconv4(x) # out : B x 32 x 28 x 28
                x = F.relu(self.fc1(inputs)) # out : B x 16 x 14 x 14
                x = F.relu(self.fc2(x)) # out : B x 32 x 7 x 7
                # x = F.relu(self.fc3(x)) 
                mu_x = self.fc31(x) # out : B x latent_dim
                logvar_x = self.fc32(x) # out : B x latent_dim

                return mu_x, logvar_x
        
encoder = EncoderMLP2()

# checking dimensions...
# 
batch_example_size = 8

x = torch.randn((batch_example_size,1,28,28))
print(f"input = {x.shape}")

mu, logvar = encoder(x.view(-1,784))
print(f"output mu = {mu.shape}")
print(f"output logvar = {logvar.shape}")

# Decoder to sample back from vector latent_dim to 28 x 28 within [0,1]^784

class DecoderMLP2(nn.Module):
        def __init__(self, latent_dim=latent_dim, n_neurons=256):
                super(DecoderMLP2, self).__init__()

                # self.fc1 = nn.Linear(latent_dim, n_neurons)
                # self.fc2 = nn.Linear(n_neurons, n_neurons)
                # self.fc3 = nn.Linear(n_neurons, 1*28*28)
                
                self.fc4 = nn.Linear(latent_dim, 256)
                self.fc5 = nn.Linear(256, 512)
                self.fc6 = nn.Linear(512, 784)
                
        def forward(self, inputs):
                x = F.relu(self.fc4(inputs))
                x = F.relu(self.fc5(x))
                x = F.sigmoid(self.fc6(x))
                
                return x
        
decoder = DecoderMLP2()

# check dimensions

x = torch.randn(8, latent_dim) # B (batch size) x latent_dim
print(f"input shape = {x.shape}")
out = decoder(x)
print(f"output shape = {out.shape}")

print(out.min())
print(out.max())

class MnistVAE2(nn.Module):
    
    def __init__(self, n_neurons=256, latent_dim=2): #, scale=1.0):
        super(MnistVAE2, self).__init__()
        
        self.encoder = EncoderMLP2(latent_dim=latent_dim)
        self.decoder = DecoderMLP2(latent_dim=latent_dim)
        # self.scale = scale
        
    def rsample(self, mean, std): #, scale=None):
        # mean : B x latent_dim
        # std : B x latent_dim
        # if scale is None:
        #     scale=self.scale
        epsilon = torch.randn_like(mean) # N(0,1) shape B x latent_dim 
        z = mean + std * epsilon # * scale # B x 1 \sim \mathcal{N}(mu, var)
        return z
    
    def forward(self,x):
        mu, logvar = self.encoder(x)
        std = torch.exp(0.5 * logvar) # we scale by sqrt(diag(var))
        # print(f"entrée de rsample : mu = {mu.size()}")
        # print(f"entrée de rsample : std = {std.size()}")
        z = self.rsample(mu, std)
        # print(f"z samplé = {z.size()}")
        x_hat = self.decoder(z)
        
        return x_hat, mu, logvar
    
# check dimensions
mvae = MnistVAE2(latent_dim=latent_dim)

x = torch.randn(128, 1, 28, 28) # B (batch size) x 1 x 28 x 28

print(f"input shape = {x.shape}")

x_hat, mu, logvar = mvae(x.view(-1,784))

print(f"outputs shapes : x_hat = {x_hat.size()}, mus = {mu.size()},  logvars = {logvar.size()}")

# print(x_hat.min())
# print(x_hat.max())

def mnist_vae_bceloss(x, x_hat, mean, logvar, kl_weight=0.5):
    
    reconstruction_loss = F.binary_cross_entropy(x.view(-1,784), x_hat.view(-1,784), reduction='sum')
    kl_loss = - 0.5 * torch.sum( 1 + logvar - mean**2 - logvar.exp()) # validé
    
    # print(f"reconstruction loss = {reconstruction_loss}")
    # print(f"KL loss = {kl_loss}")
    
    return (1-kl_weight)*(reconstruction_loss) + kl_weight*kl_loss, kl_loss, reconstruction_loss

def train_MNIST_VAE_BCE(
    loader = None,
    n_epochs = 10,
    optimizer = None,
    model = None,
    kl_weight = 0.5
):
    """_summary_

    Args:
        loader (_type_, optional): sample_loader ou train_loader. Defaults to None.
        n_epochs (int, optional): _description_. Defaults to 10.
        optimizer (_type_, optional): _# Decoder to sample back from vector latent_dim to 28 x 28

    Returns:
        _type_: lists of losses
    """
    
    print(f"Start training {n_epochs} epochs on MNIST VAE")
    rec_losses = []
    kl_losses = []
    total_losses = []
    
    model.train()
    
    for epoch in range(n_epochs):
        
        # init batch measurements
        batch_kl_losses = []
        batch_rec_losses = []
        batch_total_losses = []
        
        for i, batch in enumerate(loader, 0):
            
            # forward pass
            x = batch[0].view(-1,784)
            x_hat, mean, logvar = model(x)
            print(f"max = {x_hat.max()}")
            print(f"min = {x_hat.min()}")
            
            # backward pass
            optimizer.zero_grad()
            # print(x_hat)
            total_loss, kl_loss, rec_loss = mnist_vae_bceloss(x, x_hat, mean, logvar, kl_weight)
            total_loss.backward()
            optimizer.step()
            
            # logging at batch level
            batch_kl_losses.append(kl_loss.item())
            batch_rec_losses.append(rec_loss.item())
            batch_total_losses.append(total_loss.item())
            
            # reporting out at batch level
            print(f"Batch {i+1} / {len(loader)} \
            -- total loss = {batch_total_losses[-1]:.4f} \
            -- reco loss = {batch_rec_losses[-1]:.4f} \
            -- KL loss = {batch_kl_losses[-1]:.4f}", \
            end="\r")
            
        # logging at epoch level
        total_losses.append(np.average(batch_total_losses))  
        rec_losses.append(np.average(batch_rec_losses)) 
        kl_losses.append(np.average(batch_kl_losses)) 
        
        # reporting out at epoch level
        print(f"Epoch {epoch+1} / {n_epochs} \
            -- loss = {total_losses[-1]:.4f} \
            -- rec_loss = {rec_losses[-1]:.4f} \
            -- kl_loss = {kl_losses[-1]:.4f} \
                                           "
        ) #, end="\r")
        
    return total_losses, rec_losses, kl_losses  

latent_dim = 2
mnist_vae_bce = MnistVAE2(latent_dim=latent_dim)

lr = 1e-6
optimizer = torch.optim.Adam(mnist_vae_bce.parameters(), lr=lr)

n_epochs = 5
kl_weight = 0.5

total_losses, rec_losses, kl_losses = train_MNIST_VAE_BCE(
    loader=train_loader2, 
    n_epochs=n_epochs, 
    optimizer=optimizer, 
    model=mnist_vae_bce,
    kl_weight=kl_weight
    )