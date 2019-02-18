from __future__ import print_function

import plrnn
import models

import torch as tc
import torch.nn as nn
import torch.optim as optim
import io
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter
# writer = SummaryWriter()  # comment

# import train_dvae

# create data
# TODO argparse for init
dim_z = 2
dim_x = 5
len_t = 100
true_model = plrnn.PLRNN(dim_z, dim_x, len_t)
true_model.print_par()
X_true, Z_true = true_model.forward()
X_true = X_true.detach()
Z_true = Z_true.detach()
# net.plot_xz(block=True)  # TODO add stability criterion

# TODO separate data generation from fitting


def loss_function():
    """
    Compute a one-sample approximation the ELBO (lower bound on marginal likelihood),
    normalized by batch size (length of Y in first dimension).
    """
    Z = rec_model.sample()
    Z = Z.t()
    entropy = rec_model.eval_entropy()
    likelihood = gen_model.eval_logdensity(X=X_true, Z=Z)
    # print('entropy {}'.format(entropy))
    # print('likelihood {}'.format(likelihood))
    return - (likelihood + entropy) / len_t


# batch_size = 128
epochs = 10000
cuda = False
log_interval = epochs/10

tc.manual_seed(232)
device = tc.device("cuda" if cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}


rec_model = models.MeanFieldGaussian(X_true, dim_z=dim_z, dim_x=dim_x).to(device)
gen_model = plrnn.PLRNN(dim_z, dim_x, len_t).to(device)

optimizer = optim.Adam(list(rec_model.parameters())+list(gen_model.parameters()), lr=2e-4)  # 1e-3


def train(epoch):
    gen_model.train()
    rec_model.train()
    optimizer.zero_grad()
    loss = loss_function()
    loss.backward()
    train_loss = loss.item()
    optimizer.step()
    if epoch % log_interval == 0:
        print('Epoch: {} Loss: {:.4f}'.format(epoch, train_loss))


for epoch in range(1, epochs + 1):
    train(epoch)


# TODO plot mean field vae
# TODO plot correlation
# TODO plot log likelihood

