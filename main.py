from __future__ import print_function

import plrnn

import torch as tc
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
X, Z = true_model.forward()
X = X.detach()
Z = Z.detach()
# net.plot_xz(block=True)  # TODO add stability criterion

# TODO plot mean field vae


# TODO plot correlation
# TODO plot log likelihood

# TODO to github
