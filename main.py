from __future__ import print_function

import plrnn
import models
import trainer


import torch as tc
import torch.nn as nn
import torch.autograd as autograd
import io
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter
# writer = SummaryWriter()  # comment

# import train_dvae


# TODO argparse for init
# TODO separate data generation from fitting -> save data

# specify time series meta data
par = {'dim_x': 5, 'dim_z': 2, 'len_t': 100}

# create data
true_model = plrnn.PLRNN(par)
X, Z = true_model.get_timeseries()
# net.plot_xz(block=True)

rec_model = models.SequentialVAE(X=X, ts_par=par)
gen_model = plrnn.PLRNN(par)

trainer_ = trainer.Trainer(ts_par=par, X_true=X, rec_model=rec_model, gen_model=gen_model)
trainer_.train()
