from __future__ import print_function

import plrnn
import models
import trainer

import argparse
import torch as tc
import os.path
from tensorboardX import SummaryWriter  # install and use according to https://github.com/lanpa/tensorboardX


def main(args):

    # initialize writer
    log_dir = 'runs/{}'.format(args.name)
    run_nrs = list(os.walk(log_dir))[0][1]
    if not run_nrs:
        run_nrs = ['000']
    run = str(int(max(run_nrs))+1).zfill(3)
    writer = SummaryWriter(log_dir='{}/{}'.format(log_dir, run), comment=args.comment)

    # initialize plrnn
    par = {'dim_x': args.dx, 'dim_z': args.dz, 'len_t': args.T}
    true_model = plrnn.PLRNN(par)

    # get data
    if args.load:
        true_model.load_state_dict(tc.load('save/true_model.pt'))
        X_true = tc.load('save/X_true.pt')
        Z_true = tc.load('save/Z_true.pt')
    else:  # save
        X_true, Z_true = true_model.get_timeseries()
        tc.save(X_true, 'save/X_true.pt')
        tc.save(Z_true, 'save/Z_true.pt')
        tc.save(true_model.state_dict(), 'save/true_model.pt')

    true_ll = true_model.eval_logdensity(X_true, Z_true).detach().numpy()

    # add hyperparameters to txt file
    def save_args():
        d = args.__dict__
        txt = 'Hyperparameters:\n'
        txt += '{} {}\n'.format('run', run)
        for k in d.keys():
            txt += '{} {}\n'.format(k, d[k])
        txt += '{} {}\n'.format('true ll', true_ll)
        return txt
    txt = save_args()
    writer.add_text(tag='hypers', text_string=txt, global_step=0)  # TODO

    # fit model
    rec_model = models.SequentialVAE(X=X_true, ts_par=par)
    gen_model = plrnn.PLRNN(par)
    trainer_ = trainer.Trainer(ts_par=par, writer=writer, X_true=X_true, rec_model=rec_model, gen_model=gen_model, v=args.v, lr=args.lr, n_epochs=args.n)
    trainer_.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Estimate PLRNN with Sequential Variational Autoencoder")
    parser.add_argument('--name', type=str, default='seqvae', help="name of experiment")
    parser.add_argument('--load', type=bool, default=True, help="load or create times series")
    parser.add_argument('--comment', type=str, default='', help="description of experiment")
    parser.add_argument('--v', type=int, default=1, help="verbose of printout: 0 none, 1 some")
    parser.add_argument('--dx', type=int, default=5, help="dimension of observation space")
    parser.add_argument('--dz', type=int, default=2, help="dimension of latent space")
    parser.add_argument('--T', type=int, default=100, help="length of time series")
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate of Adam optimizer")
    parser.add_argument('--n', type=int, default=1000, help="number of epochs")
    main(parser.parse_args())

