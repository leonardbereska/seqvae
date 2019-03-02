from __future__ import print_function

# import plrnn
import model_gen
import model_rec
import trainer
import helpers
import ../dynsyn/lorenz as lorenz

from matplotlib import pyplot as plt
import argparse
import torch as tc
import os.path


def main(args):

    writer, run = helpers.init_writer(args)
    true_model = model_gen.PLRNN(args)  # initialize plrnn

    # X_true, Z_true = helpers.load_ts(args, true_model)
    X_true, Z_true = true_model.get_timeseries(T=args.T)
    # true_model.show(X_true, Z_true, block=True)

    # print(tc.stack((Z_true[:,-10:], Z_test), dim=1).shape)
    # true_model.show(X_true[:,-10:], X_test[:,-10:], block=True)

    # txt = helpers.save_args(args, writer=writer)
    # print(txt)

    X_true, Y_true = lorenz.VanderPool(4., args.T)

    print('fit model: {}'.format(args.optim))
    if args.optim == 'seqvae':
        rec_model = model_rec.SequentialVAE(X=X_true, args=args)
        gen_model = model_gen.PLRNN(args=args)
        trainer_ = trainer.SeqVAETrainer(args=args, writer=writer, X_true=X_true, Z_true=Z_true, rec_model=rec_model, gen_model=gen_model, true_model=true_model)
        trainer_.train()
    if args.optim == 'sgd':
        # rec_model = model_rec.SGD(X=X_true, args=args)
        gen_model = model_gen.PLRNN(args=args)
        trainer_ = trainer.SGDTrainer(args=args, writer=writer, X_true=X_true, Z_true=Z_true, gen_model=gen_model, rec_model=gen_model, true_model=true_model)
        trainer_.train()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Estimate PLRNN with Sequential Variational Autoencoder")
    parser.add_argument('--name', type=str, default='test', help="name of experiment")
    parser.add_argument('--load', default=False, action='store_true', help="load or create times series")
    parser.add_argument('--comment', type=str, default='', help="description of experiment")
    parser.add_argument('--verbose', type=int, default=1, help="verbose of printout: 0 none, 1 some")
    parser.add_argument('--use_cuda', default=False, action='store_true', help="use GPU acceleration with CUDA")  # TODO test
    parser.add_argument('--dx', type=int, default=5, help="dimension of observation space")
    parser.add_argument('--dz', type=int, default=2, help="dimension of latent space")
    parser.add_argument('--T', type=int, default=100, help="length of time series")
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate of Adam optimizer")
    parser.add_argument('--n_epochs', type=int, default=1000, help="number of epochs")
    parser.add_argument('--optim', type=str, choices=['seqvae', 'sgd', 'dvae, None'], default='seqvae')
    main(parser.parse_args())
