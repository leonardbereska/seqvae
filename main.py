from __future__ import print_function

# import plrnn
import model_gen
import model_rec
import trainer

import argparse
import torch as tc
import os.path
from tensorboardX import SummaryWriter  # install and use according to https://github.com/lanpa/tensorboardX


def main(args):

    def init_writer(args):
        if args.verbose > 0:
            print('init writer..')
        log_dir = 'runs/{}'.format(args.name)
        if not os.path.exists('./'+log_dir):
            os.mkdir('./'+log_dir)

        run_nrs = list(os.walk(log_dir))[0][1]
        run_nrs = [r[0:3] for r in run_nrs]
        # print(run_nrs)
        if not run_nrs:
            run_nrs = ['000']
        run = str(int(max(run_nrs))+1).zfill(3)
        writer = SummaryWriter(log_dir='{}/{}-{}'.format(log_dir, run, args.comment))
        return writer

    writer = init_writer(args)


    # initialize plrnn
    true_model = model_gen.PLRNN(args)

    print('get data..')
    dir_ = 'save/dx{}dz{}'.format(args.dx, args.dz)
    if not os.path.exists(dir_):
            os.mkdir(dir_)
    # print(args.load)
    # print(args)
    if args.load:
        print('load time series')
        true_model.load_state_dict(tc.load('{}/true_model.pt'.format(dir_)))
        X_true = tc.load('{}/X_true.pt'.format(dir_))
        Z_true = tc.load('{}/Z_true.pt'.format(dir_))
    else:  # save
        print('save groundtruth time series')
        X_true, Z_true = true_model.get_timeseries()
        tc.save(X_true, '{}/X_true.pt'.format(dir_))
        tc.save(Z_true, '{}/Z_true.pt'.format(dir_))
        tc.save(true_model.state_dict(), '{}/true_model.pt'.format(dir_))

    if args.verbose > 2:
        true_model.show(X_true=X_true, Z_true=Z_true, block=True)
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
    print(txt)
    writer.add_text(tag='hypers', text_string=txt, global_step=0)  # TODO why does the txt not load in tensorboard?

    print('fit model: {}'.format(args.optim))
    if args.optim == 'seqvae':
        rec_model = model_rec.SequentialVAE(X=X_true, args=args)
        gen_model = model_gen.PLRNN(args=args)
        trainer_ = trainer.SeqVAETrainer(args=args, writer=writer, X_true=X_true, rec_model=rec_model, gen_model=gen_model)
        trainer_.train()

    elif args.optim == 'sgd':
        rec_model = model_gen.PLRNN(args=args)
        trainer_ = trainer.SGDTrainer(args, writer, X_true, rec_model=rec_model, gen_model=rec_model)  # hacky
        trainer_.train()

    elif args.optim == 'dvae':
        raise NotImplementedError
        # rec_model = model_rec.DVAE(args=args, X_true=X_true)
        # gen_model = model_gen.PLRNN(args=args)
        # trainer_ = trainer.DVAETrainer(args=args, writer=writer, X_true=X_true, rec_model=rec_model, gen_model=gen_model)
        # trainer_.train()

    # elif args.optim == 'statvae':  # TODO
        # rec_model = model_gen.PLRNN(args)
        # trainer_ = trainer.Tra(args, writer, X_true, rec_model, rec_model)
        # trainer_.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Estimate PLRNN with Sequential Variational Autoencoder")
    parser.add_argument('--name', type=str, default='states', help="name of experiment")
    parser.add_argument('--load', default=False, action='store_true', help="load or create times series")
    parser.add_argument('--comment', type=str, default='', help="description of experiment")
    parser.add_argument('--verbose', type=int, default=1, help="verbose of printout: 0 none, 1 some")
    parser.add_argument('--use_cuda', default=False, action='store_true', help="use GPU acceleration with CUDA")  # TODO test
    parser.add_argument('--dx', type=int, default=50, help="dimension of observation space")
    parser.add_argument('--dz', type=int, default=5, help="dimension of latent space")
    parser.add_argument('--T', type=int, default=100, help="length of time series")
    parser.add_argument('--lr', type=float, default=2e-4, help="learning rate of Adam optimizer")
    parser.add_argument('--n_epochs', type=int, default=1000, help="number of epochs")
    parser.add_argument('--optim', type=str, choices=['seqvae', 'sgd', 'dvae, None'], default='None')
    main(parser.parse_args())

