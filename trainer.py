import torch as tc
import torch.optim as optim
# import dataset
import model_gen


class Trainer:
    def __init__(self, args, writer, Z_true, X_true, rec_model, gen_model, true_model):

        self.dim_x = args.dx
        self.dim_z = args.dz
        self.len_t = args.T

        self.Z_true = Z_true
        self.X_true = X_true
        self.verbose = args.verbose

        tc.manual_seed(232)
        self.device = tc.device("cuda" if args.use_cuda else "cpu")
        self.kwargs = {'num_workers': 1, 'pin_memory': True} if args.use_cuda else {}  # do I need this?
        self.writer = writer

        self.rec_model = rec_model.to(self.device)
        self.gen_model = gen_model.to(self.device)
        self.true_model = true_model
        self.optimizer = optim.Adam(list(self.rec_model.parameters())+list(gen_model.parameters()), lr=args.lr)

        self.n_epochs = args.n_epochs
        self.log_interval = self.n_epochs / 10

    # def loss_function(self, epoch):
        # raise NotImplementedError

    def train(self):
        def step(epoch):
            self.gen_model.train()
            self.rec_model.train()
            self.optimizer.zero_grad()
            loss = self.loss_function(epoch)
            loss.backward()
            train_loss = loss.item()
            self.optimizer.step()
            return train_loss

        # with autograd.set_detect_anomaly(True):  # for debugging
        # loss_prev = 0
        # diff_prev = 0
        # counter = 0
        # thresh = 1.2
        for epoch in range(1, self.n_epochs + 1):
            loss = step(epoch)
            # diff = loss - loss_prev
            # if diff > diff_prev:  # TODO think about convergence criterion
                # counter += 1
            # else:
                # counter = 0
            # loss_prev = loss
            # diff_prev = diff
            # if counter >= 5:
                # break

    def test(self):
        n_steps = self.len_t
        z0 = self.Z_true[:, -1]
        X_test, Z_test = self.gen_model.get_timeseries(z0=z0, T=n_steps, noise=True)
        return self.true_model.eval_logdensity(X_test, Z_test)


class SGDTrainer(Trainer):
    def __init__(self, args, writer, Z_true, X_true, rec_model, gen_model, true_model):
        super(SGDTrainer, self).__init__(args, writer, Z_true, X_true, rec_model, gen_model, true_model)
        # lr = 2e-4
        # n_epochs  15000

    def loss_function(self, epoch):
        logdensity = self.gen_model.eval_logdensity(self.X_true, self.rec_model.Z)
        self.writer.add_scalar(tag='likelihood', scalar_value=logdensity, global_step=epoch)
        if epoch % self.log_interval == 0 and self.verbose > 0:
                    test_ll = self.test()
                    print('Epoch: {} Train LL: {:.4f} Test LL: {:.4f}'.format(epoch, logdensity, test_ll))
        return - logdensity


class SeqVAETrainer(Trainer):
    def __init__(self, args, writer, Z_true, X_true, rec_model, gen_model, true_model):
        super(SeqVAETrainer, self).__init__(args, writer, Z_true, X_true, rec_model, gen_model, true_model)

    def loss_function(self, epoch):
            """
            Compute a one-sample approximation the ELBO (lower bound on marginal likelihood),
            normalized by batch size (length of Y in first dimension).
            """
            # TODO batch samples
            entropy, Z_sample = self.rec_model.forward()  # batch of samples and entropy of covariance
            likelihood = self.gen_model.eval_logdensity(X=self.X_true, Z=Z_sample.t())
            self.writer.add_scalar(tag='likelihood', scalar_value=likelihood, global_step=epoch)
            self.writer.add_scalar(tag='entropy', scalar_value=entropy, global_step=epoch)
            if epoch % self.log_interval == 0 and self.verbose > 0:
                    test_ll = self.test()
                    print('Epoch: {} Entropy: {:.4f} Likelihood: {:.4f} Test LL: {:.4f}'.format(epoch, entropy, likelihood, test_ll))
            return - (likelihood + entropy) / self.len_t




class DVAETrainer(Trainer):
    def __init__(self, args, writer, X_true, rec_model, gen_model):
        super(DVAETrainer, self).__init__(args, writer, X_true, rec_model, gen_model)
        # lr = 2e-4
        # n_epochs  15000

        batch_size = 128

        train_set = dataset.Pair(self.X_true)
        self.train_loader = tc.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, **self.kwargs)
        self.mse = tc.nn.MSELoss()

    def loss_function(self, r1, r2, x1, x2, mu, logvar):
        MSE1 = self.mse(r1, x1.view(-1, self.dim_x))
        MSE2 = self.mse(r2, x2.view(-1, self.dim_x))
        KLD = -0.5 * tc.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return MSE1 + MSE2 + KLD

    def train(self):
        def step(epoch):
            self.rec_model.train()
            train_loss = 0
            for batch_idx, (x1, x2, _) in enumerate(self.train_loader):
                x1 = x1.to(self.device)
                self.optimizer.zero_grad()
                r1, r2, mu, logvar = self.rec_model(x1)
                loss = self.loss_function(r1, r2, x1, x2, mu, logvar)
                loss.backward()
                train_loss += loss.item()
                self.optimizer.step()
            if epoch % self.log_interval == 0 and self.verbose > 0:
                print('Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(self.train_loader.dataset)))

            # not part of the loss, stop gradients here
            Z = self.rec_model.encode(self.X_true.t()).detach()
            # print(Z)
            # self.gen_model.AW = self.rec_model.AW.detach()
            # self.gen_model.B = self.rec_model.B.detach()
            # self.gen_model.h = self.rec_model.h.detach()
            likelihood = self.gen_model.eval_logdensity(X=self.X_true, Z=Z.t())
            # AW=self.rec_model.AW, B=self.rec_model.B.weight, h=self.rec_model.h)  # TODO z0, Q, R, Q0 wrong
            self.writer.add_scalar(tag='likelihood', scalar_value=likelihood, global_step=epoch)

        for epoch in range(1, self.n_epochs + 1):
            step(epoch)

