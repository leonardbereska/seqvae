import torch as tc
import torch.optim as optim
import dataset


class Trainer:
    def __init__(self, args, writer, X_true, rec_model, gen_model):

        self.dim_x = args.dx
        self.dim_z = args.dz
        self.len_t = args.T

        self.X_true = X_true
        self.verbose = args.verbose

        tc.manual_seed(232)
        device = tc.device("cuda" if args.use_cuda else "cpu")
        self.kwargs = {'num_workers': 1, 'pin_memory': True} if args.use_cuda else {}  # do I need this?
        self.writer = writer

        self.rec_model = rec_model.to(device)
        self.gen_model = gen_model.to(device)
        self.optimizer = optim.Adam(list(self.rec_model.parameters())+list(gen_model.parameters()), lr=args.lr)

        self.n_epochs = args.n_epochs
        self.log_interval = self.n_epochs / 10

    # def loss_function(self, epoch):
        # raise NotImplementedError

    def train(self):
        def step(epoch):
            self.gen_model.train()
            # if self.rec_model is not None:
            self.rec_model.train()
            self.optimizer.zero_grad()
            loss = self.loss_function(epoch)
            loss.backward()
            # train_loss = loss.item()
            self.optimizer.step()

        # with autograd.set_detect_anomaly(True):  # for debugging
        for epoch in range(1, self.n_epochs + 1):
            step(epoch)


class SGDTrainer(Trainer):
    def __init__(self, args, writer, X_true, rec_model, gen_model):
        super(SGDTrainer, self).__init__(args, writer, X_true, rec_model, gen_model)
        # lr = 2e-4
        # n_epochs  15000

    def loss_function(self, epoch):
        logdensity = self.rec_model.eval_logdensity(self.X_true, self.rec_model.Z)
        self.writer.add_scalar(tag='likelihood', scalar_value=logdensity, global_step=epoch)
        if epoch % self.log_interval == 0 and self.verbose > 0:
                    print('Epoch: {} Log Density: {:.4f}'.format(epoch, logdensity))
        return - logdensity


class SeqVAETrainer(Trainer):
    def __init__(self, args, writer, X_true, rec_model, gen_model):
        super(SeqVAETrainer, self).__init__(args, writer, X_true, rec_model, gen_model)

    def loss_function(self, epoch):
            """
            Compute a one-sample approximation the ELBO (lower bound on marginal likelihood),
            normalized by batch size (length of Y in first dimension).
            """
            Z = self.rec_model.sample()
            Z = Z.t()
            # entropy, sample = self.rec_model.forward()  # batch of samples and entropy of covariance
            likelihood = self.gen_model.eval_logdensity(X=self.X_true, Z=Z)
            self.writer.add_scalar(tag='likelihood', scalar_value=likelihood, global_step=epoch)
            entropy = self.rec_model.eval_entropy()
            self.writer.add_scalar(tag='entropy', scalar_value=entropy, global_step=epoch)
            if epoch % self.log_interval == 0 and self.verbose > 0:
                    print('Epoch: {} Entropy: {:.4f} Likelihood: {:.4f}'.format(epoch, entropy, likelihood))
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
        r1, r2, mu, logvar = model(x1)
        MSE1 = self.mse(r1, x1.view(-1, net.N))
        MSE2 = self.mse(r2, x2.view(-1, net.N))
        KLD = -0.5 * tc.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return MSE1 + MSE2 + KLD

    def train(self):
        def step(epoch):
            model.train()
            train_loss = 0
            for batch_idx, (x1, x2, _) in enumerate(self.train_loader):
                x1 = x1.to(device)
                self.optimizer.zero_grad()
                loss = self.loss_function(r1, r2, x1, x2, mu, logvar)
                loss.backward()
                train_loss += loss.item()
                self.optimizer.step()
            if epoch % self.log_interval == 0:
                print('Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))

        for epoch in range(1, self.n_epochs + 1):
            step(epoch)

def train_dvae(net):
    # epochs = 1000
    # cuda = False
    # log_interval = 100

    # tc.manual_seed(232)
    # device = tc.device("cuda" if cuda else "cpu")

    optimizer = optim.Adam(model.parameters(), lr=2e-4)  # 1e-3

