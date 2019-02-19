import torch as tc
import torch.optim as optim


class Trainer:
    def __init__(self, ts_par, X_true, rec_model, gen_model, lr=1e-3, n_epochs=1000):

        # instantiate rng's
        # self.srng = np.random.RandomStreams(seed=234)
        # self.nrng = np.random.RandomState(124)

        # tc.manual_seed(232)
        # device = tc.device("cuda" if self.cuda else "cpu")
        # kwargs = {'num_workers': 1, 'pin_memory': True} if self.cuda else {}

        # batch_size = 128
        self.dim_x = ts_par['dim_x']
        self.dim_z = ts_par['dim_z']
        self.len_t = ts_par['len_t']

        self.X_true = X_true

        cuda = False
        tc.manual_seed(232)
        device = tc.device("cuda" if cuda else "cpu")
        kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

        self.rec_model = rec_model.to(device)
        self.gen_model = gen_model.to(device)
        self.optimizer = optim.Adam(list(self.rec_model.parameters())+list(gen_model.parameters()), lr=lr)

        # instantiate our prior & recognition models
        # self.mrec = rec_model(rec_params, self.Y, self.xDim, self.yDim, self.srng, self.nrng)
        # self.mprior = gen_model(gen_params, self.xDim, self.yDim, srng=self.srng, nrng=self.nrng)

        # self.isTrainingRecognitionModel = True
        # self.isTrainingGenerativeModel = True

        self.epochs = n_epochs
        self.log_interval = self.epochs/10

    def loss_function(self):
        """
        Compute a one-sample approximation the ELBO (lower bound on marginal likelihood),
        normalized by batch size (length of Y in first dimension).
        """
        Z = self.rec_model.sample()
        Z = Z.t()
        likelihood = self.gen_model.eval_logdensity(X=self.X_true, Z=Z)
        entropy = self.rec_model.eval_entropy()
        # print('entropy {}'.format(entropy))
        # print('likelihood {}'.format(likelihood))
        return - (likelihood + entropy) / self.len_t

    def train(self):
        def step(epoch):
            self.gen_model.train()
            self.rec_model.train()
            self.optimizer.zero_grad()
            loss = self.loss_function()
            loss.backward()
            train_loss = loss.item()
            self.optimizer.step()
            if epoch % self.log_interval == 0:
                print('Epoch: {} Loss: {:.4f}'.format(epoch, train_loss))

        # with autograd.set_detect_anomaly(True):  # for debugging
        for epoch in range(1, self.epochs + 1):
            step(epoch)


# TODO plot function
# TODO plot mean field vae
# TODO plot correlation
# TODO plot log likelihood

