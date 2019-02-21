import torch as tc
import torch.optim as optim


class Trainer:
    def __init__(self, ts_par, writer, X_true, rec_model, gen_model, v, lr=1e-3, n_epochs=1000):

        self.dim_x = ts_par['dim_x']
        self.dim_z = ts_par['dim_z']
        self.len_t = ts_par['len_t']

        self.X_true = X_true
        self.v = v

        cuda = False
        tc.manual_seed(232)
        device = tc.device("cuda" if cuda else "cpu")
        kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
        self.writer = writer

        self.rec_model = rec_model.to(device)
        self.gen_model = gen_model.to(device)
        self.optimizer = optim.Adam(list(self.rec_model.parameters())+list(gen_model.parameters()), lr=lr)

        self.n_epochs = n_epochs
        self.log_interval = self.n_epochs / 10

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
        if epoch % self.log_interval == 0 and self.v > 0:
                print('Epoch: {} Entropy: {:.4f} Likelihood: {:.4f}'.format(epoch, entropy, likelihood))
        return - (likelihood + entropy) / self.len_t

    def train(self):
        def step(epoch):
            self.gen_model.train()
            self.rec_model.train()
            self.optimizer.zero_grad()
            loss = self.loss_function(epoch)
            loss.backward()
            # train_loss = loss.item()
            self.optimizer.step()

        # with autograd.set_detect_anomaly(True):  # for debugging
        for epoch in range(1, self.n_epochs + 1):
            step(epoch)
