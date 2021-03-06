import torch.nn as nn
import torch as tc
import torch.nn.functional as F
import numpy as np
import calc_blk as calc
from matplotlib import pyplot as plt

# TODO add RNN
# TODO add VAE


class VAE(nn.Module):
    def __init__(self, n_input, n_z):
        super(VAE, self).__init__()
        self.n_input = n_input
        self.n_z = n_z
        n_hidden = 10
        self.fc1 = nn.Linear(n_input, n_hidden)
        # self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc21 = nn.Linear(n_hidden, n_z)
        self.fc22 = nn.Linear(n_hidden, n_z)
        self.fc3 = nn.Linear(n_z, n_input, bias=False)  # observation model

    def encode(self, x):
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        return self.fc21(x), self.fc22(x)

    def reparameterize(self, mu, logvar):
        std = tc.exp(0.5*logvar)
        eps = tc.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        return self.fc3(F.relu(z))

    def forward(self, x):
        x = x.view(-1, self.n_input)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class DVAE(nn.Module):
    def __init__(self, args, X_true):
        super(DVAE, self).__init__()
        self.X_true = X_true
        self.n_input = args.dx
        self.n_z = args.dz
        n_hidden = 10
        self.fc1 = nn.Linear(self.n_input, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, self.n_z)

        self.h = nn.Parameter(tc.randn(1, self.n_z))  # first dim is batch
        self.AW = nn.Parameter(tc.randn(self.n_z, self.n_z))

        self.logS = nn.Parameter(tc.randn(1, self.n_z))
        self.B = nn.Linear(self.n_z, self.n_input, bias=False)  # observation model

    def print_par(self):
        A = tc.diag(tc.diag(self.AW))
        W = self.AW - A
        print('A={}\n'.format(A))
        print('B={}\n'.format(self.B.weight))
        print('W={}\n'.format(W))
        print('h={}\n'.format(self.h))
        print('S={}\n'.format(tc.diag(tc.exp(self.logS).reshape([self.n_z,]))))

    def encode(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        z = self.fc3(x)
        return z

    def plrnn_step(self, z):
        A = tc.diag(tc.diag(self.AW))
        W = self.AW - A
        a = tc.mm(z, A)

        w = tc.mm(F.relu(z), W)
        z = tc.add(tc.add(a, w), self.h)
        return z

    def reparametrize(self, mu):
        std = tc.exp(0.5*self.logS)
        eps = tc.randn_like(std)
        return eps.mul(std) + mu

    def decode(self, z):
        return F.relu(self.B(z))

    def forward(self, x):
        x = x.view(-1, self.n_input)
        zt1 = self.encode(x)
        mu = self.plrnn_step(zt1)
        zt2 = self.reparametrize(mu)
        return self.decode(zt1), self.decode(zt2), mu, self.logS


class StaticVAE(nn.Module):
    """
    Define a mean field variational approximate posterior (Recognition Model). Here,
    "mean field" is over time, so that for x = (x_1, \dots, x_t, \dots, x_T):

    x ~ \prod_{t=1}^T N( mu_t(y_t), sigma_t(y_t) ).

    Each covariance sigma_t is a full [n x n] covariance matrix (where n is the size
    of the latent space).
    """

    def __init__(self, X, dim_z, dim_x):
        super(StaticVAE, self).__init__()
        self.dim_z = dim_z
        self.dim_x = dim_x
        self.dim_h = 10
        self.X = X.clone().detach().t()  # no need for dataloader, only one sample.
        self.T = self.X.shape[0]

        self.fc1 = nn.Linear(self.dim_x, self.dim_h)
        self.fc_mu = nn.Linear(self.dim_h, self.dim_z)
        self.fc_lam = nn.Linear(self.dim_h, self.dim_z*self.dim_z)

    def encode(self, x):
        x = x.view(-1, self.dim_x)
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        lam = self.fc_lam(x)
        return mu, lam.view(-1, self.dim_z, self.dim_z)

    def reparametrize(self, mu, lam):
        # TODO is Lambda already Cholesky decomposed? -> No: TODO
        # TODO did Archer et.al. forget that??
        # shape mu (b, z)
        # shape Lambda (b, z, z)
        eps = tc.randn_like(mu)
        s_ = tc.einsum('tgh,th->tg', lam, eps)  # t is time, g,h are dim_z
        # def step(samp_rt, nsampt):
            # return tc.mm(nsampt, samp_rt.t())
        # for t in range(self.T):
        # s_ = theano.scan(fn=step, sequences=[self.LambdaChol, eps])[0]
        return mu + s_

    def forward(self, x):
        mu, lam = self.encode(x)
        Z = self.reparametrize(mu, lam)
        return Z

    def eval_entropy(self):
        x = self.X  # "sample" observations (only ONE timeline)
        _, lam = self.encode(x)
        # TODO (saving parameters in NN, as only tridiagonal number is needed)

        def comp_logdet(Rt):
            return tc.log(tc.abs(tc.det(Rt)))  # no 1/2 as S = Rt*Rt

        det_ = tc.empty(self.T)
        for t in range(lam.shape[0]):
            det_[t] = comp_logdet(lam[t, :, :])
        return det_.sum() + self.dim_x * self.T/2. * (1 + np.log(2 * np.pi))

    def sample(self):
        x = self.X  # "sample" observations (only ONE timeline)
        z = self.forward(x)
        return z


class SequentialVAE(nn.Module):

    def __init__(self, X, args):
        super(SequentialVAE, self).__init__()
        self.dim_z = args.dz
        self.dim_x = args.dx
        self.dim_h = self.dim_x  # TODO tune hyper-parameters
        self.X = X.clone().detach().t()  # no need for dataloader, only one sample.
        self.T = args.T

        self.fc1 = nn.Linear(self.dim_x, self.dim_h)  # TODO tune architecture
        self.fc_mu = nn.Linear(self.dim_h, self.dim_z)
        self.fc_A = nn.Linear(self.dim_h, self.dim_z * self.dim_z)  # A = NN(xt)
        self.fc_B = nn.Linear(2 * self.dim_h, self.dim_z * self.dim_z)  # B = NN(xt, xt-1)

    def encode(self, x):
        """
        Encode mean (mu) and blocks of covariance matrix (A diagonal, B off-diagonal)
        :param x: observation: (T, dim_x)
        :return: mu, A, B
        """
        x = x.view(-1, self.dim_x)  # TODO account for batchsize
        x = F.relu(self.fc1(x))

        mu = self.fc_mu(x)
        A = self.fc_A(x)
        B = self.fc_B(tc.cat((x[1:], x[:-1]), dim=1))

        A = A.view(-1, self.dim_z, self.dim_z) + tc.eye(self.dim_z)  # TODO is adding the tc.eye really necessary? why?
        B = B.view(-1, self.dim_z, self.dim_z)

        return mu, A, B

    def make_psd(self, A, B):  # TODO needs review, somehow sometimes not positive semi-definite
        """
        Make a block-tridiagonal matrix positive semi-definite: amounts to mat*mat^T
        :param A, B: diagonal, off-diagonal blocks. shape: (T, dim_z, dim_z), (T-1, dim_z, dim_z)
        :return: AA, BB: blocks in psd
        """
        def b_t(mat):  # batch transpose
            return tc.einsum('bij->bji', mat)
        diag_square = tc.bmm(A, b_t(A))  # make pos. semi-def.
        offd_square = tc.bmm(B, b_t(B))  # shape (T-1, dim_z, dim_z)
        offd_square = tc.cat((tc.zeros(1, self.dim_z, self.dim_z), offd_square))  # reshape from T-1 to T
        pos_const = 1e-6  # this seems to be needed, is the matrix not pos. semi-def. already?
        AA = tc.add(diag_square, offd_square).add(pos_const * tc.eye(self.dim_z))  # + positive constant
        # BB = tc.bmm(A[:-1], b_t(B))  # what is this doing?
        BB = tc.bmm(B, b_t(A[:-1]))  # what is this doing?
        # def is_psd(self, mat):
        # ev, _ = tc.symeig(mat)
        # return (ev >= 0).sum() == len(ev)  # are all Eigenvalues greater equal zero?
        # for t in range(self.len_t):
        # assert self.is_psd(AA[t])
        # TODO how to assert that overall matrix is psd?
        return AA, BB

    # def forward(self, X):
        # mu, A, B = self.encode(X)  # encode mean and covariance with neural net on observations
        # AA, BB = self.make_psd(A, B)  # make positive semi-definite
        # chol_A, chol_B = h.blk_tridag_chol(AA, BB)  # get cholesky from block tri-diagonal
        # return mu, chol_A, chol_B

    def forward(self):
        X = self.X  # (T, dim_X)
        mu, A, B = self.encode(X)  # encode mean and covariance with neural net on observations: mu(T, dim_z), A(T, dim_z, dim_z)
        AA, BB = self.make_psd(A, B)  # make positive semi-definite
        chol_A, chol_B = calc.blk_tridag_chol(AA, BB)  # get cholesky from block tri-diagonal
        r = self.get_inv_chol(chol_A, chol_B)

        # eval entropy
        thresh = 1e-5
        r = tc.add(F.relu(tc.add(r, -thresh)), thresh)  # make positive, negative value would be nan after log
        logdet_ = - tc.log(tc.einsum('tii->ti', r)).sum()  # -2 * 1/2
        entropy = logdet_ + self.dim_x * self.T / 2. * (1 + np.log(2 * np.pi))

        # sample
        eps = tc.randn_like(mu)  # sample from normal distr. with mean mu and std s
        r = tc.einsum('tij->tji', r)  # transpose TODO could be simplified
        sample_z = mu + tc.einsum('tij,tj->ti', r, eps)  # equivalent to tc.bmv

        return entropy, sample_z

    def get_inv_chol(self, A, B):
        b = tc.diag(tc.ones(self.dim_z)).expand(1, self.dim_z, self.dim_z).repeat(self.T, 1, 1)  # identity matrix
        inv_chol = calc.blk_chol_inv(A, B, b)
        return inv_chol
