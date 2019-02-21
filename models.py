import torch.nn as nn
import torch as tc
import torch.nn.functional as F
import numpy as np
import helpers as h


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
        x = x.view(-1, self.dim_x)  # TODO account for batchsize
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

    def __init__(self, X, ts_par, hidden_size=10):
        super(SequentialVAE, self).__init__()
        self.dim_z = ts_par['dim_z']
        self.dim_x = ts_par['dim_x']
        self.dim_h = self.dim_x  # TODO tune hyper-parameters
        self.X = X.clone().detach().t()  # no need for dataloader, only one sample.
        self.T = ts_par['len_t']

        self.fc1 = nn.Linear(self.dim_x, self.dim_h)
        self.fc_mu = nn.Linear(self.dim_h, self.dim_z)
        self.fc_A = nn.Linear(self.dim_h, self.dim_z * self.dim_z)  # A = NN(xt)
        self.fc_B = nn.Linear(2 * self.dim_h, self.dim_z * self.dim_z)  # B = NN(xt, xt-1)

    def encode(self, x):
        """
        Encode mean (mu) and blocks of covariance matrix (A diagonal, B off-diagonal)
        :param x: observation at t
        :return: mu, A, B
        """
        x = x.view(-1, self.dim_x)  # TODO account for batchsize
        x = F.relu(self.fc1(x))

        mu = self.fc_mu(x)
        A = self.fc_A(x)
        B = self.fc_B(tc.cat((x[1:], x[:-1]), dim=1))

        A = A.view(-1, self.dim_z, self.dim_z)
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
        pos_const = 1  # this seems to be needed, is the matrix not pos. semi-def. already?
        AA = tc.add(diag_square, offd_square).add(pos_const * tc.eye(self.dim_z))  # + positive constant
        BB = tc.bmm(A[:-1], b_t(B))  # what is this doing?
        # def is_psd(self, mat):
        # ev, _ = tc.symeig(mat)
        # return (ev >= 0).sum() == len(ev)  # are all Eigenvalues greater equal zero?
        # for t in range(self.len_t):
        # assert self.is_psd(AA[t])
        # TODO how to assert that overall matrix is psd?
        return AA, BB

    def forward(self, X):
        mu, A, B = self.encode(X)  # encode mean and covariance with neural net on observations
        AA, BB = self.make_psd(A, B)  # make positive semi-definite
        chol_A, chol_B = h.blk_tridag_chol(AA, BB)  # get cholesky from block tri-diagonal
        return mu, chol_A, chol_B

    def get_inv_chol(self, A, B):
        b = tc.diag(tc.ones(self.dim_z)).expand(1, self.dim_z, self.dim_z).repeat(self.T, 1, 1)  # identity matrix
        inv_chol = h.blk_chol_inv(A, B, b)
        return inv_chol

    def eval_entropy(self):
        X = self.X
        mu, A, B = self.forward(X)
        r = self.get_inv_chol(A, B)
        thresh = 1e-5
        r = tc.add(F.relu(tc.add(r, -thresh)), thresh)  # make positive, negative value would be nan after log
        logdet_ = - tc.log(tc.einsum('tii->ti', r)).sum()  # -2 * 1/2
        return logdet_ + self.dim_x * self.T / 2. * (1 + np.log(2 * np.pi))

    def sample(self):
        X = self.X  # "sample" observations (only one time series)
        mu, A, B = self.forward(X)  # get mean and tridiagonal form of cholesky of covariance
        r = self.get_inv_chol(A, B)
        eps = tc.randn_like(mu)  # sample from normal distr. with mean mu and std s
        r = tc.einsum('tij->tji', r)  # transpose TODO could be simplified
        z = mu + tc.einsum('tij,tj->ti', r, eps)  # equivalent to tc.bmv
        return z  # shape (T, dim_z)
