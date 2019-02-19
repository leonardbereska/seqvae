# Build a network class
import torch.nn as nn
import torch.nn.functional as F
import torch as tc
import numpy as np
import numpy.random as r
from matplotlib import pyplot as plt


class PLRNN(nn.Module):
    """
    Piece-wise linear recurrent neural network
    z(0) ~ N(z0, Q0 * Q0')
    z(t) ~ N(A z(t-1) + W max(z(t-1),0) + h + K s(t), Q * Q')
    x(t) ~ N(NN(z(t)), R * R')

    """
    # TODO add stability criterion: divide by eigenvalues
    # TODO incorporate K st
    def __init__(self, ts_par):
        super(PLRNN, self).__init__()
        self.dim_x = ts_par['dim_x']
        self.dim_z = ts_par['dim_z']
        self.len_t = ts_par['len_t']

        f = 0.5  # AW needs to be positive semi-definite, works ok
        mat = f * (tc.rand(self.dim_z, self.dim_z) - 1)  # some matrix with values between -0.5 and 0.5
        self.AW = nn.Parameter(tc.mm(mat.t(), mat))
        self.h = nn.Parameter(tc.randn(self.dim_z, ))  # first dim is batch
        self.B = nn.Parameter(tc.randn(self.dim_x, self.dim_z))

        self.Q0 = nn.Parameter(tc.rand(self.dim_z, ))  # TODO add generalization to non-diagonal
        self.Q = nn.Parameter(tc.rand(self.dim_z, ))
        self.R = nn.Parameter(tc.rand(self.dim_x, ))

        self.Z = nn.Parameter(tc.randn(self.dim_z, self.len_t))
        self.z0 = nn.Parameter(tc.randn(self.dim_z))

        # TODO: create AW as a general positive semi-definite matrix
        # generate an (n + 1) × (n + 1) orthogonal matrix,
        # take an n × n one and a uniformly distributed unit vector of dimension n + 1.
        # Construct a Householder reflection from the vector, then apply it to the smaller matrix
        # (embedded in the larger size with a 1 at the bottom right corner).
        # make general matrix with U L V^T with U, V orthogonal and L diagonal
        # with all entries smaller one

    def plot_xz(self, block=False):
        plt.subplot(121)
        plt.title('latent states')
        plt.xlabel('time')
        plt.plot(list(range(0, self.len_t)), self.Z.t().detach().numpy())
        plt.subplot(122)
        plt.title('observations')
        plt.xlabel('time')
        plt.plot(list(range(0, self.len_t)), self.X.t().detach().numpy())
        plt.show(block=block)

    def print_par(self):
        print('PLRNN parameters:\n')
        np.set_printoptions(precision=2)
        print('A={}\n'.format(tc.diag(tc.diag(self.AW))))
        print('W={}\n'.format(self.AW - tc.diag(self.AW)))
        print('h={}\n'.format(self.h))
        print('B={}\n'.format(self.B))
        print('Q={}\n'.format(self.Q))
        print('Q0={}\n'.format(self.Q0))
        print('R={}\n'.format(self.R))

    def sample_latent(self):
        Z = tc.randn(self.dim_z, self.len_t)
        A = tc.diag(tc.diag(self.AW))
        W = self.AW - A
        zt = self.Q0 * tc.randn(self.dim_z)  # z0
        for t in range(self.len_t):
            eps = self.Q * tc.randn(self.dim_z)
            zt = A @ zt + W @ F.relu(zt) + self.h + eps  # TODO make nn module
            Z[:, t] = zt
        return Z

    def forward(self):
        Z = self.sample_latent()
        X = tc.zeros((self.dim_x, self.len_t))
        for t in range(self.len_t):
            eta = self.R * tc.randn(self.dim_x)  # TODO specify seed
            xt = self.B @ Z[:, t] + eta
            X[:, t] = xt
        return X, Z

    def get_timeseries(self):
        X, Z = self.forward()
        X = X.detach()
        Z = Z.detach()
        return X, Z

    def eval_logdensity(self, X, Z):
        # Z.shape = (T, dim_z)
        # X.shape = (T, dim_x)
        A = tc.diag(tc.diag(self.AW))
        W = self.AW - A
        T = X.shape[1]
        resX = X - self.B @ Z

        Zt = Z[:, 0:(T-1)]
        resZ = Z[:, 1:] - A @ Zt - W @ F.relu(Zt) - self.h.unsqueeze(1)
        resZ0 = (self.z0 - Z[:, 0]).unsqueeze(1)

        def dist(res, mat):
            return - 0.5 * (res.t() @ tc.inverse(tc.diag(mat**2)) @ res).sum()

        def logdet(mat):
            return 0.5 * tc.log(tc.det(tc.diag(1/(mat**2))))

        return dist(resX, self.R) + dist(resZ, self.Q) + dist(resZ0, self.Q0) \
            + logdet(self.R) * T + logdet(self.Q) * (T-1) + logdet(self.Q0)\
            - 0.5 * (self.dim_z + self.dim_x) * np.log(2*np.pi) * T


