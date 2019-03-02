import torch.nn as nn
import torch.nn.functional as F
import torch as tc
import numpy as np
from matplotlib import pyplot as plt


# TODO add LDS

class PLRNN(nn.Module):
    """
    Piece-wise linear recurrent neural network
    z(0) ~ N(z0, Q0 * Q0')
    z(t) ~ N(A z(t-1) + W max(z(t-1),0) + h + K s(t), Q * Q')
    x(t) ~ N(NN(z(t)), R * R')

    """
    # TODO incorporate K st
    # TODO add generalization to non-diagonal covariance Q, R
    def __init__(self, args):
        super(PLRNN, self).__init__()
        self.dim_x = args.dx
        self.dim_z = args.dz
        self.len_t = args.T

        f = 0.5  # AW needs to be positive semi-definite, works ok
        mat = (tc.rand(self.dim_z, self.dim_z) - 1) * f # some matrix with values between -0.5 and 0.5
        mat = tc.mm(mat, mat.t())  # symmetric positive semi-def.
        max_ev = max(tc.symeig(mat)[0])
        mat /= (max_ev * 1.3)  # stability criterion: divide by eigenvalues
        self.AW = nn.Parameter(mat)

        self.h = nn.Parameter(tc.randn(self.dim_z, ))  # first dim is batch
        self.B = nn.Parameter(tc.randn(self.dim_x, self.dim_z))

        self.Q0 = nn.Parameter(tc.rand(self.dim_z, ))  # covariance matrix diagonals
        self.Q = nn.Parameter(tc.rand(self.dim_z, ))
        self.R = nn.Parameter(tc.rand(self.dim_x, ))

        self.Z = nn.Parameter(tc.randn(self.dim_z, self.len_t))
        self.z0 = nn.Parameter(tc.randn(self.dim_z))


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

    def get_timeseries(self, z0=None, T=None, noise=True, detach=True):
        if T is None:
            T = self.len_t

        Z = tc.randn(self.dim_z, T)
        A = tc.diag(tc.diag(self.AW))
        W = self.AW - A
        if z0 is None:
            zt = self.Q0 * tc.randn(self.dim_z)  # z0
        else:
            zt = z0
        for t in range(T):
            if noise:
                eps = self.Q * tc.randn(self.dim_z)
            else:
                eps = tc.zeros(self.dim_z)
            zt = A @ zt + W @ F.relu(zt) + self.h + eps  # TODO make nn module
            Z[:, t] = zt

        X = tc.zeros((self.dim_x, T))
        for t in range(T):
            if noise:
                eta = self.R * tc.randn(self.dim_x)  # TODO specify seed
            else:
                eta = tc.zeros(self.dim_x)
            xt = self.B @ Z[:, t] + eta
            X[:, t] = xt

        if detach:
            Z = Z.detach()
            X = X.detach()  # interrupts loss signal
        return X, Z  # TODO return Z.t()

    def eval_logdensity(self, X, Z):
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



class PLRNNsyn(nn.Module):
    """
    Piece-wise linear recurrent neural network
    z(0) ~ N(z0, Q0 * Q0')
    z(t) ~ N(A z(t-1) + W max(z(t-1),0) + h + K s(t), Q * Q')
    x(t) ~ N(NN(z(t)), R * R')

    """
    # TODO incorporate K st
    def __init__(self, args):
        super(PLRNNsyn, self).__init__()
        self.dim_x = args.dx
        self.dim_z = args.dz
        self.len_t = args.T

        f = 0.5  # AW needs to be positive semi-definite, works ok
        mat = (tc.rand(self.dim_z, self.dim_z) - 1) * f # some matrix with values between -0.5 and 0.5
        mat = tc.mm(mat, mat.t())  # symmetric positive semi-def.
        max_ev = max(tc.symeig(mat)[0])
        mat /= (max_ev * 1.3)  # stability criterion: divide by eigenvalues
        self.AW1 = nn.Parameter(mat)
        self.AW2 = nn.Parameter(mat)

        self.h1 = nn.Parameter(tc.randn(self.dim_z, ))  # first dim is batch
        self.h2 = nn.Parameter(tc.randn(self.dim_z, ))  # first dim is batch
        self.B = nn.Parameter(tc.randn(self.dim_x, self.dim_z))

        self.Q0 = nn.Parameter(tc.rand(self.dim_z, ))  # covariance matrix diagonals
        self.Q = nn.Parameter(tc.rand(self.dim_z, ))
        self.R = nn.Parameter(tc.rand(self.dim_x, ))

        self.Z = nn.Parameter(tc.randn(self.dim_z, self.len_t))
        self.z0 = nn.Parameter(tc.randn(self.dim_z))

    def print_par(self):
        print('PLRNN parameters:\n')
        np.set_printoptions(precision=2)
        print('A1={}\n'.format(tc.diag(tc.diag(self.AW1))))
        print('W1={}\n'.format(self.AW1 - tc.diag(self.AW1)))
        print('A2={}\n'.format(tc.diag(tc.diag(self.AW2))))
        print('W2={}\n'.format(self.AW2 - tc.diag(self.AW2)))
        print('h1={}\n'.format(self.h1))
        print('h2={}\n'.format(self.h2))
        print('B={}\n'.format(self.B))
        print('Q={}\n'.format(self.Q))
        print('Q0={}\n'.format(self.Q0))
        print('R={}\n'.format(self.R))

    def get_timeseries(self, z0=None, a0=None, T=None, noise=True, detach=True):
        if T is None:
            T = self.len_t

        Z = tc.randn(self.dim_z, T)
        A1 = tc.diag(tc.diag(self.AW1))
        A2 = tc.diag(tc.diag(self.AW2))
        W1 = self.AW1 - A1
        W2 = self.AW2 - A2
        if z0 is None:
            zt = self.Q0 * tc.randn(self.dim_z)  # z0
        else:
            zt = z0
        if a0 is None:
            at = self.Q0 * tc.randn(self.dim_z)  # z0  # TODO change Q0
        else:
            at = a0

        for t in range(T):
            if noise:
                eps = self.Q * tc.randn(self.dim_z)
            else:
                eps = tc.zeros(self.dim_z)

            at = A1 @ at + W1 @ F.relu(zt)
            bt = F.sigmoid(zt) * (self.h1 - zt)
            zt = A2 @ zt + W2 @ (at * bt) + self.h2 + eps
            Z[:, t] = zt

        X = tc.zeros((self.dim_x, T))
        for t in range(T):
            if noise:
                eta = self.R * tc.randn(self.dim_x)
            else:
                eta = tc.zeros(self.dim_x)
            xt = self.B @ Z[:, t] + eta
            X[:, t] = xt

        if detach:
            Z = Z.detach()
            X = X.detach()  # interrupts loss signal
        return X, Z

    def eval_logdensity(self, X, Z):
        A1 = tc.diag(tc.diag(self.AW1))
        A2 = tc.diag(tc.diag(self.AW2))
        W1 = self.AW1 - A1
        W2 = self.AW2 - A2
        T = X.shape[1]

        resX = X - self.B @ Z

        bt = F.sigmoid(Z) * (self.h1 - Z)
        at = tc.zeros(self.dim_z, T)
        for t in range(T):
            at = A1 @ at * W1 @ F.relu(Z[:, t])
            at[:, t] = at

        Ztm1 = Z[:, 0:(T-1)]
        resZ = Z[:, 1:] - A2 @ Ztm1 - at * bt - self.h.unsqueeze(1)

        resZ0 = (self.z0 - Z[:, 0]).unsqueeze(1)

        def dist(res, mat):
            return - 0.5 * (res.t() @ tc.inverse(tc.diag(mat**2)) @ res).sum()

        def logdet(mat):
            return 0.5 * tc.log(tc.det(tc.diag(1/(mat**2))))

        return dist(resX, self.R) + dist(resZ, self.Q) + dist(resZ0, self.Q0) \
            + logdet(self.R) * T + logdet(self.Q) * (T-1) + logdet(self.Q0)\
            - 0.5 * (self.dim_z + self.dim_x) * np.log(2*np.pi) * T
