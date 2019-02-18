import torch.nn as nn
import torch as tc
import torch.nn.functional as F
import numpy as np


class MeanFieldGaussian(nn.Module):
    """
    Define a mean field variational approximate posterior (Recognition Model). Here,
    "mean field" is over time, so that for x = (x_1, \dots, x_t, \dots, x_T):

    x ~ \prod_{t=1}^T N( mu_t(y_t), sigma_t(y_t) ).

    Each covariance sigma_t is a full [n x n] covariance matrix (where n is the size
    of the latent space).
    """

    def __init__(self, X, dim_z, dim_x):
        super(MeanFieldGaussian, self).__init__()
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


    # def reparameterize(self, mu, logvar):
    #     std = tc.exp(0.5*logvar)
    #     eps = tc.randn_like(std)
    #     return eps.mul(std).add_(mu)
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
        # self.encode_lambda()
        # det_, updates = theano.scan(fn=comp_trace, sequences=[self.LambdaChol])
        return det_.sum() + self.dim_x * self.T/2. * (1 + np.log(2 * np.pi))

    def sample(self):
        x = self.X  # "sample" observations (only ONE timeline)
        z = self.forward(x)
        return z


class SeqVAE(nn.Module):

    def __init__(self, X, dim_z, dim_x):
        super(SeqVAE, self).__init__()
        self.dim_z = dim_z
        self.dim_x = dim_x
        self.dim_h = 10
        self.X = X.clone().detach().t()  # no need for dataloader, only one sample.
        self.T = self.X.shape[0]

        self.fc1 = nn.Linear(self.dim_x, self.dim_h)
        self.fc_mu = nn.Linear(self.dim_h, self.dim_z)
        self.fc_lam_a = nn.Linear(self.dim_h, self.dim_z*self.dim_z)
        self.fc_lam_b = nn.Linear(self.dim_h, self.dim_z*self.dim_z)  # TODO xt xt-1

    def encode(self, x):
        x = x.view(-1, self.dim_x)  # TODO account for batchsize
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        lam_a = self.fc_lam_a(x)
        lam_b = self.fc_lam_b(x)
        lam_a.view(-1, self.dim_z, self.dim_z)
        lam_b.view(-1, self.dim_z, self.dim_z)
        return mu, lam_a, lam_b

    def eval_entropy(self):
        x = self.X
        inv_chol = self.forward(x)
        det_ = tc.log(tc.abs(1. / tc.det(inv_chol)))
        return det_ + self.dim_x * self.T / 2. * (1 + np.log(2 * np.pi))

    def sample(self):
        x = self.X  # "sample" observations (only ONE timeline)
        inv_chol = self.forward(x)
        mu = self.calc_mu(x)
        eps = tc.randn_like(mu)
        z = mu + inv_chol.t() @ eps
        return z

    def constr_cov(self, a, b):

        tc.cat([a, b, tc.zeros_like(a)], dim=0)

    def forward(self, x):
        mu, lam_a, lam_b = self.encode(x)
        assert(mu.shape == (self.T, self.dim_x))
        assert(lam_a.shape == (self.T, self.dim_z, self.dim_z))
        assert(lam_b.shape == (self.T-1, self.dim_z, self.dim_z))

        # TODO get cholesky efficient
        cov = constr_cov(lam_a, lam_b)
        chol = get_chol(cov)
        # TODO calc inverse efficient
        inv_chol = tc.inverse(chol)
        return inv_chol


class SmoothingTimeSeries(RecognitionModel):
    """
    A "smoothing" recognition model. The constructor accepts neural networks which are used to parameterize mu and Sigma.
    x ~ N( mu(y), sigma(y) )
    """

    def __init__(self,RecognitionParams,Input,xDim,yDim,srng = None,nrng = None):
        """
        :parameters:
            - Input : 'y' theano.tensor.var.TensorVariable (n_input)
                Observation matrix based on which we produce q(x)
            - RecognitionParams : (dictionary)
                Dictionary of timeseries-specific parameters. Contents:
                     * A -
                     * NN paramers...
                     * others... TODO
            - xDim, yDim, zDim : (integers) dimension of
                latent space (x) and observation (y)
        """
        super(SmoothingTimeSeries, self).__init__(Input,xDim,yDim,srng,nrng)

#        print RecognitionParams

        self.Tt = Input.shape[0]
        # These variables allow us to control whether the network is deterministic or not (if we use Dropout)
        self.mu_train = RecognitionParams['NN_Mu']['is_train']
        self.lambda_train = RecognitionParams['NN_Lambda']['is_train']

        # This is the neural network that parameterizes the state mean, mu
        self.NN_Mu = RecognitionParams['NN_Mu']['network']
        # Mu will automatically be of size [T x xDim]
        self.Mu = lasagne.layers.get_output(self.NN_Mu, inputs = self.Input)

        self.NN_Lambda = RecognitionParams['NN_Lambda']['network']
        lambda_net_out = lasagne.layers.get_output(self.NN_Lambda, inputs=self.Input)
        self.NN_LambdaX = RecognitionParams['NN_LambdaX']['network']
        lambdaX_net_out = lasagne.layers.get_output(self.NN_LambdaX, inputs=T.concatenate([self.Input[:-1], self.Input[1:]], axis=1))
        # Lambda will automatically be of size [T x xDim x xDim]
        self.AAChol = T.reshape(lambda_net_out, [self.Tt, xDim, xDim]) + T.eye(xDim)
        self.BBChol = T.reshape(lambdaX_net_out, [self.Tt-1, xDim, xDim]) #+ 1e-6*T.eye(xDim)

        self._initialize_posterior_distribution(RecognitionParams)

    def _initialize_posterior_distribution(self, RecognitionParams):

        ################## put together the total precision matrix ######################

        # Diagonals must be PSD
        diagsquare = T.batched_dot(self.AAChol, self.AAChol.dimshuffle(0,2,1))
        odsquare = T.batched_dot(self.BBChol, self.BBChol.dimshuffle(0,2,1))
        self.AA = diagsquare + T.concatenate([T.shape_padleft(T.zeros([self.xDim,self.xDim])), odsquare]) + 1e-6*T.eye(self.xDim)
        self.BB = T.batched_dot(self.AAChol[:-1], self.BBChol.dimshuffle(0,2,1))

        # compute Cholesky decomposition
        self.the_chol = blk_tridag_chol(self.AA, self.BB)

        # symbolic recipe for computing the the diagonal (V) and
        # off-diagonal (VV) blocks of the posterior covariance
        self.V, self.VV, self.S = compute_sym_blk_tridiag(self.AA, self.BB)
        self.postX = self.Mu

        # The determinant of the covariance is the square of the determinant of the cholesky factor (twice the log).
        # Determinant of the Cholesky factor is the product of the diagonal elements of the block-diagonal.
        def comp_log_det(L):
            return T.log(T.diag(L)).sum()
        self.ln_determinant = -2*theano.scan(fn=comp_log_det, sequences=self.the_chol[0])[0].sum()

    def getSample(self):
        normSamps = self.srng.normal([self.Tt, self.xDim])
        return self.postX + blk_chol_inv(self.the_chol[0], self.the_chol[1], normSamps, lower=False, transpose=True)

    def evalEntropy(self):
        return self.ln_determinant/2 + self.xDim*self.Tt/2.0*(1+np.log(2*np.pi))

