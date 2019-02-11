import torch.nn as nn
import torch as tc
import numpy as np

class MeanFieldGaussian(nn.Module):
    """
    Define a mean field variational approximate posterior (Recognition Model). Here,
    "mean field" is over time, so that for x = (x_1, \dots, x_t, \dots, x_T):

    x ~ \prod_{t=1}^T N( mu_t(y_t), sigma_t(y_t) ).

    Each covariance sigma_t is a full [n x n] covariance matrix (where n is the size
    of the latent space).

    """

    def __init__(self, rec_par, X, dim_z, dim_x):
        """
        :parameters:
            - Input : 'y' theano.tensor.var.TensorVariable (n_input)
                Observation matrix based on which we produce q(x)
            - xDim, yDim, zDim : (integers) dimension of
                latent space (x), observation (y)
        """
        super(MeanFieldGaussian, self).__init__()
        self.dim_z = dim_z
        self.dim_x = dim_x
        self.dim_h = 10
        self.X = X
        self.T = X.shape[0]
        # self.mu_train = rec_par['NN_Mu']['is_train']
        # self.NN_Mu = rec_par['NN_Mu']['network']
        # self.postX = lasagne.layers.get_output(self.NN_Mu, inputs=self.X)
        # 
        # self.lambda_train = rec_par['NN_Lambda']['is_train']
        # self.NN_Lambda = rec_par['NN_Lambda']['network']
        # 
        # lambda_net_out = lasagne.layers.get_output(self.NN_Lambda, inputs=self.X)
        # self.LambdaChol = T.reshape(lambda_net_out, [self.T, dim_z, dim_z])
       
        self.fc1 = nn.Linear(self.dim_x, self.dim_h)
        self.fc2 = nn.Linear(self.dim_x, self.dim_h)
        self.mu = nn.Sequential(nn.Linear(self.dim_h, self.dim_z), nn.ReLU)
        self.Lambda = nn.Sequential(nn.Linear(self.dim_h, self.dim_z * self.dim_z), nn.ReLU)
       
    def encode_mu(self, x):
        x = self.fc1(x)
        return self.mu(x)

    def encode_lambda(self, x):
        x = self.fc2(x)
        return self.Lambda(x)

    # def reparameterize(self, mu, logvar):
    #     std = tc.exp(0.5*logvar)
    #     eps = tc.randn_like(std)
    #     return eps.mul(std).add_(mu)


    def eval_entropy(self):
        def comp_trace(Rt):
            return tc.log(tc.abs(tc.det(Rt)))  # 1/2 the log determinant

        det_, updates = theano.scan(fn=comp_trace, sequences=[self.LambdaChol])
        return det_.sum() + self.xDim * self.T / 2.0 * (1 + np.log(2 * np.pi))

    def sample(self):
        eps = tc.randn(self.T, self.dim_z)

        def step(samp_rt, nsampt):
            return tc.mm(nsampt, samp_rt.T)

        for t in range(self.T):

        s_ = theano.scan(fn=step, sequences=[self.LambdaChol, eps])[0]
        return s_ + self.encode_mu(self.X)

    # def get_par(self):
    #     network_params = lasagne.layers.get_all_params(self.NN_Mu) + lasagne.layers.get_all_params(self.NN_Lambda)
    #     return network_params
        
    # def get_summary(self, yy):
    #     out = {}
    #     out['xsm'] = numpy.asarray(self.postX.eval({self.X:yy}), dtype=theano.config.floatX)
    #     V = T.batched_dot(self.LambdaChol, self.LambdaChol.dimshuffle(0,2,1))
    #     out['Vsm'] = numpy.asarray(V.eval({self.X:yy}), dtype=theano.config.floatX)
    #     out['VVsm'] = np.zeros([yy.shape[0]-1, self.xDim, self.xDim]).astype(theano.config.floatX)
    #     return out