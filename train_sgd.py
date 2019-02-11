import plrnn
import torch as tc
import torch.optim as optim


def train_sgd(true_model):
    """
    Train a PLRNN directly with SGD (Stochastic Gradient Descent) on the complete log likelihood log p(X,Z)
    :param true_model: original plrnn model
    :return: model: trained model
    """

    X, Z = true_model.forward()
    X = X.detach()
    Z = Z.detach()
    print(true_model.eval_logdensity(X, Z))

    cuda = False
    device = tc.device("cuda" if cuda else "cpu")
    model = plrnn.PLRNN(true_model.nz, true_model.nx, true_model.len_t).to(device)
    print(true_model.eval_logdensity(X, Z))
    print(model.eval_logdensity(X, model.Z))

    n_epochs = 15000
    log_interval = int(n_epochs / 10)
    lr = 2e-4
    optimizer = optim.Adam(model.parameters(), lr=lr)

    def train(epoch):
        model.train()
        optimizer.zero_grad()
        loss = - model.eval_logdensity(X, model.Z)  # fit other model observations with own states
        # writer.add_scalar('Log Density', net.evalLogDensity(X=X, Z=Z), iteration)
        loss.backward()
        optimizer.step()
        if epoch % log_interval == 0:
            print('Epoch: {} Log Density: {:.4f}'.format(epoch, -loss.item()))

    for epoch in range(1, n_epochs + 1):
        train(epoch)
    return model
