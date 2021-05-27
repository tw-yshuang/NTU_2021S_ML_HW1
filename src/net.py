import torch.nn as nn


class Net_01(nn.Module):
    '''A simple fully-connected deep neural network'''

    def __init__(self, input_dim):
        super(Net_01, self).__init__()

        # Define your neural network here
        # TODO: How to modify this model to achieve better performance?
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        # Mean squared error loss
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, x):
        '''Given input of size (batch_size x input_dim), compute output of the network'''
        return self.net(x).squeeze(1)
        # return self.net(x)

    def cal_loss(self, pred, target):
        '''Calculate loss'''
        # TODO: you may implement L1/L2 regularization here
        return self.criterion(pred, target)


class NeuralNet(nn.Module):
    '''A simple fully-connected deep neural network'''

    def __init__(self, input_dim):
        super(NeuralNet, self).__init__()

        # Define your neural network here
        # TODO: How to modify this model to achieve better performance?
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Mean squared error loss
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, x):
        '''Given input of size (batch_size x input_dim), compute output of the network'''
        return self.net(x).squeeze(1)

    def cal_loss(self, pred, target):
        '''Calculate loss'''
        # TODO: you may implement L1/L2 regularization here
        return self.criterion(pred, target)