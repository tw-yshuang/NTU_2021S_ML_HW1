import torch.nn as nn


class Net01(nn.Module):
    '''A simple fully-connected deep neural network'''

    def __init__(self, input_dim=1):
        super(Net01, self).__init__()

        # Define your neural network here
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        '''Given input of size (batch_size x input_dim), compute output of the network'''
        return self.net(x).squeeze(1)
        # return self.net(x)


class Net02(nn.Module):
    '''A simple fully-connected deep neural network'''

    def __init__(self, input_dim=1):
        super(Net02, self).__init__()

        # Define your neural network here
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        '''Given input of size (batch_size x input_dim), compute output of the network'''
        return self.net(x).squeeze(1)


class Net03(nn.Module):
    '''A simple fully-connected deep neural network'''

    def __init__(self, input_dim=1):
        super(Net03, self).__init__()

        # Define your neural network here
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        '''Given input of size (batch_size x input_dim), compute output of the network'''
        return self.net(x).squeeze(1)


class Net04(nn.Module):
    '''A simple fully-connected deep neural network'''

    def __init__(self, input_dim=1):
        super(Net04, self).__init__()

        # Define your neural network here
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        '''Given input of size (batch_size x input_dim), compute output of the network'''
        return self.net(x).squeeze(1)


class Net05(nn.Module):
    '''A simple fully-connected deep neural network'''

    def __init__(self, input_dim=1):
        super(Net05, self).__init__()

        # Define your neural network here
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        '''Given input of size (batch_size x input_dim), compute output of the network'''
        return self.net(x).squeeze(1)


class Net06(nn.Module):
    '''A simple fully-connected deep neural network'''

    def __init__(self, input_dim=1):
        super(Net06, self).__init__()

        # Define your neural network here
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        '''Given input of size (batch_size x input_dim), compute output of the network'''
        return self.net(x).squeeze(1)
        # return self.net(x)


class Net061(nn.Module):
    '''A simple fully-connected deep neural network'''

    def __init__(self, input_dim=1):
        super(Net061, self).__init__()

        # Define your neural network here
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        '''Given input of size (batch_size x input_dim), compute output of the network'''
        return self.net(x).squeeze(1)
        # return self.net(x)


class Net062(nn.Module):
    '''A simple fully-connected deep neural network'''

    def __init__(self, input_dim=1):
        super(Net062, self).__init__()

        # Define your neural network here
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        '''Given input of size (batch_size x input_dim), compute output of the network'''
        return self.net(x).squeeze(1)
        # return self.net(x)


class Net063(nn.Module):
    '''A simple fully-connected deep neural network'''

    def __init__(self, input_dim=1):
        super(Net063, self).__init__()

        # Define your neural network here
        self.net = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        '''Given input of size (batch_size x input_dim), compute output of the network'''
        return self.net(x).squeeze(1)
        # return self.net(x)


class Net064(nn.Module):
    '''A simple fully-connected deep neural network'''

    def __init__(self, input_dim=1):
        super(Net064, self).__init__()

        # Define your neural network here
        self.net = nn.Sequential(
            nn.Linear(input_dim, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
        )

    def forward(self, x):
        '''Given input of size (batch_size x input_dim), compute output of the network'''
        return self.net(x).squeeze(1)
        # return self.net(x)


class Net065(nn.Module):
    '''A simple fully-connected deep neural network'''

    def __init__(self, input_dim=1):
        super(Net065, self).__init__()

        # Define your neural network here
        self.net = nn.Sequential(
            nn.Linear(input_dim, 2),
            nn.ReLU(),
            nn.Linear(2, 1),
        )

    def forward(self, x):
        '''Given input of size (batch_size x input_dim), compute output of the network'''
        return self.net(x).squeeze(1)
        # return self.net(x)


class Net0651(nn.Module):
    '''A simple fully-connected deep neural network'''

    def __init__(self, input_dim=1):
        super(Net0651, self).__init__()

        # Define your neural network here
        self.net = nn.Sequential(
            nn.Linear(input_dim, 2),
            nn.ReLU(),
            nn.Linear(2, 2),
            nn.ReLU(),
            nn.Linear(2, 1),
        )

    def forward(self, x):
        '''Given input of size (batch_size x input_dim), compute output of the network'''
        return self.net(x).squeeze(1)
        # return self.net(x)


class Net066(nn.Module):
    '''A simple fully-connected deep neural network'''

    def __init__(self, input_dim=1):
        super(Net066, self).__init__()

        # Define your neural network here
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.ReLU(),
            nn.Linear(1, 1),
        )

    def forward(self, x):
        '''Given input of size (batch_size x input_dim), compute output of the network'''
        return self.net(x).squeeze(1)
        # return self.net(x)


class Net067(nn.Module):
    '''A simple fully-connected deep neural network'''

    def __init__(self, input_dim=1):
        super(Net067, self).__init__()

        # Define your neural network here
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1),
        )

    def forward(self, x):
        '''Given input of size (batch_size x input_dim), compute output of the network'''
        return self.net(x).squeeze(1)
        # return self.net(x)


class Net07(nn.Module):
    '''A simple fully-connected deep neural network'''

    def __init__(self, input_dim=1):
        super(Net07, self).__init__()

        # Define your neural network here
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        '''Given input of size (batch_size x input_dim), compute output of the network'''
        return self.net(x).squeeze(1)
        # return self.net(x)


class Net08(nn.Module):
    '''A simple fully-connected deep neural network'''

    def __init__(self, input_dim=1):
        super(Net08, self).__init__()

        # Define your neural network here
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        '''Given input of size (batch_size x input_dim), compute output of the network'''
        return self.net(x).squeeze(1)
        # return self.net(x)
