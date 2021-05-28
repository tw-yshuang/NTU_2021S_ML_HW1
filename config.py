import numpy as np
import torch
import torch.nn as nn
import src.net as net


def get_device():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('Device State:', device)

    return device


class DL_Config(object):
    def __init__(self) -> None:
        self.basic_config()
        self.net_config()
        self.performance_config()
        self.save_config()

    def basic_config(self):
        self.SEED: int = 24
        self.NUM_EPOCH: int = 2800
        self.BATCH_SIZE: int = 512
        self.earlyStop: int or None = None

        np.random.seed(self.SEED)
        torch.manual_seed(self.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.SEED)

    def net_config(self):
        self.isClassified = False
        self.net = net.Net04(93).to(get_device())
        self.loss_func = nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=1e-4, momentum=0.9)
        self.min_MES = 1000.0

    def performance_config(self):
        self.printPerformance: bool = True
        self.showPlot: bool = True
        self.savePerformance: bool = True
        self.savePlot: bool = True

    def save_config(self):
        self.saveDir = './out/test/'
        self.saveModel = True
        self.checkpoint = 0
        self.bestModelSave = True
        self.onlyParameters = True
