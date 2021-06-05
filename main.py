import csv
import threading
from torch.utils.data.dataloader import DataLoader

from src.lib.word_operator import str_format
from src.data_process import get_dataloader
from src.train_process import DL_Model


class TaskThread(threading.Thread):
    def __init__(
        self,
        model: DL_Model,
        mode: str = 'train',
        loader: DataLoader or None = None,
        val_loader: DataLoader or None = None,
    ):
        threading.Thread.__init__(self)
        self.model = model
        self.mode = mode
        self.loader = loader
        self.val_loader = val_loader
        self.result = None

    def run(self):
        if self.mode == 'train':
            self.result = self.model.training(self.loader, self.val_loader)
        elif self.mode == 'test':
            self.result = self.model.testing(self.loader)

    def execute(self):
        self.start()
        try:
            self.join()
        except KeyboardInterrupt:
            self.model.earlyStop = self.model.epoch + 1


def train():
    model = DL_Model()
    task = TaskThread(
        model=model,
        loader=get_dataloader('./Data/covid.train.csv', mode='train', batch_size=model.BATCH_SIZE, n_jobs=1),
        val_loader=get_dataloader('./Data/covid.train.csv', mode='valid', batch_size=model.BATCH_SIZE, n_jobs=1),
    )
    task.execute()


def pre_train():
    model = DL_Model()
    model.load_model('out/0605-1243_Net063_MSELoss_lr-1.0e-04_BS-512/final_e4000_2.129e-03.pickle')
    task = TaskThread(
        model=model,
        loader=get_dataloader('./Data/covid.train.csv', mode='train', batch_size=model.BATCH_SIZE, n_jobs=1),
        val_loader=get_dataloader('./Data/covid.train.csv', mode='valid', batch_size=model.BATCH_SIZE, n_jobs=1),
    )

    task.execute()


def full_train():
    model = DL_Model()
    model.saveDir = './out/full/'
    task = TaskThread(
        model=model,
        loader=get_dataloader('./Data/covid.train.csv', mode='full', batch_size=model.BATCH_SIZE, n_jobs=1),
    )

    task.execute()


def full_pre_train():
    model = DL_Model()
    model.load_model('out/full/0605-1440_Net065_MSELoss_lr-1.0e-04_BS-512/e2400_0.000e+00.pickle')
    task = TaskThread(
        model=model,
        loader=get_dataloader('./Data/covid.train.csv', mode='full', batch_size=model.BATCH_SIZE, n_jobs=1),
    )

    task.execute()


def test(model_path='out/0605-1236_Net062_MSELoss_lr-1.0e-04_BS-512/best-loss_e3785_2.112e-03.pickle'):
    model = DL_Model()
    model.load_model(model_path)
    task = TaskThread(
        model=model,
        mode='test',
        loader=get_dataloader('./Data/covid.test.csv', mode='test', batch_size=model.BATCH_SIZE, n_jobs=1),
    )
    task.execute()

    with open(f'{model_path[: model_path.rfind(".pickle")]}.csv', 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(task.result):
            writer.writerow([i, p])

    print(str_format("done!!", fore='g'))


if __name__ == '__main__':
    train()
    # pre_train()
    # full_train()
    # full_pre_train()
    # test()

    # test_paths = [
    #     'out/full/0605-1440_Net065_MSELoss_lr-1.0e-04_BS-512/0605-1750_Net065_MSELoss_lr-1.0e-04_BS-512/e2550_1.829e-03.pickle',
    #     'out/full/0605-1440_Net065_MSELoss_lr-1.0e-04_BS-512/0605-1750_Net065_MSELoss_lr-1.0e-04_BS-512/e2600_1.861e-03.pickle',
    #     'out/full/0605-1440_Net065_MSELoss_lr-1.0e-04_BS-512/0605-1750_Net065_MSELoss_lr-1.0e-04_BS-512/e2650_1.867e-03.pickle',
    #     'out/full/0605-1440_Net065_MSELoss_lr-1.0e-04_BS-512/0605-1750_Net065_MSELoss_lr-1.0e-04_BS-512/e2700_1.864e-03.pickle',
    #     'out/full/0605-1440_Net065_MSELoss_lr-1.0e-04_BS-512/0605-1750_Net065_MSELoss_lr-1.0e-04_BS-512/e2750_1.870e-03.pickle',
    #     'out/full/0605-1440_Net065_MSELoss_lr-1.0e-04_BS-512/0605-1750_Net065_MSELoss_lr-1.0e-04_BS-512/e2800_1.818e-03.pickle',
    #     'out/full/0605-1440_Net065_MSELoss_lr-1.0e-04_BS-512/0605-1750_Net065_MSELoss_lr-1.0e-04_BS-512/e2850_1.821e-03.pickle',
    #     'out/full/0605-1440_Net065_MSELoss_lr-1.0e-04_BS-512/0605-1750_Net065_MSELoss_lr-1.0e-04_BS-512/e2900_1.877e-03.pickle',
    #     'out/full/0605-1440_Net065_MSELoss_lr-1.0e-04_BS-512/0605-1750_Net065_MSELoss_lr-1.0e-04_BS-512/e2950_1.845e-03.pickle',
    # ]
    # for path in test_paths:
    #     test(path)
