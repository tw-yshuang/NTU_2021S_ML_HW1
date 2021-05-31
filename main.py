import threading
from torch.utils.data.dataloader import DataLoader
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


def pre_training():
    model = DL_Model()
    model.load_model('./out/0528-1830_Net04_MSELoss_lr-1.0e-04_BS-512/best-loss_e5567_2.579e-03.pickle')
    task = TaskThread(
        model=model,
        loader=get_dataloader('./Data/covid.train.csv', mode='train', batch_size=model.BATCH_SIZE, n_jobs=1, haveLabel=True),
        val_loader=get_dataloader('./Data/covid.train.csv', mode='valid', batch_size=model.BATCH_SIZE, n_jobs=1, haveLabel=True),
    )

    task.start()
    try:
        task.join()
    except KeyboardInterrupt:
        model.earlyStop = model.epoch + 1


def main():
    model = DL_Model()
    task = TaskThread(
        model=model,
        loader=get_dataloader('./Data/covid.train.csv', mode='train', batch_size=model.BATCH_SIZE, n_jobs=1, haveLabel=True),
        val_loader=get_dataloader('./Data/covid.train.csv', mode='valid', batch_size=model.BATCH_SIZE, n_jobs=1, haveLabel=True),
    )
    task.start()
    try:
        task.join()
    except KeyboardInterrupt:
        model.earlyStop = model.epoch + 1


if __name__ == '__main__':
    main()
    # pre_training()
    # test()

    # asyncio.run_coroutine_threadsafe(test())
