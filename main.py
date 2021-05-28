import threading
from os import wait
from src.data_process import get_dataloader
from src.train_process import DL_Model


class TaskThread(threading.Thread):
    def __init__(self, model, loader, val_loader):
        threading.Thread.__init__(self)
        self.model = model
        self.loader = loader
        self.val_loader = val_loader

    def run(self):
        self.model.training(self.loader, self.val_loader)


def pre_training():
    model = DL_Model()
    model.load_model('./out/0528-1830_Net04_MSELoss_lr-1.0e-04_BS-512/best-loss_e5567_2.579e-03.pickle')
    train_loader = get_dataloader('./Data/covid.train.csv', mode='train', batch_size=model.BATCH_SIZE, haveLabel=True)
    valid_loader = get_dataloader('./Data/covid.train.csv', mode='valid', batch_size=model.BATCH_SIZE, haveLabel=True)

    model.training(train_loader, valid_loader)


def main():
    model = DL_Model()
    train_loader = get_dataloader('./Data/covid.train.csv', mode='train', batch_size=model.BATCH_SIZE, haveLabel=True)
    valid_loader = get_dataloader('./Data/covid.train.csv', mode='valid', batch_size=model.BATCH_SIZE, haveLabel=True)

    try:
        model.training(train_loader, valid_loader)
    except KeyboardInterrupt:
        model.TOTAL_EPOCH = model.epoch
        model.save_process()
        model.performance.visualize_info(generatePlot=model.savePlot, saveDir=model.saveDir)


# TODO: how to let last model can finish it's job and savemodel & savePlot after KeyboardInterrupt?
def test():

    model = DL_Model()
    train_loader = get_dataloader('./Data/covid.train.csv', mode='train', batch_size=model.BATCH_SIZE, haveLabel=True)
    valid_loader = get_dataloader('./Data/covid.train.csv', mode='valid', batch_size=model.BATCH_SIZE, haveLabel=True)

    task = TaskThread(model, train_loader, valid_loader)
    task.start()
    try:
        task.join()
    except KeyboardInterrupt:
        model.earlyStop = model.epoch + 1


if __name__ == '__main__':
    # main()
    # pre_training()
    test()

    # asyncio.run_coroutine_threadsafe(test())
