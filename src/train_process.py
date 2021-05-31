import sys, os, glob, time
from typing import List
import numpy as np
import torch
from torch.autograd.grad_mode import no_grad
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(__package__))
from config import DL_Config
from src.lib.Model_Perform_Tool import Model_Perform_Tool
from src.lib.word_operator import str_format


class DL_Performance(object):
    def __init__(self) -> None:
        self.train_loss_ls: List[float] = []
        self.val_loss_ls: List[float] = []
        self.train_acc_ls: List[int] = []
        self.val_acc_ls: List[int] = []

        self.best_acc_epoch: int = 0
        self.best_loss_epoch: int = 0

    def print(self, epoch: int, total_epoch: int, time_start: time, end: str = '\n'):
        if len(self.train_loss_ls) <= epoch:
            raise ProcessLookupError(str_format("Wrong no. of epoch !!", style='blink', fore='r'))

        val_word = " | "
        if len(self.val_loss_ls) > epoch:  # avoid error that doesn't have validation in pre-train process.
            if len(self.val_acc_ls) > epoch:
                val_acc = (
                    str_format(f'{self.val_acc_ls[epoch]*100:.2f}', fore='y')
                    if self.val_acc_ls[self.best_acc_epoch] == self.val_acc_ls[epoch]
                    else f'{self.val_acc_ls[epoch]*100:.2f}'
                )
                val_word += f"Val acc: {val_acc}, "

            val_loss = (
                str_format(f'{self.val_loss_ls[epoch]:.5e}', fore='y')
                if self.val_loss_ls[self.best_loss_epoch] == self.val_loss_ls[epoch]
                else f'{self.val_loss_ls[epoch]:.5e}'
            )
            val_word += f"loss: {val_loss}"

        train_word = f"Train acc: {self.train_acc_ls[epoch]*100:.2f}, " if len(self.train_acc_ls) > epoch else "Train "
        train_word += f"loss: {self.train_loss_ls[epoch]:.5e}"

        print(f"[{epoch+1:>2d}/{total_epoch}] {time.time() - time_start:.3f}sec, {train_word}{val_word}", end=end)

    def visualize_info(self, generatePlot=True, generateCSV=False, saveDir: str or None = './out'):
        self.visualize = Model_Perform_Tool(
            self.train_loss_ls,
            self.val_loss_ls,
            self.train_acc_ls if self.train_acc_ls != [] else None,
            self.val_acc_ls if self.val_acc_ls != [] else None,
            saveDir,
        )
        if generatePlot:
            self.visualize.draw_plot(startNumEpoch=len(self.train_loss_ls) // 5)
            print(str_format("Compelete generate plot !!", fore='g'))
        if generateCSV:
            self.visualize.save_history_csv()
            print(str_format("Compelete generate csv !!", fore='g'))


class DL_Model(DL_Config):
    def __init__(self, device: str = 'cuda:0') -> None:
        try:  # form pre-train model
            self.performance = self.performance
        except AttributeError:  # new model
            super().__init__()
            self.performance = DL_Performance()

        self.device = device

        if self.saveModel:
            # generate directory by '{date}-{time}_{model}_{loss-function}_{optimizer}_{lr}_{batch-size}'
            self.saveDir = f'{self.saveDir}{time.strftime("%m%d-%H%M")}_{self.net.__class__.__name__}_{self.loss_func.__class__.__name__}_lr-{self.optimizer.defaults["lr"]:.1e}_BS-{self.BATCH_SIZE}'

        self.epoch_start = len(self.performance.train_loss_ls)
        self.TOTAL_EPOCH = self.epoch_start + self.NUM_EPOCH

        self.train_acc = 0.0
        self.train_loss = 0.0
        self.val_acc = 0.0
        self.val_loss = 0.0

    def training(self, loader: DataLoader, val_loader: DataLoader or None = None, saveModel: bool = False):
        # training
        for self.epoch in range(self.epoch_start, self.TOTAL_EPOCH):
            start_time = time.time()
            num_right = 0
            sum_loss = 0.0

            self.net.train()
            for data, label in loader:
                data = data.to(self.device)
                label = label.cpu()
                self.optimizer.zero_grad()
                pred = self.net(data).cpu()
                loss = self.loss_func(pred, label)
                loss.backward()
                self.optimizer.step()

                # calculate acc & loss
                sum_loss += loss.item()

                if self.isClassified:
                    pred_result = torch.argmax(pred).numpy()
                    num_right += sum(pred_result == label.numpy())

            self.train_loss = sum_loss / len(loader.dataset)
            self.performance.train_loss_ls.append(self.train_loss)

            if self.isClassified:
                self.train_acc = num_right / len(loader.dataset)  # loader.sampler.num_samples
                self.performance.train_acc_ls.append(self.train_acc)

            if self.printPerformance:
                if val_loader is not None:
                    self.performance.print(self.epoch, self.TOTAL_EPOCH, start_time, end='\r')
                    self.valiating(val_loader)

                self.performance.print(self.epoch, self.TOTAL_EPOCH, start_time)

            # save model
            if self.saveModel or saveModel:
                self.saveModel = True
                self.save_process()

            # early stop
            if self.earlyStop is not None:
                # self.epoch_start = self.epoch
                self.TOTAL_EPOCH = self.earlyStop
                # TODO: there has something better earlyStop method!! need to find out, go go:!!
                # self.earlyStop = None
                # self.training(loader, val_loader)
                self.save_process()
                break

        self.performance.visualize_info(generatePlot=self.savePlot, saveDir=self.saveDir)
        return True

    def valiating(self, loader: DataLoader):
        num_right = 0
        sum_loss = 0.0

        self.net.eval()  # change model to the evaluation(val or test) mode.
        with no_grad():
            for data, label in loader:
                data = data.to(self.device)
                label = label.cpu()

                # valiating process
                pred = self.net(data).cpu()
                loss = self.loss_func(pred, label)
                self.optimizer.step()

                # calculate loss
                sum_loss += loss.item()

                if self.isClassified:
                    pred_label = torch.argmax(pred, dim=1).numpy()
                    num_right += sum(pred_label == label.numpy())

            # valiation info. record
            # loader.sampler.num_samples
            self.val_loss = sum_loss / len(loader.dataset)
            self.performance.val_loss_ls.append(self.val_loss)
            if self.performance.val_loss_ls[self.performance.best_loss_epoch] > self.val_loss:
                self.performance.best_loss_epoch = self.epoch

            if self.isClassified:
                self.val_acc = num_right / len(loader.dataset)
                self.performance.val_acc_ls.append(self.val_acc)
                if self.performance.val_acc_ls[self.performance.best_acc_epoch] < self.val_acc:
                    self.performance.best_acc_epoch = self.epoch

    def testing(self, loader: DataLoader):
        self.net.eval()
        result_ls = np.array([])
        with no_grad():
            for dataset in loader:
                data = dataset.to(self.device)

                pred = self.net(data).cpu()
                results = torch.argmax(pred, dim=1).numpy()

                result_ls = np.concatenate((result_ls, results), axis=0).astype('int8')

            return result_ls

    def save_process(self):
        if self.saveDir is None:
            raise ProcessLookupError(str_format("Need to type path in saveDir from class: DL_Setting", fore='r'))

        try:
            if not os.path.exists(self.saveDir):
                os.mkdir(self.saveDir)
                print(str_format(f"Successfully created the directory: {self.saveDir}", fore='g'))
        except OSError:
            raise OSError(str_format(f"Fail to create the directory {self.saveDir} !", fore='r'))

        # final epoch
        if self.epoch + 1 == self.TOTAL_EPOCH:
            self.save_model(f'{self.saveDir}/final_e{self.epoch+1:03d}_{self.val_loss:.3e}.pickle')
        # checkpoint
        if self.checkpoint > 0 and self.epoch % self.checkpoint == 0:
            self.save_model(f'{self.saveDir}/e{self.epoch+1:03d}_{self.val_loss:.3e}.pickle')
        # best model
        if self.bestModelSave and self.epoch > 0:
            for key, best_epoch in {'acc': self.performance.best_acc_epoch, 'loss': self.performance.best_loss_epoch}.items():
                if self.epoch == best_epoch:
                    [os.remove(filename) for filename in glob.glob(f'{self.saveDir}/best-{key}*.pickle')]
                    self.save_model(f'{self.saveDir}/best-{key}_e{self.epoch+1:03d}_{self.val_loss:.3e}.pickle')

    def save_model(self, path):
        if self.onlyParameters:
            # torch.save(self.net.state_dict(), path)
            net = self.net
            optimizer = self.optimizer
            self.net = self.net.state_dict()
            self.optimizer = self.optimizer.state_dict()
            torch.save(self, path)
            self.net = net
            self.optimizer = optimizer
        else:
            torch.save(self, path)

    def load_model(self, path, fullNet=False):
        model: DL_Model = torch.load(path)
        self.performance = model.performance
        self.saveDir = path[: path.rfind('/') + 1]

        self.__init__()

        if fullNet:
            self.net = model.net
            self.optimizer = model.optimizer
        else:
            self.net.load_state_dict(model.net)
            self.optimizer.load_state_dict(model.optimizer)
        self.net.eval()


if __name__ == '__main__':
    aa = DL_Model()
    aa.training(None)
