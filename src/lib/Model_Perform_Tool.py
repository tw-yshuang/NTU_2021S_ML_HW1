from typing import List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from torch._C import set_flush_denormal

matplotlib.use('AGG')


class Model_Perform_Tool(object):
    def __init__(
        self,
        train_loss_ls: List[float] or None = None,
        test_loss_ls: List[float] or None = None,
        train_acc_ls: List[float] or None = None,
        test_acc_ls: List[float] or None = None,
        saveDir: str = './out',
    ) -> None:
        self.num_epoch = len(train_loss_ls)
        self.train_loss_ls = train_loss_ls
        self.test_loss_ls = test_loss_ls
        self.train_acc_ls = train_acc_ls
        self.test_acc_ls = test_acc_ls
        self.saveDir = saveDir

    def save_history_csv(self):
        import pandas as pd

        # <<<set the name that won't let program auto cover it~~>>>
        name_mark = str(self.test_acc_ls[-1])[2:5]

        df = pd.DataFrame(
            {
                'EPOCH': range(1, self.num_epoch + 1),
                'train_acc': self.train_acc_ls,
                'train_loss': self.train_loss_ls,
                'test_acc': self.test_acc_ls,
            }
        )
        df.to_csv(f'{self.saveDir}/{name_mark}_history.csv')

    def draw_plot(self, startNumEpoch: int = 10):
        name_mark = str(self.train_loss_ls[-1])[:5]
        epochs = range(startNumEpoch + 1, self.num_epoch + 1)
        plt.clf()

        for key, values_ls in {'Loss': [self.train_loss_ls, self.test_loss_ls], 'Acc': [self.train_acc_ls, self.test_acc_ls]}.items():
            for idx, values in enumerate(values_ls):
                if values is not None:
                    values = values[startNumEpoch:]
                    best_value = min(values) if key == 'Loss' else max(values)
                    plt.plot(epochs, values)
                    # 设置数字标签
                    i = startNumEpoch
                    for epoch, value in zip(epochs, values):
                        i += 1
                        if i % (self.num_epoch // 5) == 0 or i == self.num_epoch or best_value == value:
                            value = np.round(value, 4)
                            plt.text(epoch, value, value, ha='center', va=('bottom' if idx == 0 else 'top'), fontsize=10)

            if [True for values in values_ls if values is not None]:
                # 設定圖片標題，以及指定字型設定，x代表與圖案最左側的距離，y代表與圖片的距離
                plt.title(key, x=0.5, y=1.03)
                # 设置刻度字体大小
                plt.xticks(fontsize=10)
                plt.yticks(fontsize=10)
                # 標示x軸(labelpad代表與圖片的距離)
                plt.xlabel('Epoch', fontsize=10)
                # 標示y軸(labelpad代表與圖片的距離)
                plt.ylabel(key, fontsize=10)
                plt.savefig(f'{self.saveDir}/{name_mark}{key}.png')
                plt.clf()
