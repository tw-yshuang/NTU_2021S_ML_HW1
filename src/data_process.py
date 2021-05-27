import csv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class COVID19Dataset(Dataset):
    def __init__(self, path: str, mode='train', haveLabel=False) -> None:
        self.mode = mode

        with open(path, 'r') as f:
            data = list(csv.reader(f))
            data = np.array(data[1:])[:, 1:].astype(float)

        if haveLabel:
            feats = range(93)

        if mode == 'test':
            # Testing data
            # data: 893 x 93 (40 states + day 1 (18) + day 2 (18) + day 3 (17))
            data = data[:, feats]
            self.data = torch.FloatTensor(data)
        else:
            # Training data (train/valid sets)
            # data: 2700 x 94 (40 states + day 1 (18) + day 2 (18) + day 3 (18))
            label = data[:, -1]
            data = data[:, feats]

            # Splitting training data into train & valid sets
            if mode == 'train':
                indices = [i for i in range(data.shape[0]) if i % 10 != 0]
            elif mode == 'valid':
                indices = [i for i in range(data.shape[0]) if i % 10 == 0]

            # Convert data into PyTorch tensors
            self.data = torch.FloatTensor(data[indices])
            self.label = torch.FloatTensor(label[indices])

        # * Normalize features
        # self.data[:, 40:].mean(dim=0, keepdim=True) -> get average value in every column after no. of 40 column
        # self.data[:, 40:].std(dim=0, keepdim=True) -> get standard deviation in every column after no. of 40 column
        self.data[:, 40:] = (self.data[:, 40:] - self.data[:, 40:].mean(dim=0, keepdim=True)) / self.data[:, 40:].std(
            dim=0, keepdim=True
        )

        self.dim = self.data.shape[1]
        print(f'Finished reading the {mode} set of COVID19 Dataset ({len(self.data)} samples found, each dim = {self.dim})')

    def __getitem__(self, index):
        # Returns one sample at a time
        if self.mode is 'test':
            # For testing (no label)
            return self.data[index]
        else:
            # For training
            return self.data[index], self.label[index]

    def __len__(self):
        # Returns the size of the dataset
        return len(self.data)


def get_dataloader(path, mode, batch_size, n_jobs=1, haveLabel=False):
    '''Generates a dataset, then is put into a dataloader.
    mode: 'train', 'vaild', 'test' '''
    dataset = COVID19Dataset(path, mode=mode, haveLabel=haveLabel)  # Construct dataset
    dataloader = DataLoader(
        dataset,
        batch_size,
        shuffle=(mode == 'train'),
        drop_last=False,
        num_workers=n_jobs,
        pin_memory=False,
    )  # Construct dataloader
    return dataloader


if __name__ == '__main__':
    get_dataloader('./Data/covid.train.csv', mode='train', batch_size=128)
