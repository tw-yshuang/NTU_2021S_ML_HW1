from src.data_process import get_dataloader
from src.train_process import DL_Model


def main():
    model = DL_Model()
    train_loader = get_dataloader('./Data/covid.train.csv', mode='train', batch_size=model.BATCH_SIZE, haveLabel=True)
    valid_loader = get_dataloader('./Data/covid.train.csv', mode='valid', batch_size=model.BATCH_SIZE, haveLabel=True)

    model.training(train_loader, valid_loader)


if __name__ == '__main__':
    main()
