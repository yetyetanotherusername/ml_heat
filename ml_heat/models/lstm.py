import os
import zarr
from tqdm import tqdm
import numpy as np

from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Dataset

from ml_heat.preprocessing.transform_data import DataTransformer
from ml_heat.helper import plot_setup


dt = DataTransformer()


class SXlstm(nn.Module):
    def __init__(self, input_size=4, h_size=100, layers=1, out_size=1):
        super().__init__()

        self.h_size = h_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=h_size,
            num_layers=layers,
            batch_first=True
        )

        self.linear = nn.Linear(h_size, out_size)
        self.state = (torch.zeros(1, 1, h_size), torch.zeros(1, 1, h_size))

    def forward(self, X):
        Y, self.state = self.lstm(X, self.state)
        return self.linear(Y)


class Data(Dataset):
    def __init__(self, path, set_name):
        super(Data, self).__init__()
        self.store = zarr.DirectoryStore(path)
        self.group = zarr.hierarchy.group(store=self.store)

        self.keys = self.group[set_name][:]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        return self.group[self.keys[idx]][:, :]


class LSTM(object):
    def __init__(self):
        self.store = dt.zarr_store
        self.model_store = os.path.join(
            os.getcwd(), 'ml_heat', '__data_store__', 'models', 'lstm')
        self.model_path = os.path.join(self.model_store, 'lstm')

        self.epochs = 2
        self.learning_rate = 0.01
        self.momentum = 0.9

    def binary_acc(self, y_pred, y_test):
        y_pred_tag = torch.round(torch.sigmoid(y_pred))

        correct_results_sum = (y_pred_tag == y_test).sum().float()
        acc = correct_results_sum / y_test.shape[0]
        acc = torch.round(acc * 100)

        return acc

    def train(self):
        use_cuda = torch.cuda.is_available()

        device = torch.device("cuda:0" if use_cuda else "cpu")

        if use_cuda:
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True

        sxnet = SXlstm()
        sxnet.to(device)
        try:
            sxnet.load_state_dict(
                torch.load(self.model_path, map_location=device))
        except Exception:
            pass

        optimizer = optim.Adam(
            sxnet.parameters(),
            lr=self.learning_rate
        )

        criterion = nn.BCEWithLogitsLoss()
        # pos_weight=torch.FloatTensor([250.]).to(device))

        traindata = Data(self.store, 'train_keys')
        trainloader = DataLoader(
            dataset=traindata,
            batch_size=1,
            shuffle=True,
            num_workers=os.cpu_count(),
            pin_memory=True
        )

        params = {
            'desc': 'epoch progress',
            'smoothing': 0.01,
            'total': len(traindata)
        }

        losses = []
        accuracies = []

        for e in range(self.epochs):  # loop over the dataset multiple times
            epoch_loss = 0
            epoch_acc = 0
            epoch_len = 0
            for batch in tqdm(trainloader, **params):
                x = batch[:, :, 1:].to(device)
                y = batch[:, :, 0].to(device)

                optimizer.zero_grad()

                sxnet.state = (
                    torch.zeros(1, 1, sxnet.h_size).to(device),
                    torch.zeros(1, 1, sxnet.h_size).to(device))

                # forward + backward + optimize
                y_pred = sxnet(x).reshape(y.shape)
                loss = criterion(y_pred, y)
                acc = self.binary_acc(y_pred, y)

                loss.backward(retain_graph=True)
                optimizer.step()

                epoch_loss += loss.item()
                epoch_acc += acc.item()
                epoch_len += y.shape[1]

            losses.append(epoch_loss / epoch_len)
            accuracies.append(epoch_acc / epoch_len)

            print(f'Epoch {e+0:03}: | '
                  f'Loss: {epoch_loss/epoch_len:.5f} | '
                  f'Acc: {epoch_acc/epoch_len:.3f}')

            torch.save(sxnet.state_dict(), self.model_path)

        plt = plot_setup()

        plt.plot(range(self.epochs), losses)
        plt.savefig(os.path.join(self.model_store, 'train_loss.pdf'))
        plt.figure()
        plt.plot(range(self.epochs), accuracies)
        plt.savefig(os.path.join(self.model_store, 'train_accuracy.pdf'))

        print('Finished Training')

    def validate(self):
        print('Starting Validation')
        use_cuda = torch.cuda.is_available()
        device = torch.device('cuda:0' if use_cuda else 'cpu')

        if use_cuda:
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True

        sxnet = SXlstm()
        sxnet.to(device)
        try:
            sxnet.load_state_dict(torch.load(
                self.model_path, map_location=device))
        except FileNotFoundError:
            print('Warning: no trained model found')
        torch.multiprocessing.set_sharing_strategy('file_system')

        testdata = Data(self.store, 'test_keys')
        testloader = DataLoader(
            testdata,
            batch_size=1,
            shuffle=False,
            num_workers=os.cpu_count(),
            pin_memory=True
        )

        y_list = []
        y_label_list = []
        sxnet.eval()

        params = {
            'desc': 'validation progress',
            'smoothing': 0.01,
            'total': len(testdata)
        }

        with torch.no_grad():
            for batch in tqdm(testloader, **params):
                x = batch[:, :, 1:].to(device)
                y_label = batch[:, :, 0]
                sxnet.state = (
                    torch.zeros(1, 1, sxnet.h_size).to(device),
                    torch.zeros(1, 1, sxnet.h_size).to(device))
                y = torch.sigmoid(sxnet(x)).reshape(y_label.shape)
                y = torch.round(y)
                y_list.extend(y.cpu().flatten().tolist())
                y_label_list.extend(y_label.flatten().tolist())

        print('\n#####################################')
        print('Confusion Matrix')
        print('#####################################\n')
        print(confusion_matrix(y_label_list, y_list))
        print('\n')

        print('#####################################')
        print('Classification Report')
        print('#####################################\n')
        print(classification_report(
            y_label_list, y_list,
            target_names=['no heat', 'heat'],
            digits=6,
            zero_division=0
        ))

    def test_dataloader(self):
        traindata = Data(
            os.path.join(self.store, 'origin')
        )

        trainloader = DataLoader(
            dataset=traindata,
            batch_size=1,
            shuffle=False,
            num_workers=4,
        )

        data = iter(trainloader).next()

        writer.add_graph(SXNet(), data[:, 1:])
        writer.close()

        for idx, row in enumerate(trainloader):
            x = row[:, 1:]
            y = row[:, 0]

            print(x)
            print(y)

            break

        # plot_timings(trainloader, model_time=0.2, n_batches=4)

    def run(self):
        self.train()
        self.validate()
        # self.test_dataloader()


def main():
    obj = LSTM()
    obj.run()


if __name__ == '__main__':
    main()
