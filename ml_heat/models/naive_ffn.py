import os
import numpy as np
from tqdm import tqdm
import zarr
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

from ml_heat.preprocessing.transform_data import DataTransformer
from ml_heat.helper import plot_timings


dt = DataTransformer()


class SXNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(288 + 1, 1000)
        self.layer2 = nn.Linear(1000, 1000)
        self.layer3 = nn.Linear(1000, 1000)
        self.layer4 = nn.Linear(1000, 1)

        self.bn1 = nn.BatchNorm1d(1000)
        self.bn2 = nn.BatchNorm1d(1000)
        self.bn3 = nn.BatchNorm1d(1000)

    def forward(self, X):
        X = F.selu(self.bn1(self.layer1(X)))
        X = F.selu(self.bn2(self.layer2(X)))
        X = F.selu(self.bn3(self.layer3(X)))
        y = self.layer4(X)

        return y


class Data(IterableDataset):
    def __init__(self, path):
        super(Data, self).__init__()

        store = zarr.NestedDirectoryStore(path)
        self.array = zarr.open(store, mode='r')

    def __iter__(self):
        return iter(self.array)


class NaiveFNN(object):
    def __init__(self):
        self.store = dt.zarr_store
        self.model_store = os.path.join(
            os.getcwd(), 'ml_heat', '__data_store__', 'models', 'naive_ffn')

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
        # torch.backends.cudnn.benchmark = True

        sxnet = SXNet()
        sxnet.to(device)
        optimizer = optim.Adam(
            sxnet.parameters(),
            lr=self.learning_rate
        )

        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.FloatTensor([250.]).to(device))

        traindata = Data(self.store)
        trainloader = DataLoader(
            dataset=traindata,
            batch_size=50000,
            shuffle=False,
            num_workers=1
        )

        params = {
            'desc': 'epoch progress',
            'smoothing': 0.01,
            'total': traindata.array.shape[0] // trainloader.batch_size
        }

        for e in range(self.epochs):  # loop over the dataset multiple times
            epoch_loss = 0
            epoch_acc = 0
            epoch_len = 0
            for batch in tqdm(trainloader, **params):
                x = batch[:, 1:].to(device)
                y = batch[:, 0].unsqueeze(0).T.to(device)

                optimizer.zero_grad()

                # forward + backward + optimize
                y_pred = sxnet(x)
                loss = criterion(y_pred, y)
                acc = self.binary_acc(y_pred, y)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_acc += acc.item()
                epoch_len += y.shape[1]

            print(f'Epoch {e+0:03}: | '
                  f'Loss: {epoch_loss/epoch_len:.5f} | '
                  f'Acc: {epoch_acc/epoch_len:.3f}')

            torch.save(sxnet.state_dict(), self.model_store)

        print('Finished Training')

    def validate(self):
        print('Starting Validation')
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")

        sxnet = SXNet()
        sxnet.to(device)
        # sxnet.load_state_dict(torch.load(self.model_store))
        torch.multiprocessing.set_sharing_strategy('file_system')

        testdata = Data(self.store)
        testloader = DataLoader(
            testdata,
            batch_size=50000,
            shuffle=False,
            num_workers=1
        )

        y_pred_list = []
        y_list = []
        sxnet.eval()

        params = {
            'desc': 'epoch progress',
            'smoothing': 0.01,
            'total': testdata.array.shape[0] // testloader.batch_size
        }

        with torch.no_grad():
            for batch in tqdm(testloader, **params):
                x = batch[:, 1:].to(device)
                y = batch[:, 0].unsqueeze(0).T
                y_test_pred = sxnet(x)
                y_test_pred = torch.sigmoid(y_test_pred)
                y_pred_tag = torch.round(y_test_pred)
                y_pred_list.append(y_pred_tag.cpu().numpy())
                y_list.append(y.numpy())
        y_pred_list = np.concatenate([a.squeeze() for a in y_pred_list])
        y_list = np.concatenate([a.squeeze() for a in y_list])

        print('\n#####################################')
        print('Confusion Matrix')
        print('#####################################\n')
        print(confusion_matrix(y_list, y_pred_list))
        print('\n')

        print('#####################################')
        print('Classification Report')
        print('#####################################\n')
        print(classification_report(
            y_list, y_pred_list,
            target_names=['no heat', 'heat'],
            digits=6,
            zero_division=0
        ))

    def test_dataloader(self):
        traindata = Data(
            self.store
        )

        trainloader = DataLoader(
            dataset=traindata,
            batch_size=50000,
            shuffle=False,
            num_workers=4
        )

        for row in trainloader:
            x = row[:, 1:]
            y = row[:, 0]

            print(x)
            print(y)

            break

    def run(self):
        self.train()
        self.validate()
        # self.test_dataloader()


def main():
    obj = NaiveFNN()
    obj.run()


if __name__ == '__main__':
    main()
