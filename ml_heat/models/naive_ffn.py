import os
import numpy as np
from tqdm import tqdm
import zarr
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ml_heat.preprocessing.transform_data import DataTransformer

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


class Data(Dataset):
    def __init__(self, mask, path):
        super(Data, self).__init__()

        store = zarr.LRUStoreCache(
            zarr.NestedDirectoryStore(path),
            5 * 10 ** 9
        )

        self.array = zarr.open(store, mode='r')
        self.mask = mask
        self.arrlen = np.sum(self.mask)

    def __len__(self):
        return self.arrlen

    def __getitem__(self, subindex):
        index = np.nonzero(self.mask)[0][subindex]
        row = self.array[index, :]
        y = [row[0]]
        x = row[1:]
        return torch.FloatTensor(x), torch.FloatTensor(y)


class NaiveFNN(object):
    def __init__(self):
        self.store = dt.zarr_store
        self.model_store = os.path.join(
            os.getcwd(), 'ml_heat', '__data_store__', 'models', 'naive_ffn')

        self.epochs = 2
        self.learning_rate = 0.01
        self.momentum = 0.9

        array = zarr.open(zarr.NestedDirectoryStore(self.store), mode='r')
        indices = range(array.shape[0])

        train_indices, test_indices = train_test_split(
            indices, test_size=0.33, random_state=42)

        train_mask = self.check_a_in_b(indices, sorted(train_indices))
        test_mask = self.check_a_in_b(indices, sorted(test_indices))

        self.partition = {'train': train_mask, 'validation': test_mask}

    def check_a_in_b(self, a, b):
        assert len(a) >= len(b)
        out = np.zeros(len(a), dtype=bool)
        out[np.in1d(a, b)] = True
        return out

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

        traindata = Data(
            self.partition['train'],
            self.store
        )

        trainloader = DataLoader(
            dataset=traindata,
            batch_size=50000,
            shuffle=True,
            num_workers=4
        )

        params = {
            'desc': 'epoch progress',
            'smoothing': 0.01
        }

        for e in range(self.epochs):  # loop over the dataset multiple times
            epoch_loss = 0
            epoch_acc = 0
            epoch_len = 0
            for x, y in tqdm(trainloader, **params):
                x = x.to(device)
                y = y.to(device)
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
        sxnet.load_state_dict(torch.load(self.model_store))
        torch.multiprocessing.set_sharing_strategy('file_system')

        testdata = Data(
            self.partition['validation'],
            self.store
        )

        testloader = DataLoader(
            testdata,
            batch_size=1,
            shuffle=False,
            num_workers=4
        )

        y_pred_list = []
        y_list = []
        sxnet.eval()
        with torch.no_grad():
            for x, y in tqdm(testloader, desc='validation'):
                x = x[-1, :, :].to(device)
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

    def run(self):
        self.train()
        self.validate()


def main():
    obj = NaiveFNN()
    obj.run()


if __name__ == '__main__':
    main()
