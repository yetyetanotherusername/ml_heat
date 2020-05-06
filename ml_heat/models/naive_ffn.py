import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ml_heat.preprocessing.transform_data import DataTransformer

from ml_heat.helper import (
    load_animal,
    duplicate_shift
)

dt = DataTransformer()


class SXNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2 * 288 + 1, 1000)
        self.layer2 = nn.Linear(1000, 1000)
        self.layer3 = nn.Linear(1000, 1000)
        self.layer4 = nn.Linear(1000, 1)

    def forward(self, X):
        X = F.relu(self.layer1(X))
        X = F.relu(self.layer2(X))
        X = F.relu(self.layer3(X))
        y = self.layer4(X)

        return y


class Data(Dataset):
    def __init__(self, animal_ids, path, win_len):
        self.animal_ids = animal_ids
        self.store = path
        self.scaler = StandardScaler()
        self.time_window_len = win_len

    def __len__(self):
        return len(self.animal_ids)

    def __getitem__(self, index):
        animal_id = self.animal_ids[index]
        x, y = self.prepare_animal(animal_id)

        return torch.FloatTensor(x), torch.FloatTensor(y)

    def prepare_animal(self, animal_id):
        data = load_animal(self.store, animal_id)

        data['annotation'] = np.logical_or(
            np.logical_or(data.pregnant, data.cyclic), data.inseminated)

        data = data.drop([
            'race',
            'country',
            'temp',
            'temp_filtered',
            'pregnant',
            'cyclic',
            'inseminated',
            'deleted'
        ], axis=1)

        data.act = pd.to_numeric(data.act.round(decimals=2), downcast='float')
        data.act_group_mean = pd.to_numeric(
            data.act_group_mean.round(decimals=2), downcast='float')

        data[['act', 'act_group_mean']] = self.scaler.fit_transform(
            data[['act', 'act_group_mean']])

        shifts = range(self.time_window_len)

        act_shift = duplicate_shift(data.act, shifts, 'act')
        group_shift = duplicate_shift(
            data.act_group_mean, shifts, 'act_group_mean')

        data = pd.concat([data, act_shift, group_shift], axis=1)
        data = data.drop(['act', 'act_group_mean'], axis=1)
        data = data.dropna()
        data.DIM = pd.to_numeric(data.DIM, downcast='signed')
        data.annotation = data.annotation.astype(int)

        y = data.annotation.values
        x = data.drop('annotation', axis=1).values

        return x, y


class NaiveFNN(object):
    def __init__(self, animal_ids=None):
        self.animal_ids = animal_ids
        self.store = dt.feather_store
        self.model_store = os.path.join(
            os.getcwd(), 'ml_heat', '__data_store__', 'models', 'naive_ffn')
        if self.animal_ids is None:
            self.animal_ids = os.listdir(self.store)[:10000]

        self.epochs = 2
        self.learning_rate = 0.01
        self.momentum = 0.9
        self.time_window_len = 288

        train_animals, test_animals = train_test_split(
            self.animal_ids, test_size=0.33, random_state=42)

        self.partition = {'train': train_animals, 'validation': test_animals}

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
            self.store,
            self.time_window_len
        )

        trainloader = DataLoader(
            dataset=traindata,
            batch_size=1,
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
                y = y.T.unsqueeze(0).to(device)
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
            self.store,
            self.time_window_len
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
                x = x.to(device)
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

    def crawler(self):
        for animal_id in tqdm(self.animal_ids):
            frame = load_animal(self.store, animal_id)

            if frame.shape[0] < self.time_window_len:
                os.remove(os.path.join(self.store, animal_id))

    def run(self):
        self.train()
        self.validate()


def main():
    obj = NaiveFNN()
    obj.run()


if __name__ == '__main__':
    main()
