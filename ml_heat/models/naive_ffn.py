import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ml_heat.preprocessing.transform_data import DataTransformer

from ml_heat.helper import (
    load_organisation,
    duplicate_shift
)

dt = DataTransformer()


class SXNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2 * 144 + 1, 1000)
        self.layer2 = nn.Linear(1000, 1000)
        self.layer3 = nn.Linear(1000, 1000)
        self.layer4 = nn.Linear(1000, 1)

    def forward(self, X):
        X = F.relu(self.layer1(X))
        X = F.relu(self.layer2(X))
        X = F.relu(self.layer3(X))
        y = self.layer4(X)

        return y


class TrainData(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


class TestData(Dataset):
    pass


class NaiveFNN(object):
    def __init__(self, organisation_ids=None):
        self.organisation_ids = organisation_ids
        self.store = dt.feather_store
        if self.organisation_ids is None:
            self.organisation_ids = os.listdir(self.store)
        self.animal = None
        self.x = pd.DataFrame()
        self.y = pd.Series()
        self.x_test = None
        self.y_test = None

        self.learning_rate = 0.01
        self.momentum = 0.9
        self.epochs = 50

    def prepare_animal(self):
        data = self.animal

        # data = data.droplevel(['organisation_id', 'group_id', 'animal_id'])

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

        shifts = range(144)

        act_shift = duplicate_shift(data.act, shifts, 'act')
        group_shift = duplicate_shift(
            data.act_group_mean, shifts, 'act_group_mean')

        data = pd.concat([data, act_shift, group_shift], axis=1)
        data = data.drop(['act', 'act_group_mean'], axis=1)
        data = data.dropna()
        data.DIM = pd.to_numeric(data.DIM, downcast='signed')
        data.annotation = data.annotation.astype(int)
        self.y = self.y.append(data.annotation)
        self.x = self.x.append(data.drop('annotation', axis=1))

    def binary_acc(self, y_pred, y_test):
        y_pred_tag = torch.round(torch.sigmoid(y_pred))

        correct_results_sum = (y_pred_tag == y_test).sum().float()
        acc = correct_results_sum / y_test.shape[0]
        acc = torch.round(acc * 100)

        return acc

    def train(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        sxnet = SXNet()
        sxnet.to(device)
        optimizer = optim.Adam(
            sxnet.parameters(),
            lr=self.learning_rate
        )

        criterion = nn.BCEWithLogitsLoss()

        scaler = StandardScaler()
        self.x = scaler.fit_transform(self.x)
        self.x_test = scaler.fit_transform(self.x_test)

        traindata = TrainData(torch.FloatTensor(self.x),
                              torch.FloatTensor(self.y))

        trainloader = DataLoader(
            dataset=traindata,
            batch_size=50000,
            shuffle=True,
            num_workers=4
        )

        # testloader = torch.utils.data.DataLoader(
        #     testset,
        #     batch_size=4,
        #     shuffle=False,
        #     num_workers=2
        # )

        for e in range(self.epochs):  # loop over the dataset multiple times
            epoch_loss = 0
            epoch_acc = 0
            for x, y in trainloader:
                x = x.to(device)
                y = y.unsqueeze(1).to(device)

                optimizer.zero_grad()

                # forward + backward + optimize
                y_pred = sxnet(x)
                loss = criterion(y_pred, y)
                acc = self.binary_acc(y_pred, y)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_acc += acc.item()

            print(f'Epoch {e+0:03}: | '
                  f'Loss: {epoch_loss/len(trainloader):.5f} | '
                  f'Acc: {epoch_acc/len(trainloader):.3f}')

        print('Finished Training')

    def prepare_organisation(self, organisation_id):
        data = load_organisation(self.store, organisation_id)

        animal_ids = data.index.unique(level='animal_id')

        for animal_id in tqdm(animal_ids):
            self.animal = data.loc[(
                slice(None),
                slice(None),
                animal_id,
                slice(None)
            ), slice(None)]

            self.prepare_animal()

    def split_data(self):
        self.x, self.x_test, self.y, self.y_test = train_test_split(
            self.x, self.y, test_size=0.33, random_state=42)

    def run(self):
        self.prepare_organisation('59e7515edb84e482acce8339')
        self.split_data()
        self.train()


def main():
    obj = NaiveFNN()
    obj.run()


if __name__ == '__main__':
    main()
