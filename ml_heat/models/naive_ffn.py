import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

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


class Data(Dataset):
    def __init__(self, animal_ids, path):
        self.animal_ids = animal_ids
        self.store = path
        self.scaler = StandardScaler()

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

        shifts = range(144)

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

        x = self.scaler.fit_transform(x)

        return x, y


class NaiveFNN(object):
    def __init__(self, animal_ids=None):
        self.animal_ids = animal_ids
        self.store = dt.feather_store
        if self.animal_ids is None:
            self.animal_ids = os.listdir(self.store)[:200]
        self.animal = None
        self.x = None
        self.y = None
        self.x_test = None
        self.y_test = None

        self.learning_rate = 0.01
        self.momentum = 0.9
        self.epochs = 2

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
        torch.backends.cudnn.benchmark = True

        sxnet = SXNet()
        sxnet.to(device)
        optimizer = optim.Adam(
            sxnet.parameters(),
            lr=self.learning_rate
        )

        criterion = nn.BCEWithLogitsLoss()

        traindata = Data(self.partition['train'], self.store)
        testdata = Data(self.partition['validation'], self.store)

        trainloader = DataLoader(
            dataset=traindata,
            batch_size=1,
            shuffle=True,
            num_workers=4
        )

        testloader = DataLoader(
            testdata,
            batch_size=1,
            shuffle=False,
            num_workers=4
        )

        for e in range(self.epochs):  # loop over the dataset multiple times
            epoch_loss = 0
            epoch_acc = 0
            for x, y in tqdm(trainloader, desc='epoch progress'):
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

            print(f'Epoch {e+0:03}: | '
                  f'Loss: {epoch_loss/len(trainloader):.5f} | '
                  f'Acc: {epoch_acc/len(trainloader):.3f}')

        print('Finished Training')

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
                y_list.append(y.cpu().numpy())
        y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
        y_list = [a.squeeze().tolist() for a in y_list]

        print(classification_report(
            self.y_test, y_pred_list,
            target_names=['no heat', 'heat'],
            digits=6
        ))

    def run(self):
        self.train()


def main():
    obj = NaiveFNN()
    obj.run()


if __name__ == '__main__':
    main()
