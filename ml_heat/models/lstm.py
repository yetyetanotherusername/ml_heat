import os
import zarr
from tqdm import tqdm

from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Dataset

from ml_heat.preprocessing.transform_data import DataTransformer
from ml_heat.helper import plot_setup


dt = DataTransformer()


class SXlstm(nn.Module):
    def __init__(self, device, batch_size, input_size=4, h_size=100, layers=1, out_size=1):
        super().__init__()

        self.h_size = h_size
        self.layers = layers
        self.device = device
        self.batch_size = batch_size
        self.out_size = out_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=h_size,
            num_layers=layers,
            batch_first=True
        ).to(device)

        self.reset_state()

        self.linear = nn.Linear(h_size, out_size).to(device)

    def forward(self, X, lengths):

        batch_size, seq_len, _ = X.size()

        self.reset_state()

        X = torch.nn.utils.rnn.pack_padded_sequence(
            X, lengths, batch_first=True, enforce_sorted=False)

        Y, self.state = self.lstm(X, self.state)

        Y, _ = torch.nn.utils.rnn.pad_packed_sequence(
            Y, batch_first=True)

        Y = Y.contiguous()
        Y = Y.view(-1, Y.shape[2])
        Y = self.linear(Y)

        return Y.view(batch_size, seq_len, self.out_size)

    def reset_state(self):
        self.state = (
            torch.autograd.Variable(
                torch.randn(
                    self.layers,
                    self.batch_size,
                    self.h_size
                ).to(self.device)
            ),
            torch.autograd.Variable(
                torch.randn(
                    self.layers,
                    self.batch_size,
                    self.h_size
                ).to(self.device)
            )
        )


class Data(Dataset):
    def __init__(self, path, set_name):
        super(Data, self).__init__()
        self.store = zarr.DirectoryStore(path)
        self.group = zarr.hierarchy.group(store=self.store)

        self.keys = self.group[set_name]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        return torch.from_numpy(self.group[self.keys[idx]][:, :])


class PadSequence:
    def __call__(self, batch):
        lengths = torch.FloatTensor([x.shape[0] for x in batch])
        padded_batch = torch.nn.utils.rnn.pad_sequence(
            batch, batch_first=True, padding_value=-1)
        return padded_batch, lengths


class LSTM(object):
    def __init__(self):
        self.store = dt.zarr_store
        self.model_store = os.path.join(
            os.getcwd(), 'ml_heat', '__data_store__', 'models', 'lstm')
        self.model_path = os.path.join(self.model_store, 'lstm')

        self.epochs = 10
        self.learning_rate = 0.001
        self.momentum = 0.9
        self.batch_size = 50

    def binary_acc(self, y_pred, y_test):
        y_pred_tag = torch.round(torch.sigmoid(y_pred))

        correct_results_sum = (y_pred_tag == y_test).sum().float()
        acc = correct_results_sum / y_test.shape[0]
        acc = torch.round(acc * 100)

        return acc

    def reshape_outputs(self, y_pred, y):
        mask = y > -0.5
        mask = mask.flatten()
        y = y.flatten()[mask]
        y_pred = y_pred.flatten()[mask]

        return y_pred, y

    def train(self):
        use_cuda = torch.cuda.is_available()

        device = torch.device("cuda:0" if use_cuda else "cpu")

        if use_cuda:
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True

        sxnet = SXlstm(device, self.batch_size)
        try:
            sxnet.load_state_dict(
                torch.load(self.model_path, map_location=device))
        except Exception as e:
            print(e)

        optimizer = optim.Adam(
            sxnet.parameters(),
            lr=self.learning_rate
        )

        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.FloatTensor([100.]).to(device))

        traindata = Data(self.store, 'train_keys')
        trainloader = DataLoader(
            dataset=traindata,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=os.cpu_count(),
            pin_memory=True,
            collate_fn=PadSequence()
        )

        params = {
            'desc': 'epoch progress',
            'smoothing': 0.01,
            'total': len(traindata) // self.batch_size
        }

        losses = []
        accuracies = []

        for e in range(self.epochs):  # loop over the dataset multiple times
            epoch_loss = 0
            epoch_acc = 0
            epoch_len = 0
            for batch, lengths in tqdm(trainloader, **params):
                batch = batch.to(device)
                lengths = lengths.to(device)

                x = batch[:, :, 1:]
                y = batch[:, :, 0]

                optimizer.zero_grad()

                # forward + backward + optimize
                y_pred = sxnet(x, lengths)

                y_pred, y = self.reshape_outputs(y_pred, y)

                loss = criterion(y_pred, y)
                acc = self.binary_acc(y_pred, y)

                loss.backward(retain_graph=True)
                optimizer.step()
                epoch_loss += loss.item()
                epoch_acc += acc.item()
                epoch_len += 1

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

        sxnet = SXlstm(device, self.batch_size)
        try:
            sxnet.load_state_dict(torch.load(
                self.model_path, map_location=device))
        except FileNotFoundError:
            print('Warning: no trained model found')
        except Exception as e:
            print(e)
        torch.multiprocessing.set_sharing_strategy('file_system')

        testdata = Data(self.store, 'test_keys')
        testloader = DataLoader(
            testdata,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=os.cpu_count(),
            pin_memory=True,
            collate_fn=PadSequence()
        )

        y_pred_list = []
        y_list = []
        sxnet.eval()

        params = {
            'desc': 'validation progress',
            'smoothing': 0.01,
            'total': len(testdata)
        }

        with torch.no_grad():
            for batch, lengths in tqdm(testloader, **params):
                batch = batch.to(device)
                lengths = lengths.to(device)

                x = batch[:, :, 1:]
                y = batch[:, :, 0]

                y_pred = torch.sigmoid(sxnet(x, lengths))
                y_pred = torch.round(y_pred)

                y_pred, y = self.reshape_outputs(y_pred, y)

                y_pred_list.extend(y_pred.cpu().tolist())
                y_list.extend(y.cpu().tolist())

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
