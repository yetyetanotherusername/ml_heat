import zarr
import os
import torch
import math
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from ml_heat.preprocessing.transform_data import DataTransformer
from ml_heat.models.naive_ffn import SXNet, Data, worker_init_fn
from ml_heat.helper import plot_setup


dt = DataTransformer()


class ClassificationPlot(object):
    def __init__(self):
        self.zarr_path = os.path.join(dt.zarr_store, 'origin')
        self.zarr_store = zarr.DirectoryStore(self.zarr_path)
        self.zarr_array = zarr.open(self.zarr_store, mode='r')
        self.model_store = os.path.join(
            dt.model_store, 'naive_ffn', 'naive_ffn')
        self.data_start = 0
        self.data_end = 1000000

    def init_model(self):
        use_cuda = torch.cuda.is_available()

        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        # torch.backends.cudnn.benchmark = True

        sxnet = SXNet()
        sxnet.to(self.device)
        sxnet.load_state_dict(
            torch.load(self.model_store, map_location=self.device))
        self.model = sxnet

    def init_dataloader(self):
        if self.device == 'cpu':
            num_workers = 0
        else:
            num_workers = os.cpu_count()

        self.data = Data(self.zarr_path, self.data_start, self.data_end)
        dataloader = DataLoader(
            dataset=self.data,
            batch_size=50000,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn
        )
        self.dataloader = dataloader

    def run_simulation(self):
        y_pred_list = []
        y_list = []
        self.model.eval()

        params = {
            'desc': 'validation progress',
            'smoothing': 0.01,
            'total': math.ceil(
                (self.data_end - self.data_start) / self.dataloader.batch_size)
        }

        with torch.no_grad():
            for batch in tqdm(self.dataloader, **params):
                x = batch[:, 1:].to(self.device)
                y = batch[:, 0].unsqueeze(0).T
                y_test_pred = torch.sigmoid(self.model(x))
                y_pred_tag = torch.round(y_test_pred)
                y_pred_list.append(y_pred_tag.cpu().numpy().flatten())
                y_list.append(y.numpy().flatten())

        self.y_list = np.concatenate(y_list)
        self.y_pred_list = np.concatenate(y_pred_list)

    def plot(self):
        plt = plot_setup()
        data = self.zarr_array[self.data_start:self.data_end, 1:3]

        dims = data[:, 0]
        heat_feature = data[:, 1]

        plt.plot(dims)
        plt.plot(heat_feature)

        ax = plt.gca()
        ymax, ymin = ax.get_ylim()

        ax.fill_between(
            range(0, self.data_end - self.data_start), ymin, ymax,
            where=self.y_pred_list,
            facecolor='r',
            alpha=0.5,
            label='classification'
        )

        ax.fill_between(
            range(0, self.data_end - self.data_start), ymin, ymax,
            where=self.y_list,
            facecolor='g',
            alpha=0.5,
            label='classification'
        )

        plt.grid()
        plt.show()

    def run(self):
        self.init_model()
        self.init_dataloader()
        self.run_simulation()
        self.plot()


def main():
    obj = ClassificationPlot()
    obj.run()


if __name__ == '__main__':
    main()
