#!/usr/bin/python3
# coding: utf8

import os
import statsmodels.api as sm
import pandas as pd


class ArimaTest(object):
    def __init__(self):
        self.store_path = os.path.join(os.getcwd(), 'ml_heat/__data_store__')
        self.train_store_path = os.path.join(self.store_path, 'traindata.hdf5')
        self.data = None
        self.result = None

    def load_data(self):
        query = 'animal_id == "59e75f2b9e182f68cf25721d"'
        self.data = pd.read_hdf(
            self.train_store_path,
            'dataset',
            where=query).reset_index().set_index('datetime').act

    def train_model(self):
        model = sm.tsa.ARIMA(self.data, (5, 1, 5))
        fitted_model = model.fit()
        self.result = fitted_model.predict()

    def plot(self):
        print(self.result)

    def run(self):
        self.load_data()
        self.train_model()
        self.plot()


def main():
    test = ArimaTest()
    test.run()


if __name__ == '__main__':
    main()
