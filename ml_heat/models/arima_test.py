#!/usr/bin/python3
# coding: utf8

import os
import statsmodels.api as sm
import pandas as pd

from ml_heat.helper import (
    load_vxframe,
    vaex_to_pandas,
    plot_setup
)


class ArimaTest(object):
    def __init__(self):
        self.store_path = os.path.join(
            os.getcwd(), 'ml_heat', '__data_store__', 'vaex_store')
        self.data = None
        self.result = None

    def load_data(self):
        vxframe = load_vxframe(self.store_path)
        vxframe = vxframe[vxframe.animal_id == '59e75f2b9e182f68cf25721d']
        self.data = vaex_to_pandas(
            vxframe).reset_index().set_index('datetime').act

    def train_model(self):
        # model = sm.tsa.SARIMAX(
        #     self.data, order=(5, 0, 5), seasonal_order=(5, 0, 5, 144)
        # )

        model = sm.tsa.ARIMA(self.data, order=(1, 0, 1))
        fitted_model = model.fit()
        self.result = fitted_model.predict()

    def plot(self):
        plt = plot_setup()
        print(self.result)
        self.result.name = 'result'
        self.data.name = 'data'
        pd.concat([self.data, self.result], axis=1).plot()
        plt.grid()
        plt.show()

    def run(self):
        self.load_data()
        self.train_model()
        self.plot()


def main():
    test = ArimaTest()
    test.run()


if __name__ == '__main__':
    main()
