import os
import xgboost as xg
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from ml_heat.preprocessing.transform_data import DataTransformer

from ml_heat.helper import (
    load_organisation,
    duplicate_shift
)

dt = DataTransformer()


class GradientBoostedTrees(object):
    def __init__(self, organisation_ids=None):
        self.organisation_ids = organisation_ids
        self.store = dt.feather_store
        if self.organisation_ids is None:
            self.organisation_ids = os.listdir(self.store)
        self.animal = None
        self.X = pd.DataFrame()
        self.y = pd.Series()
        self.X_test = None
        self.y_test = None

        self.model = xg.XGBClassifier(
            tree_method='gpu_hist',
            verbose=3,
            n_estimators=10
        )

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
        self.y = self.y.append(data.annotation)
        self.X = self.X.append(data.drop('annotation', axis=1))

    def fit_model(self):
        self.model.fit(
            self.X,
            self.y,
            verbose=True,
            # eval_set=[(self.X, self.y), (self.X_test, self.y_test)],
            # eval_metric='rmse'
        )

    def loop_organisation(self, organisation_id):
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
        self.X, self.X_test, self.y, self.y_test = train_test_split(
            self.X, self.y, test_size=0.33, random_state=42)

    def run(self):
        self.loop_organisation('59e7515edb84e482acce8339')
        # self.split_data()
        self.fit_model()
        # self.evaluate_model()
        # self.get_data()
        # self.test()


def main():
    obj = GradientBoostedTrees()
    obj.run()


if __name__ == '__main__':
    main()
