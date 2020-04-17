import xgboost as xg
import numpy as np
import pandas as pd
from ml_heat.preprocessing.transform_data import DataTransformer

from ml_heat.helper import (
    load_organisation,
    duplicate_shift
)

tf = DataTransformer()


class GradientBoostedTrees(object):
    def __init__(self, organisation_ids=None):
        self.organisation_ids = organisation_ids
        self.store = tf.feather_store
        if self.organisation_ids is None:
            organisation_ids = tf.organisation_ids

    def prepare_animal(self, organisation_id):
        data = load_organisation(self.store, organisation_id)
        data = data.loc[(
            slice(None),
            slice(None),
            '59e75f2b9e182f68cf25721d',
            slice(None)
        ), slice(None)]

        data = data.droplevel(['organisation_id', 'group_id', 'animal_id'])

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

        days = 1
        shift = days * 144

        act_shift = duplicate_shift(data.act, shift, 'act')
        group_shift = duplicate_shift(
            data.act_group_mean, shift, 'act_group_mean')

        data = pd.concat([data, act_shift, group_shift], axis=1)
        data = data.drop(['act', 'act_group_mean'], axis=1)
        data = data.dropna()
        print(data.info())

    def run(self):
        self.prepare_data()
        # self.train_model()
        # self.evaluate_model()
        # self.get_data()
        # self.test()


def main():
    obj = GradientBoostedTrees()
    obj.run()


if __name__ == '__main__':
    main()
