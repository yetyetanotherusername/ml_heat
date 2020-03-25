import xgboost as xg
import vaex as vx

from ml_heat.preprocessing.transform_data import DataTransformer
from ml_heat.helper import (
    load_data,
    load_vxframe
)


class GradientBoostedTrees(object):
    def __init__(self, organisation_ids=None):
        self.organisation_ids = organisation_ids
        self.vxstore = DataTransformer().vxstore

    def get_data(self):
        data = load_vxframe(self.vxstore)

        print(data.head())

    def run(self):
        self.get_data()


def main():
    obj = GradientBoostedTrees()
    obj.run()


if __name__ == '__main__':
    main()
