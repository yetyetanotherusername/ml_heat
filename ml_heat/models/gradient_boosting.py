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
        self.data = load_vxframe(self.vxstore)

    def prepare_data(self):
        pass

    def train_model(self):
        pass

    def evaluate_model(self):
        pass

    def run(self):
        self.prepare_data()
        self.train_model()
        self.evaluate_model()


def main():
    obj = GradientBoostedTrees()
    obj.run()


if __name__ == '__main__':
    main()
