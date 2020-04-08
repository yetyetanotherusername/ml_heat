import xgboost as xg
import vaex as vx
import numpy as np

from ml_heat.preprocessing.transform_data import DataTransformer
from ml_heat.helper import (
    load_data,
    load_vxframe
)


@vx.register_dataframe_accessor('mlheat', override=True)
class mlheat:
    def __init__(self, df):
        self.df = df

    def shift(self, column, n, cyclic=True, fill_value=None, inplace=False):
        # make a copy without column
        df = self.df.copy().drop(column)
        # make a copy with just the colum
        df_column = self.df[[column]]
        # slice off the head and tail
        if cyclic:
            df_head = df_column[-n:]
        else:
            df_head = vx.from_dict(
                {column: np.ma.masked_all(n, dtype=self.df[column].dtype)})
        df_tail = df_column[:-n]
        # stitch them together
        df_shifted = df_head.concat(df_tail)
        # and join (based on row number)
        return df.join(df_shifted, inplace=inplace)


class GradientBoostedTrees(object):
    def __init__(self, organisation_ids=None):
        self.organisation_ids = organisation_ids
        self.vxstore = DataTransformer().vxstore
        self.data = load_vxframe(self.vxstore)

    def prepare_data(self):
        x = np.arange(10)
        y = x**2
        df = vx.from_arrays(x=x, y=y)
        df['shifted_y'] = df.y
        df2 = df.mlheat.shift('shifted_y', -2, cyclic=True, fill_value=np.nan)
        print(df2)

    def get_data(self):
        data = load_vxframe(self.vxstore)
        data = data.sort(['organisation_id', 'group_id', 'animal_id', ])
        print(data.head())

    def run(self):
        self.prepare_data()
        self.train_model()
        self.evaluate_model()
        self.get_data()


def main():
    obj = GradientBoostedTrees()
    obj.run()


if __name__ == '__main__':
    main()
