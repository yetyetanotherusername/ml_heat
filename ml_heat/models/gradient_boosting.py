# import xgboost as xg
import vaex as vx
import numpy as np

from ml_heat.preprocessing.transform_data import DataTransformer

from ml_heat.helper import (
    load_data,
    load_vxframe,
    animal_count_per_orga,
    vaex_to_pandas
)


@vx.register_dataframe_accessor('mlheat', override=True)
class mlheat:
    def __init__(self, df):
        self.df = df

    def shift(self, column, n, inplace=False):
        # make a copy without column
        df = self.df.copy().drop(column)
        # make a copy with just the colum
        df_column = self.df[[column]]
        # slice off the head and tail
        df_head = df_column[-n:]
        df_tail = df_column[:-n]
        # stitch them together
        df_shifted = df_tail.concat(df_head)
        print(df_shifted)
        # and join (based on row number)
        return df.join(df_shifted, inplace=inplace)


class GradientBoostedTrees(object):
    def __init__(self, organisation_ids=None):
        self.organisation_ids = organisation_ids
        self.vxstore = DataTransformer().vxstore
        self.data = load_vxframe(self.vxstore)

    # def prepare_data(self):
    #     x = np.arange(10)
    #     y = x**2
    #     df = vx.from_arrays(x=x, y=y)
    #     df['shifted_y'] = df.y
    #     df = df.mlheat.shift('shifted_y', 2)
    #     print(df)

    def prepare_data(self):
        data = self.data[
            self.data.organisation_id == '5b84ecc2f812a709c8b0cc23']
        data = data.drop(
            ['organisation_id',
             'temp',
             'temp_filtered',
             'race',
             'country',
             'deleted',
             'group_id'])
        data = data.to_pandas_df().dropna().set_index(
            ['animal_id', 'datetime']).sort_index()
        print(data)

    def test(self):
        transformer = DataTransformer()
        print(transformer.organisation_ids)
        print(
            animal_count_per_orga(
                transformer.readfile, transformer.organisation_ids))

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
