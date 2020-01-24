import os
import shutil
import unittest
import pandas as pd
import vaex as vx

from ml_heat.helper import load_data


TESTDIR = 'tests/__testdir__'


class HelperTests(unittest.TestCase):
    def setUp(self):
        os.mkdir(TESTDIR)

    def tearDown(self):
        if os.path.exists(TESTDIR):
            shutil.rmtree(TESTDIR)

    def test_get_data(self):
        # testfile path

        filepath = os.path.join(TESTDIR, 'testfile.hdf5')

        # create testdata
        organisation_ids = ['aa', 'aa', 'ab', 'ac']
        group_ids = ['bb', 'ba', 'bd', 'bc']
        animal_ids = ['ca', 'cb', 'cc', 'cd']
        temp = [1, 2, 3, 4]
        datetimes = [
            pd.Timestamp(2020, 1, 1, 0, 0, 0),
            pd.Timestamp(2020, 1, 1, 0, 10, 0),
            pd.Timestamp(2020, 1, 1, 0, 20, 0),
            pd.Timestamp(2020, 1, 1, 0, 30, 0)]

        target = pd.DataFrame(
            {'organisation_id': organisation_ids,
             'temp': temp,
             'group_id': group_ids,
             'animal_id': animal_ids,
             'datetime': datetimes})

        vxframe = vx.from_pandas(target)
        vxframe.export_hdf5(filepath)

        target = target.set_index(
            ['organisation_id',
             'group_id',
             'animal_id',
             'datetime']).sort_index()

        # test function
        loaded_frame = load_data(TESTDIR)
        pd.testing.assert_frame_equal(target, loaded_frame)

        # test with subset of organisations
        loaded_frame = load_data(TESTDIR, ['aa', 'ab'])

        pd.testing.assert_frame_equal(
            target.loc[['aa', 'ab']],
            loaded_frame)

        # TODO: test with vaex return dtype
        # on hold until i know how to test for equality in vaex
        # loaded_frame = load_data(path, dtype='vaex')
        # print(vxframe)
        # print(loaded_frame)
        # self.assertEqual(vxframe, loaded_frame)
