#!/usr/bin/python3
# coding: utf8

import os
import shutil
import pickle
import unittest
import numpy as np
import pandas as pd

from ml_heat.data_loading.get_data import (
    parse_csv,
    DataLoader
)

TESTDIR = 'tests/__testdir__'


class GetDataTests(unittest.TestCase):
    def setUp(self):
        os.mkdir(TESTDIR)

    def tearDown(self):
        if os.path.exists(TESTDIR):
            shutil.rmtree(TESTDIR)

    def test_parse_csv(self):
        data = zip(np.random.rand(100), np.random.rand(100))
        frame = pd.DataFrame(
            data,
            index=pd.date_range(
                start=pd.Timestamp(2019, 1, 1),
                periods=100,
                freq='10T'))

        testfile = os.path.join(TESTDIR, 'testfile.csv')
        frame.to_csv(testfile)

        parse_csv(testfile, 'UTC')

        readpath = testfile.replace('.csv', '')

        with open(readpath, 'rb') as file:
            ret = pickle.load(file)

        frame = frame.tz_localize('UTC')
        frame.columns = frame.columns.values.astype(str)

        pd.testing.assert_frame_equal(frame, ret)
