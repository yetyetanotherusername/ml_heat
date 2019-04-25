#!/usr/bin/python3
# coding: utf8

import os
import json
import pickle
import h5py as h5
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


def transform_organisation(organisation_id, readpath, temp_path):
    with h5.File(readpath, 'r') as readfile:

        animal_ids = list(readfile[f'data/{organisation_id}'].keys())
        animal_ids = list(filter(
            lambda x: x != 'organisation', animal_ids))

        framelist = []
        for animal_id in animal_ids:
            animal = json.loads(
                readfile[f'data/{organisation_id}/{animal_id}/animal'][()])

            try:
                data = pd.read_hdf(
                    readpath,
                    key=f'data/{organisation_id}/{animal_id}/sensordata')
            except KeyError:
                continue
            except Exception as e:
                print(e)

            if data.empty:
                continue

            # TODO: add annotations from animal object

            data['organisation_id'] = organisation_id
            data['group_id'] = animal['group_id']
            data['animal_id'] = animal_id

            framelist.append(data)

    if not framelist:
        return organisation_id

    frame = pd.concat(framelist, sort=False)
    frame.index.names = ['datetime']

    frame = frame.set_index(
        ['organisation_id', 'group_id', 'animal_id', frame.index])

    frame = frame.sort_index()

    write_path = os.path.join(temp_path, organisation_id)
    with open(write_path, 'wb') as writefile:
        pickle.dump(frame, writefile)

    return organisation_id


class DataTransformer(object):
    def __init__(self, organisation_ids=None):
        self._organisation_ids = organisation_ids
        self.store_path = os.path.join(os.getcwd(), 'ml-heat/__data_store__')
        self.raw_store_path = os.path.join(self.store_path, 'rawdata.hdf5')
        self.train_store_path = os.path.join(self.store_path, 'traindata.hdf5')

        self.process_pool = ProcessPoolExecutor(os.cpu_count())

    def readfile(self):
        return h5.File(self.raw_store_path, 'r')

    @property
    def organisation_ids(self):
        if self._organisation_ids is None:
            with self.readfile() as file:
                self._organisation_ids = list(file['data'].keys())
        return self._organisation_ids

    def transform_data(self):
        print('Transforming data...')
        # create temp storage folder
        temp_path = os.path.join(self.store_path, 'temp')
        if not os.path.exists(temp_path):
            os.mkdir(temp_path)

        results = [self.process_pool.submit(
            transform_organisation, _id, self.raw_store_path, temp_path)
            for _id in self.organisation_ids]

        kwargs = {
            'total': len(results),
            'unit': 'organisations',
            'unit_scale': True,
            'leave': True
        }

        for f in tqdm(as_completed(results), **kwargs):
            pass

        print('Transformation finished')

    def store_data(self):
        print('Writing data to hdf file...')
        temp_path = os.path.join(self.store_path, 'temp')
        files = os.listdir(temp_path)
        filepaths = [os.path.join(temp_path, p) for p in files]

        with pd.HDFStore(self.train_store_path) as train_store:
            for filepath in tqdm(filepaths):
                with open(filepath, 'rb') as file:
                    frame = pickle.load(file)

                if frame.empty:
                    os.remove(filepath)
                    continue

                try:
                    train_store.append(key='dataset', value=frame)
                    os.remove(filepath)
                except KeyError as e:
                    print(e)
                except ValueError as e:
                    print(e)

        print('Finished writing training data...')
        print('Cleaning up')
        os.rmdir(temp_path)
        print('Done!')

    def run(self):
        self.transform_data()
        self.store_data()


def main():
    transformer = DataTransformer()
    transformer.run()


if __name__ == '__main__':
    main()
