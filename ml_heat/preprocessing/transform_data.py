#!/usr/bin/python3
# coding: utf8

import os
import json
import pickle
import h5py as h5
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from sxutils.models.animal.cycle import AnimalLifecycle
from sxutils.munch.cyclemunch import cycle_munchify
from animal_serde import AnimalSchemaV2


def calculate_dims(index, animal):
    animal_cycle = AnimalLifecycle()

    # transform animal document timestamps
    animal = AnimalSchemaV2().load(animal).data

    lifecycle_object = cycle_munchify(animal['lifecycle'])

    for cycle in lifecycle_object.cycles_with_padding:
        for event in cycle.events:
            animal_cycle.addCycleEvent(event)

    # pass only unique dates to dim calc to reduce overhead
    unique_dates = index.normalize().drop_duplicates()
    dims = animal_cycle.dims(unique_dates)

    series = pd.Series(dims, index=unique_dates, name='dims')
    left = pd.Series(0, index=index, name='util')
    frame = pd.concat([left, series], axis=1)
    frame.dims = frame.dims.fillna(method='ffill')
    frame = frame.dropna()

    return frame.dims


def transform_organisation(organisation_id, readpath, temp_path):
    with h5.File(readpath, 'r') as readfile:

        organisation = json.loads(
            readfile[f'data/{organisation_id}/organisation'][()])

        animal_ids = list(readfile[f'data/{organisation_id}'].keys())
        animal_ids = list(filter(
            lambda x: x != 'organisation', animal_ids))

        framelist = []
        for animal_id in animal_ids:
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

            animal = json.loads(
                readfile[f'data/{organisation_id}/{animal_id}/animal'][()])

            # TODO: add annotations from animal object

            # remove localization -> index is localtime without tzinfo
            # needed so we can have all animal indices in one column
            data = data.tz_localize(None)

            # shorten string fields (hdf5 serde has limits on string length)
            if len(animal['race']) > 9:
                race = [word[0] + '_' for word in animal['race'].split('_')]
                race = ''.join(race)
            else:
                race = animal['race']

            if len(organisation['partner_id']) > 9:
                partner_id = organisation['partner_id'][:9]
            else:
                partner_id = organisation['partner_id']

            # country field may be unavailable
            country = animal.get('metadata', {}).get('country', float('nan'))

            data['organisation_id'] = organisation_id
            data['group_id'] = animal['group_id']
            data['animal_id'] = animal_id
            data['race'] = race
            data['country'] = country
            data['partner_id'] = partner_id
            data['DIM'] = calculate_dims(data.index, animal)

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
        self.store_path = os.path.join(os.getcwd(), 'ml_heat/__data_store__')
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

        # for organisation_id in self.organisation_ids:
        #     transform_organisation(
        #         organisation_id,
        #         self.raw_store_path,
        #         temp_path)

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

        with pd.HDFStore(self.train_store_path, complevel=9) as train_store:
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
                    print(frame)
                    print(e)

        print('Finished writing training data...')
        print('Cleaning up...')
        os.rmdir(temp_path)
        print('Done!')

    def clear_data(self):
        if os.path.exists(self.train_store_path):
            os.remove(self.train_store_path)

    def test(self):
        print('Test: Loading data...')
        frame = pd.read_hdf(self.train_store_path, key='dataset')

        print(
            frame.loc[(
                '59e7515edb84e482acce8339',
                '59e75177575fc94638c1f8e7',
                '59e75f2b9e182f68cf25721d')])

    def run(self):
        self.clear_data()
        self.transform_data()
        self.store_data()
        self.test()


def main():
    transformer = DataTransformer()
    transformer.run()


if __name__ == '__main__':
    main()