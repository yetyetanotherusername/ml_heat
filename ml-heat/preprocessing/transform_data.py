#!/usr/bin/python3
# coding: utf8

import os
import json
import h5py as h5
import pandas as pd


class DataTransformer(object):
    def __init__(self, organisation_ids=None):
        self.train_store_path = (os.getcwd() +
                                 '/ml-heat/__data_store__/traindata.hdf5')
        self.raw_store_path = (os.getcwd() +
                               '/ml-heat/__data_store__/rawdata.hdf5')

        self._organisation_ids = organisation_ids

    def readfile(self):
        return h5.File(self.raw_store_path, 'r')

    @property
    def organisation_ids(self):
        if self._organisation_ids is None:
            with self.readfile() as file:
                self._organisation_ids = list(file['data'].keys())
        return self._organisation_ids

    def transform_data(self):
        print('Loading data from organisations')
        olen = len(self.organisation_ids)
        with self.readfile() as file, \
                pd.HDFStore(self.train_store_path) as train_store:

            for idx, organisation_id in enumerate(self.organisation_ids):
                framelist = []

                pstring = ('Loading organisations: '
                           f'{round((idx + 1) / olen * 100)}%, '
                           f'{idx + 1}/{olen}')

                # organisation = json.loads(
                #     file[f'data/{organisation_id}/organisation'][()])

                animal_ids = list(file[f'data/{organisation_id}'].keys())

                animal_ids = list(filter(
                    lambda x: x != 'organisation', animal_ids))

                alen = len(animal_ids)
                for aidx, animal_id in enumerate(animal_ids):

                    astring = (' | Loading animals: '
                               f'{round((aidx + 1) / alen * 100)}%, '
                               f'{aidx + 1}/{alen}')

                    print(pstring + astring + (' ' * 30), end='\r', flush=True)

                    animal = json.loads(
                        file[f'data/{organisation_id}/{animal_id}/animal'][()])

                    try:
                        data = pd.read_hdf(
                            self.raw_store_path,
                            key=f'data/{organisation_id}/{animal_id}/sensordata')
                    except KeyError:
                        continue

                    data['organisation_id'] = organisation_id
                    data['group_id'] = animal['group_id']
                    data['animal_id'] = animal_id

                    framelist.append(data)

                if not framelist:
                    continue
                print(f'{pstring}{astring} | Storing to file...',
                      end='\r', flush=True)
                frame = pd.concat(framelist, sort=False)
                frame.index.names = ['datetime']

                frame = frame.set_index(
                    ['organisation_id', 'group_id', 'animal_id', frame.index])

                frame = frame.sort_index()

                try:
                    train_store.append(key='dataset', value=frame)
                except KeyError as e:
                    print(e)
                except ValueError as e:
                    print(e)

    def run(self):
        self.transform_data()


def main():
    transformer = DataTransformer()
    transformer.run()


if __name__ == '__main__':
    main()
