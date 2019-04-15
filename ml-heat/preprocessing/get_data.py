#!/usr/bin/python3
# coding: utf8

import os
import json
import datetime
import h5py as h5
import pandas as pd
from sxapi import LowLevelAPI, APIv2

PRIVATE_TOKEN = ('i-am-no-longer-a-token')

PRIVATE_ENDPOINT = 'http://127.0.0.1:8787/internapi/v1/'

PUBLIC_ENDPOINTv2 = 'https://api-staging.smaxtec.com/api/v2/'


class DataLoader(object):
    def __init__(self):
        self.api = APIv2(
            api_key=PRIVATE_TOKEN, endpoint=PUBLIC_ENDPOINTv2,
            asynchronous=True).low

        self.oldapi = LowLevelAPI(
            api_key=PRIVATE_TOKEN,
            private_endpoint=PRIVATE_ENDPOINT).privatelow

        self.store_path = os.getcwd() + '/ml-heat/__data_store__/rawdata.hdf5'
        self._organisation_ids = None
        self._animal_ids = None
        self._animal_orga_map = None

    def readfile(self):
        return h5.File(self.store_path, 'r')

    def writefile(self):
        return h5.File(self.store_path, 'a')

    @property
    def organisation_ids(self):
        if not self._organisation_ids:
            with self.readfile() as file:
                self._organisation_ids = list(file['data'].keys())
        return self._organisation_ids

    @property
    def animal_ids(self):
        if not self._animal_ids:
            self._animal_ids = []
            with self.readfile() as file:
                for organisation_id in self._organisation_ids:
                    self._animal_ids += list(
                        file[f'data/{organisation_id}'].keys())

            self._animal_ids = [
                x for x in self._animal_ids if x != 'organisation']

        return self._animal_ids

    def animal_ids_for_organisations(self, organisation_ids):
        animal_ids = []
        with self.readfile() as file:
            for organisation_id in organisation_ids:
                ids = list(file[f'data/{organisation_id}'].keys())
                filtered = [x for x in ids if x != 'organisation']
                animal_ids += filtered
        return animal_ids

    def organisation_id_for_animal_id(self, animal_id):
        if self._animal_orga_map is None:
            with self.readfile() as file:
                self._animal_orga_map = json.loads(
                    file['lookup/animal_to_orga'][()])

        try:
            return self._animal_orga_map[animal_id]
        except KeyError:
            return None
        except Exception as e:
            print(e)

    def load_organisations(self, update=False):
        # TODO: switch to apiv2 once implemented

        with self.writefile() as file:
            if 'data' not in file.keys() or update:
                print('Loading organisations')
                organisations = self.oldapi.query_organisations()
                try:
                    data_group = file.create_group(name='data')
                except ValueError:
                    data_group = file['data']

                print('Storing organisations to local cache')
                for organisation in organisations:
                    try:
                        orga_group = data_group.create_group(
                            name=organisation['_id'])
                    except ValueError:
                        orga_group = data_group[organisation['_id']]
                    if 'organisation' in orga_group.keys():
                        del orga_group['organisation']

                    orga_group.create_dataset(
                        name='organisation',
                        data=json.dumps(organisation))
            else:
                print('Organisations found in store, skipped loading')

    def load_animals(self, organisation_ids=None, update=False):
        if organisation_ids is None:
            organisation_ids = self.organisation_ids

        if not update:
            with self.readfile() as file:
                keys = file['data'].keys()

                keys = [x for x in keys if len(file[f'data/{x}'].keys()) > 1]

            filtered_orga_ids = [x for x in organisation_ids if x not in keys]

        if not filtered_orga_ids:
            return

        print('Loading animals')
        returns = self.api.get_animals_by_organisation_ids_async(
            filtered_orga_ids)

        animals = list(zip(*returns))[0]

        # flatten list of lists
        animals = [inner for outer in animals for inner in outer]

        print('Storing animals')
        with self.writefile() as file:
            if self._animal_orga_map is None:
                try:
                    self._animal_orga_map = json.loads(
                        file['lookup/animal_to_orga'][()])
                except Exception:
                    self._animal_orga_map = {}

            for animal in animals:
                organisation_id = animal['organisation_id']
                orga_group = file[f'data/{organisation_id}']

                self._animal_orga_map[animal['_id']] = organisation_id

                try:
                    a_subgroup = orga_group.create_group(
                        name=animal['_id'])
                except ValueError:
                    a_subgroup = orga_group[animal['_id']]

                if 'animal' in a_subgroup.keys():
                    del a_subgroup['animal']

                a_subgroup.create_dataset(
                    name='animal',
                    data=json.dumps(animal))

            try:
                lookup_group = file.create_group('lookup')
            except ValueError:
                lookup_group = file['lookup']

            if 'animal_to_orga' in lookup_group.keys():
                del lookup_group['animal_to_orga']

            lookup_group.create_dataset(
                name='animal_to_orga',
                data=json.dumps(self._animal_orga_map))

    def load_sensordata(self,
                        organisation_ids=None,
                        update=False,
                        metrics=['act', 'temp']):

        from_dt = str(datetime.date(2018, 12, 1))
        to_dt = str(datetime.date(2019, 4, 1))

        print('Preparing to load sensor data')
        if organisation_ids is None:
            organisation_ids = self.organisation_ids

        # retrieve animal ids to load data for
        animal_ids = self.animal_ids_for_organisations(organisation_ids)

        # determine which datafiles haven't been loaded yet
        if not update:
            filtered = []
            with self.readfile() as file:
                for organisation_id in organisation_ids:
                    keys = list(file[f'data/{organisation_id}'].keys())
                    animal_ids = [key for key in keys if key != 'organisation']
                    for animal_id in animal_ids:
                        a_keys = list(
                            file[f'data/{organisation_id}/{animal_id}'].keys())
                        if len(a_keys) < 2:
                            filtered.append(animal_id)
            animal_ids = filtered

        chunksize = 50  # animals per load - store cycle
        chunks = [animal_ids[x:x + chunksize] for x in
                  range(0, len(animal_ids), chunksize)]
        num_chunks = len(chunks)

        for idx, animal_ids in enumerate(chunks):
            print('\rProcessing chunks: '
                  f'{round((idx + 1) / num_chunks * 100)}% done, '
                  f'{idx + 1}/{num_chunks}', end='')

            data = self.api.get_data_by_animal_ids_async(
                animal_ids, from_dt, to_dt, metrics)

            for tupl in data:
                ret = tupl[0]
                animal_id = tupl[1].split('.')[-2].split('/')[-1]
                organisation_id = self.organisation_id_for_animal_id(animal_id)

                if organisation_id is None:
                    continue

                frame = pd.DataFrame()
                for metric_dict in ret:
                    if not metric_dict['data']:
                        continue
                    transposed_data = list(zip(*metric_dict['data']))
                    right = pd.DataFrame(index=transposed_data[0],
                                         data=transposed_data[1],
                                         columns=[metric_dict['metric']])

                    frame = pd.concat(
                        [frame, right], axis=1, join='outer', sort=True)

                if frame.empty:
                    continue

                frame.to_hdf(
                    self.store_path,
                    key=f'data/{organisation_id}/{animal_id}/sensordata',
                    complevel=9)

    def run(self, organisation_ids=None, update=False):
        self.load_organisations(update)
        self.load_animals(organisation_ids=organisation_ids, update=update)
        self.load_sensordata(organisation_ids=organisation_ids, update=update)


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
        frame = pd.DataFrame()
        print('Loading data from organisations')
        olen = len(self.organisation_ids)
        for idx, organisation_id in enumerate(self.organisation_ids):

            pstring = ('\rLoading organisations: '
                       f'{round((idx + 1) / olen * 100)}% done, '
                       f'{idx + 1}/{olen}')

            with self.readfile() as file:
                # organisation = json.loads(
                #     file[f'data/{organisation_id}/organisation'][()])

                animal_ids = list(file[f'data/{organisation_id}'].keys())

            animal_ids = list(filter(
                lambda x: x != 'organisation', animal_ids))

            alen = len(animal_ids)
            for aidx, animal_id in enumerate(animal_ids):

                astring = (' | Loading animals: '
                           f'{round((aidx + 1) / alen * 100)}% done, '
                           f'{aidx + 1}/{alen}')

                print(pstring + astring, end='')

                with self.readfile() as file:
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

                frame = pd.concat([frame, data])

        print('\nStoring data')
        frame.index.names = ['datetime']
        frame = frame.set_index(
            ['organisation_id', 'group_id', 'animal_id', frame.index])

        frame.to_hdf(self.train_store_path, key='dataset', complevel=9)

    def run(self):
        self.transform_data()


def main():
    loader = DataLoader()
    loader.run()

    transformer = DataTransformer()
    transformer.run()


if __name__ == '__main__':
    main()
