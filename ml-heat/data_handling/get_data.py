#!/usr/bin/python3
# coding: utf8

import os
import json
import datetime
import h5py as h5
from sxapi import LowLevelAPI, APIv2

PRIVATE_TOKEN = ('i-am-no-longer-a-token')

PRIVATE_ENDPOINT = 'http://127.0.0.1:8787/internapi/v1/'

PUBLIC_ENDPOINTv2 = 'https://api.smaxtec.com/api/v2/'


class DataLoader(object):
    def __init__(self):
        self.api = APIv2(
            api_key=PRIVATE_TOKEN, endpoint=PUBLIC_ENDPOINTv2,
            asynchronous=True).low

        self.oldapi = LowLevelAPI(
            api_key=PRIVATE_TOKEN,
            private_endpoint=PRIVATE_ENDPOINT).privatelow

        self.store_path = os.getcwd() + '/ml-heat/__data_store__/data.hdf5'
        self._organisation_ids = None
        self._animal_ids = None

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

        print('Loading animals')
        animals = self.api.get_animals_by_organisation_ids_async(
            filtered_orga_ids)

        # flatten list of lists
        animals = [inner for outer in animals for inner in outer]

        print('Storing animals')
        with self.writefile() as file:
            for animal in animals:
                organisation_id = animal['organisation_id']
                orga_group = file[f'data/{organisation_id}']

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

    def load_sensordata(self, organisation_ids=None, update=False):
        metrics = ['act', 'temp']
        from_dt = str(datetime.date(2014, 1, 1))
        to_dt = str(datetime.date(2019, 4, 1))

        print('Loading sensor data')
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

        with self.writefile() as file:
            

            data = self.api.get_data_by_animal_ids_async(
                animal_ids, from_dt, to_dt, metrics)

        print(data[0])

    def run(self, organisation_ids=None, update=False):
        self.load_organisations(update)
        self.load_animals(organisation_ids=organisation_ids, update=update)
        # self.load_sensordata(organisation_ids=organisation_ids, update=update)


def main():
    loader = DataLoader()
    loader.run()


if __name__ == '__main__':
    main()
