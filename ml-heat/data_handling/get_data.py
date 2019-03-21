#!/usr/bin/python3
# coding: utf8

import os
import json
import h5py as h5
from sxapi import LowLevelAPI, APIv2

PRIVATE_TOKEN = ('i-am-no-longer-a-token')

PRIVATE_ENDPOINT = 'http://127.0.0.1:8787/internapi/v1/'
# 'https://api.smaxtec.com/internapi/v0/'
PUBLIC_ENDPOINTv2 = 'https://api.smaxtec.com/api/v2/'


class DataLoader(object):
    # TODO: implement asynchronous operation
    def __init__(self):
        self.api = APIv2(
            api_key=PRIVATE_TOKEN, endpoint=PUBLIC_ENDPOINTv2).low

        self.oldapi = LowLevelAPI(
            api_key=PRIVATE_TOKEN,
            private_endpoint=PRIVATE_ENDPOINT).privatelow

        self.store_path = os.getcwd() + '/ml-heat/__data_store__/data.hdf5'

    def readfile(self):
        return h5.File(self.store_path, 'r')

    def writefile(self):
        return h5.File(self.store_path, 'a')

    def load_organisations(self, update=False):
        # TODO: switch to apiv2 once implemented

        with self.writefile() as file:
            if 'organisations' not in file.keys() or update:
                print('Loading organisations')
                organisations = self.oldapi.query_organisations()
                try:
                    orga_group = file.create_group(name='organisations')
                except ValueError:
                    orga_group = file['organisations']

                print('Storing organisations to local cache')
                for organisation in organisations:
                    if organisation['_id'] in orga_group.keys():
                        del orga_group[organisation['_id']]

                    orga_group.create_dataset(
                        name=organisation['_id'],
                        data=json.dumps(organisation))
            else:
                print('Organisations found in store, skipped loading')

    def load_animals(self, organisation_ids=None, update=False):
        if organisation_ids is None:
            with self.readfile() as file:
                organisation_ids = list(file['organisations'].keys())

        num_orgas = len(organisation_ids)
        with self.writefile() as file:
            try:
                animals_group = file.create_group('animals')
            except ValueError:
                animals_group = file['animals']

            for idx, organisation_id in enumerate(organisation_ids):
                if organisation_id in animals_group.keys() and not update:
                    print('Animals found in store, skipped loading')
                    continue

                index = idx + 1
                string = ('\rLoading animals for organisation: '
                          f'{int(index / num_orgas * 100)}% done')
                print(string, end='')

                try:
                    animals = self.api.get_animals_by_organisation_id(
                        organisation_id)

                except Exception as e:
                    print('\n' + str(e))
                    continue

                try:
                    a_subgroup = animals_group.create_group(
                        name=organisation_id)
                except ValueError:
                    a_subgroup = animals_group[organisation_id]

                for animal in animals:
                    if animal['_id'] in a_subgroup.keys():
                        del a_subgroup[animal['_id']]

                    a_subgroup.create_dataset(
                        name=animal['_id'],
                        data=json.dumps(animal))

        print('\n')

    def run(self, update=False):
        # self.load_organisations(update)
        self.load_animals(update=update)


def main():
    loader = DataLoader()
    loader.run()


if __name__ == '__main__':
    main()
