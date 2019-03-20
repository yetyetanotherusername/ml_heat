#!/usr/bin/python3
# coding: utf8

from sxapi import LowLevelAPI, APIv2
import h5py as h5
import os
import json

PRIVATE_TOKEN = ('i-am-no-longer-a-token')

PRIVATE_ENDPOINT = 'http://127.0.0.1:8787/internapi/v1/'
# 'https://api.smaxtec.com/internapi/v0/'
PUBLIC_ENDPOINTv2 = 'https://api.smaxtec.com/api/v2/'


class DataLoader(object):
    def __init__(self):
        self.api = APIv2(
            api_key=PRIVATE_TOKEN, endpoint=PUBLIC_ENDPOINTv2).low

        self.oldapi = LowLevelAPI(
            api_key=PRIVATE_TOKEN,
            private_endpoint=PRIVATE_ENDPOINT).privatelow

        self.store_path = os.getcwd() + '/ml-heat/__data_store__/data.hdf5'

    def load_organisations(self, update=False):
        # TODO: switch to apiv2 once implemented

        with h5.File(self.store_path, 'r') as file:
            if 'organisations' not in file.keys():
                update = True

        if update:
            with h5.File(self.store_path, 'w') as file:
                organisations = self.oldapi.query_organisations()
                payload = json.dumps(organisations)
                file.create_dataset(name='organisations', data=payload)
        else:
            print('Organisations found in store, skipped loading')

    def load_animals(self):

        with h5.File(self.store_path, 'r') as file:
            print(file['organisations'])

        # self.animals = []
        # for organisation_id in self.organisation_ids:
        #     self.animals.append(
        #         self.api.get_animals_by_organisation_id(organisation_id))

    def run(self, update=False):
        self.load_organisations(update)
        self.load_animals()


def main():
    loader = DataLoader()
    loader.run()


if __name__ == '__main__':
    main()
