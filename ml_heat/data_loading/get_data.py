#!/usr/bin/python3
# coding: utf8

import os
import json
import pickle
import datetime
import h5py as h5
import pandas as pd
from tqdm import tqdm
from sxapi import LowLevelAPI, APIv2
from anthilldb.settings import LiveConfig
from anthilldb.client import DirectDBClient
from anthilldb.commands.downloader import download_key

from concurrent.futures import (
    ThreadPoolExecutor,
    ProcessPoolExecutor,
    as_completed
)

PRIVATE_TOKEN = ('i-am-no-longer-a-token')

PRIVATE_ENDPOINT = 'http://127.0.0.1:8787/internapi/v1/'

PUBLIC_ENDPOINTv2 = 'https://api-staging.smaxtec.com/api/v2/'

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.abspath(
    os.path.join(os.getcwd(), 'key.json'))


# this is defined here because it is used in a process pool -> needs to be
# pickleable and only functions that are defined on the top level can be
# serialized
def parse_csv(filepath, timezone):
    try:
        frame = pd.read_csv(
            filepath, index_col=0, parse_dates=[0],
            date_parser=lambda col: pd.to_datetime(col, utc=True))
    except pd.errors.EmptyDataError:
        os.remove(filepath)
    except Exception as e:
        print(e)

    frame = frame.tz_convert(timezone)

    os.remove(filepath)

    if frame.empty:
        return

    writepath = filepath.replace('.csv', '')
    with open(writepath, 'wb') as file:
        pickle.dump(frame, file)


class DataLoader(object):
    def __init__(self):
        self.api = APIv2(
            api_key=PRIVATE_TOKEN, endpoint=PUBLIC_ENDPOINTv2,
            asynchronous=True).low

        self.oldapi = LowLevelAPI(
            api_key=PRIVATE_TOKEN,
            private_endpoint=PRIVATE_ENDPOINT).privatelow

        self.dbclient = DirectDBClient(
            project_id=LiveConfig.GCP_PROJECT_ID,
            instance_id=LiveConfig.GCP_INSTANCE_ID,
            credentials=None,
            table_prefix=LiveConfig.TABLE_PREFIX,
            metric_definition=LiveConfig.METRICS,
            pool_size=30)

        self.thread_pool = ThreadPoolExecutor(30)
        self.process_pool = ProcessPoolExecutor(os.cpu_count())

        self.store_path = os.path.join(os.getcwd(), 'ml_heat/__data_store__')
        self.rawdata_path = os.path.join(self.store_path, 'rawdata.hdf5')
        self._organisation_ids = None
        self._animal_ids = None
        self._animal_orga_map = None

    def readfile(self):
        return h5.File(self.rawdata_path, 'r')

    def writefile(self):
        return h5.File(self.rawdata_path, 'a')

    def get_data(self, animal_id, from_dt, to_dt, metrics):
        return self.thread_pool.submit(
            self.dbclient.get_metrics, animal_id, metrics, from_dt, to_dt)

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
        # TODO: switch to apiv2 or anthilldb once implemented

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

    def load_sensordata_from_db(self,
                                organisation_ids=None,
                                update=False,
                                metrics=['act', 'temp']):

        print('Preparing to load sensor data...')
        from_dt = datetime.datetime(2018, 4, 1)
        to_dt = datetime.datetime(2019, 4, 1)

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

        temp_path = os.path.join(self.store_path, 'temp')
        os.mkdir(temp_path)

        print('Loading sensordata...')

        results = [self.thread_pool.submit(
            download_key, self.dbclient, _id, metrics, from_dt, to_dt,
            temp_path) for _id in animal_ids]

        kwargs = {
            'total': len(results),
            'unit': 'files',
            'unit_scale': True,
            'leave': True
        }
        for f in tqdm(as_completed(results), **kwargs):
            pass

        print('Download finished...')

    def csv_to_pickle(self):
        print('Processing data into storage format...')

        temp_path = os.path.join(self.store_path, 'temp')
        files = [s for s in os.listdir(temp_path) if s.endswith('.csv')]
        filepaths = [os.path.join(temp_path, x) for x in files]

        futures = []
        for filepath in filepaths:
            animal_id = filepath.split('/')[-1].split('.csv')[0]
            organisation_id = self.organisation_id_for_animal_id(animal_id)
            with self.readfile() as file:
                organisation = json.loads(
                    file[f'data/{organisation_id}/organisation'][()])
            timezone = organisation['timezone']

            futures.append(
                self.process_pool.submit(
                    parse_csv, filepath, timezone))

        kwargs = {
            'total': len(futures),
            'unit': 'files',
            'unit_scale': True,
            'leave': True
        }

        for f in tqdm(as_completed(futures), **kwargs):
            pass

        print('Processing finished...')

    def pickle_to_hdf(self):
        print('Writing data to hdf file...')

        temp_path = os.path.join(self.store_path, 'temp')
        files = [s for s in os.listdir(temp_path) if not s.endswith('.csv')]
        filepaths = [os.path.join(temp_path, p) for p in files]

        for filepath in tqdm(filepaths):
            animal_id = filepath.split('/')[-1]
            organisation_id = self.organisation_id_for_animal_id(animal_id)

            with open(filepath, 'rb') as file:
                try:
                    frame = pickle.load(file)
                except EOFError:
                    os.remove(filepath)
                    continue

            if frame.empty:
                os.remove(filepath)
                continue

            frame.to_hdf(
                self.rawdata_path,
                key=f'data/{organisation_id}/{animal_id}/sensordata',
                complevel=9)

            os.remove(filepath)

        print('Finished saving rawdata...')
        print('Cleaning up...')
        os.rmdir(temp_path)
        print('Done!')

    def run(self, organisation_ids=None, update=False):
        self.load_organisations(update)
        self.load_animals(organisation_ids=organisation_ids, update=update)
        self.load_sensordata_from_db(
            organisation_ids=organisation_ids, update=update)
        self.csv_to_pickle()
        self.pickle_to_hdf()


def main():
    loader = DataLoader()
    loader.run()


if __name__ == '__main__':
    main()
