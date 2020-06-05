#!/usr/bin/python3
# coding: utf8

import os
import json
import tables
import datetime
import h5py as h5
import pandas as pd
from pickle import UnpicklingError
from tqdm import tqdm
from sxapi import LowLevelAPI, APIv2
from sxapi.low import PrivateAPIv2
from anthilldb.client import DirectDBClient
from anthilldb.settings import get_config_by_name
from tables.exceptions import HDF5ExtError

from concurrent.futures import (
    ThreadPoolExecutor,
    ProcessPoolExecutor,
    as_completed
)

with open(os.path.abspath(os.path.join(os.getcwd(), 'token.json'))) as file:
    doc = json.load(file)
    live_token_string = doc['live']
    staging_token_string = doc['staging']

LIVECONFIG = get_config_by_name('live')

PRIVATE_STAGING_TOKEN = staging_token_string

PRIVATE_LIVE_TOKEN = live_token_string

PRIVATE_ENDPOINT = 'http://127.0.0.1:8787/internapi/v1/'

PUBLIC_ENDPOINTv2 = 'https://api.smaxtec.com/api/v2/'

PRIVATE_ENDPOINTv2 = 'http://127.0.0.1:8787/internapi/v2/'

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.abspath(
    os.path.join(os.getcwd(), 'key.json'))


class DataLoader(object):
    def __init__(self):
        self.api = APIv2(
            api_key=PRIVATE_LIVE_TOKEN, endpoint=PUBLIC_ENDPOINTv2,
            asynchronous=True).low

        self.privateapi = PrivateAPIv2(
            api_key=PRIVATE_LIVE_TOKEN, endpoint=PRIVATE_ENDPOINTv2)

        self.oldapi = LowLevelAPI(
            api_key=PRIVATE_LIVE_TOKEN,
            private_endpoint=PRIVATE_ENDPOINT).privatelow

        self.dbclient = DirectDBClient(
            engine='bigtable',
            engine_options=LIVECONFIG.ENGINE_OPTIONS,
            table_prefix=LIVECONFIG.TABLE_PREFIX)

        self.thread_pool = ThreadPoolExecutor(30)
        self.process_pool = ProcessPoolExecutor(os.cpu_count())

        self.store_path = os.path.join(
            os.getcwd(), 'ml_heat', '__data_store__')

        if not os.path.exists(self.store_path):
            os.mkdir(self.store_path)

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
                    # anonymization
                    organisation = self.sanitize_organisation(organisation)
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

        filtered_orga_ids = None
        if not update:
            with self.readfile() as file:
                keys = file['data'].keys()

                keys = [x for x in keys if len(file[f'data/{x}'].keys()) > 1]

            filtered_orga_ids = [x for x in organisation_ids if x not in keys]

        if not filtered_orga_ids or filtered_orga_ids is None:
            if not update:
                return
            else:
                filtered_orga_ids = organisation_ids

        futures = [self.thread_pool.submit(
                   self.api.get_animals_by_organisation_id,
                   organisation_id)
                   for organisation_id in filtered_orga_ids]

        kwargs = {
            'total': len(futures),
            'unit': 'organisations',
            'unit_scale': True,
            'leave': True,
            'smoothing': 0.001,
            'desc': 'Loading animals'
        }

        for f in tqdm(as_completed(futures), **kwargs):
            pass

        animals = [x for future in futures for x in future.result()]

        kwargs = {
            'desc': 'Loading additional events for animals',
            'unit': 'animals',
            'smoothing': 0.001
        }

        with self.writefile() as file:
            if self._animal_orga_map is None:
                try:
                    self._animal_orga_map = json.loads(
                        file['lookup/animal_to_orga'][()])
                except Exception:
                    self._animal_orga_map = {}

            kwargs['desc'] = 'Storing animals'

            for animal in tqdm(animals, **kwargs):
                organisation_id = animal['organisation_id']
                orga_group = file[f'data/{organisation_id}']

                if animal['_id'] in orga_group.keys():
                    a_subgroup = file[
                        f'data/{organisation_id}/{animal["_id"]}']

                    if 'animal' in a_subgroup.keys():
                        if 'events' in a_subgroup.keys():
                            continue

                events = self.privateapi.get_events_by_animal_id(
                    animal['_id'], True)

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

                if 'events' in a_subgroup.keys():
                    del a_subgroup['events']

                a_subgroup.create_dataset(
                    name='events',
                    data=json.dumps(events))

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
        to_dt = datetime.datetime(2020, 4, 1)

        self.dbclient.service_init()

        if organisation_ids is None:
            organisation_ids = self.organisation_ids

        # retrieve animal ids to load data for
        animal_ids = self.animal_ids_for_organisations(organisation_ids)

        temp_path = os.path.join(self.store_path, 'temp')

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
                        if len(a_keys) < 3:
                            filtered.append(animal_id)
            animal_ids = filtered

            # check if csv or pickle file exists already
            if os.path.exists(temp_path):
                files = os.listdir(temp_path)
                keys = list(set([string.split('|')[0] for string in files]))
                animal_ids = [
                    animal_id for animal_id in animal_ids
                    if animal_id not in keys]

        if not os.path.exists(temp_path):
            os.mkdir(temp_path)

        # print('Loading sensordata...')

        kwargs = {
            'desc': 'Loading sensordata',
            'unit': 'animals',
            'smoothing': 0.001
        }

        for _id in tqdm(animal_ids, **kwargs):
            self.download_metrics(
                self.dbclient, _id, metrics, from_dt, to_dt, temp_path)

        print('Download finished...')

    def download_metrics(
            self, db_client, key, metrics, from_dt, to_dt, output_file_path):

        chunks = []
        while (to_dt - from_dt).total_seconds() >= 399 * 24 * 60 * 60:
            chunk = (from_dt, from_dt + datetime.timedelta(days=399))
            chunks.append(chunk)
            from_dt += datetime.timedelta(days=398)
        chunks.append((from_dt, to_dt))

        for idx, chunk in enumerate(chunks):
            all_timeseries = db_client.get_multi_metrics(
                key, metrics, chunk[0], chunk[1])
            file_name = os.path.realpath(
                os.path.join(output_file_path, f"{key}|{idx}.csv"))
            with open(file_name, "w") as fp:
                all_timeseries.to_csv(fp)
                fp.flush()
        return key

    def csv_to_hdf(self):
        print('Writing data to hdf file...')

        temp_path = os.path.join(self.store_path, 'temp')
        files = [s for s in os.listdir(temp_path) if s.endswith('.csv')]
        filepaths = [os.path.join(temp_path, p) for p in files]
        animal_ids = list(set([s.split('|')[0] for s in files]))

        iterdict = {}
        for animal_id in tqdm(animal_ids, desc='Parsing filepaths'):
            iterdict[animal_id] = [
                path for path in filepaths if path.split(
                    os.sep)[-1].startswith(animal_id)]

        for key in tqdm(iterdict.keys()):
            organisation_id = self.organisation_id_for_animal_id(key)

            framelist = []
            for filepath in iterdict[key]:
                framelist.append(pd.read_csv(filepath, index_col='ts'))

            frame = pd.concat(framelist).sort_index()
            frame = frame.loc[~frame.index.duplicated(keep='first')]

            frame.index = pd.to_datetime(
                frame.index, unit='s').rename('datetime')

            if frame.empty:
                for filepath in iterdict[key]:
                    os.remove(filepath)
                continue

            frame.to_hdf(
                self.rawdata_path,
                key=f'data/{organisation_id}/{key}/sensordata',
                complevel=9)

            for filepath in iterdict[key]:
                os.remove(filepath)

        print('Finished saving rawdata...')
        print('Cleaning up...')
        os.rmdir(temp_path)
        print('Done!')

    def animal_count_per_orga(self):
        organisation_ids = self.organisation_ids

        data = {}
        for organisation_id in organisation_ids:
            animal_ids = self.animal_ids_for_organisations(organisation_id)
            data[organisation_id] = len(animal_ids)

        return pd.DataFrame(data)

    def sanitize_organisation(self, organisation):
        organisation.pop('name', None)
        organisation.pop('account', None)
        return organisation

    def fix_dt_index(self):
        # collect organisations & animals
        iterdict = {}

        kwargs = {
            'desc': 'Discovering data',
            'smoothing': 0.01
        }

        with self.writefile() as file:
            organisation_ids = file['data'].keys()

            for organisation_id in tqdm(organisation_ids, **kwargs):
                iterdict[organisation_id] = []
                animal_ids = [
                    key for key in file[f'data/{organisation_id}'].keys()
                    if key != 'organisation'
                ]

                for animal_id in animal_ids:
                    if 'sensordata' in file[
                            f'data/{organisation_id}/{animal_id}'].keys():

                        iterdict[organisation_id].append(animal_id)

        kwargs['desc'] = 'Converting index datatype'

        for organisation_id, animal_ids in tqdm(iterdict.items(), **kwargs):
            kwargs['desc'] = 'Processing animals'
            kwargs['leave'] = False
            kwargs['position'] = 1
            for animal_id in tqdm(animal_ids, **kwargs):
                try:
                    frame = pd.read_hdf(
                        self.rawdata_path,
                        key=f'data/{organisation_id}/{animal_id}/sensordata'
                    )
                except HDF5ExtError as e:
                    print(e)
                    print(f'data read failed on animal {animal_id}, '
                          f'organisation {organisation_id}')
                    tables.file._open_files.close_all()

                    continue

                except UnpicklingError as e:
                    print(e)
                    print(f'data read failed on animal {animal_id}, '
                          f'organisation {organisation_id}')
                    tables.file._open_files.close_all()

                    continue

                if isinstance(
                        frame.index, pd.core.indexes.datetimes.DatetimeIndex):
                    continue

                frame.index = pd.to_datetime(
                    [int(dt.timestamp()) for dt in frame.index],
                    unit='s'
                )

                frame.to_hdf(
                    self.rawdata_path,
                    key=f'data/{organisation_id}/{animal_id}/sensordata',
                    complevel=9
                )

    def del_sensordata(self):
        """
        In rare cases, pandas can produce broken datasets when writing to
        hdf5, this function can be used to delete them so they can be either
        downloaded again or discarded

        USE WITH UTTERMOST CARE
        """

        organisation_id = '5af01e0210bac288dba249ad'
        animal_id = '5b6419ff36b96c52808951b1'

        with self.writefile() as file:
            del file[f'data/{organisation_id}/{animal_id}/sensordata']

    def run(self, organisation_ids=None, update=False):
        self.load_organisations(update)
        self.load_animals(organisation_ids=organisation_ids, update=update)
        self.load_sensordata_from_db(
            organisation_ids=organisation_ids, update=update)
        self.csv_to_hdf()


def main():
    loader = DataLoader()
    loader.run()


if __name__ == '__main__':
    main()
