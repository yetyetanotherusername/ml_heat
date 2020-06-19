#!/usr/bin/python3
# coding: utf8

import os
import json
import joblib
import zarr
import h5py as h5
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing
import dask as dd

from dask.diagnostics import ProgressBar
from dask_ml.model_selection import train_test_split

from concurrent.futures import (
    ProcessPoolExecutor,
    as_completed,
)

from sklearn.preprocessing import StandardScaler

from sxutils.models.animal.cycle import AnimalLifecycle
from sxutils.munch.cyclemunch import cycle_munchify
from ml_heat.preprocessing.animal_serde import AnimalSchemaV2

from ml_heat.helper import (
    plot_setup,
    load_animal,
    store_animal,
    duplicate_shift
)


def fnn_worker(feather_store,
               z_arr,
               animal_id,
               shifts=range(288),
               fix_imbalance=False):

    data = load_animal(feather_store, animal_id)

    data['annotation'] = np.logical_or(
        np.logical_or(data.pregnant, data.cyclic), data.inseminated)

    data = data[['annotation', 'DIM', 'heat_feature']].dropna()

    data.DIM = pd.to_numeric(data.DIM, downcast='float')
    data.annotation = pd.to_numeric(
        data.annotation.astype(int), downcast='float')
    data.heat_feature = pd.to_numeric(data.heat_feature, downcast='float')

    heat_shift = duplicate_shift(data.heat_feature, shifts, 'heat_feature')
    data = pd.concat([data, heat_shift], axis=1).dropna()
    data = data.drop(['heat_feature'], axis=1)

    if fix_imbalance:
        no_heat_idx = data[data.annotation == 0].index
        no_heat_len = len(no_heat_idx)
        heat_len = data[data.annotation == 1].shape[0]
        overhang = no_heat_len - heat_len
        to_drop = np.random.choice(no_heat_idx, overhang, replace=False)
        data = data[~data.index.isin(to_drop)]

    z_arr.append(data.values)
    os.remove(os.path.join(feather_store, animal_id))
    return animal_id


def scale_worker(animal_id, columns, scaler, store):
    frame = load_animal(store, animal_id)
    frame[columns] = scaler.transform(frame[columns])

    for column in columns:
        frame[column] = pd.to_numeric(frame[column], downcast='float')

    store_animal(frame, store, animal_id)
    return animal_id


def add_heat_feature(inframe):
    inframe['heat_feature'] = (
        inframe.act - inframe.act_group_mean).rolling(12, min_periods=1).sum()
    return inframe


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

    frame.dims = pd.to_numeric(frame.dims, downcast='float')

    return frame.dims


def add_cyclic(inframe, animal):
    # TODO: enforce holdoff?
    lifecycle = animal['lifecycle']
    min_timedelta = pd.Timedelta(18, unit='days')
    max_timedelta = pd.Timedelta(24, unit='days')

    cyclic_rows = pd.DataFrame()
    for cycle in lifecycle['cycles']:
        frame = pd.DataFrame(cycle['events'])
        if frame.empty:
            continue

        # filter for relevant event types
        frame = frame[frame.event_type.isin(['heat', 'insemination'])]

        rows = len(frame.index)
        columns = [str(x) for x in range(-rows, rows + 1) if x != 0]
        frame.event_ts = pd.to_datetime(
            frame.event_ts, format=('%Y-%m-%dT%H:%M:%S'))

        for shift in columns:
            frame[shift] = frame.event_ts.diff(shift).abs().between(
                min_timedelta, max_timedelta)

        frame['cyclic'] = frame[columns].any(axis=1)
        frame.drop(columns, inplace=True, axis=1)

        cyclic_rows = cyclic_rows.append(
            frame[frame.cyclic == True], sort=True)  # noqa

    if cyclic_rows.empty:
        return inframe

    inframe.sort_index(inplace=True)

    cyclic_dts = cyclic_rows.event_ts.to_list()

    # find indices closest to cyclic heats and set cyclic column to true
    # if no matching index is found inside tolerance of 24h, disregard
    span = pd.Timedelta(hours=12)
    start_span = pd.Timedelta(hours=2)
    end_span = pd.Timedelta(hours=8)
    for dt in cyclic_dts:
        idx = None
        from_dt = dt - span
        to_dt = dt + span
        try:
            idx = inframe.loc[
                from_dt:to_dt, 'act'].rolling(6, min_periods=1).mean().idxmax()
        except ValueError:
            continue

        start = idx - start_span
        end = idx + end_span
        inframe.loc[start:end, 'cyclic'] = True
    return inframe


def add_inseminated(inframe, animal):
    lifecycle = animal['lifecycle']

    insemination_rows = pd.DataFrame()
    for cycle in lifecycle['cycles']:
        frame = pd.DataFrame(cycle['events'])
        if frame.empty:
            continue

        # filter for relevant event types
        frame = frame[frame.event_type == 'insemination']

        if frame.empty:
            continue

        frame.event_ts = pd.to_datetime(
            frame.event_ts, format=('%Y-%m-%dT%H:%M:%S'))

        insemination_rows = insemination_rows.append(frame, sort=True)

    if insemination_rows.empty:
        return inframe

    inframe.sort_index(inplace=True)

    insemination_dts = insemination_rows.event_ts.to_list()

    # find indices closest to inseminated heats and set inseminated column to
    # true if no matching index is found inside tolerance of 24h, disregard
    span = pd.Timedelta(hours=12)
    start_span = pd.Timedelta(hours=2)
    end_span = pd.Timedelta(hours=8)
    for dt in insemination_dts:
        idx = None
        from_dt = dt - span
        to_dt = dt + span
        try:
            idx = inframe.loc[
                from_dt:to_dt, 'act'].rolling(6, min_periods=1).mean().idxmax()
        except ValueError:
            continue

        start = idx - start_span
        end = idx + end_span
        inframe.loc[start:end, 'inseminated'] = True
    return inframe


def add_pregnant(inframe, animal):
    # set the pregnant flag when the user confirms an insemination was
    # successful or when there was a calving 270-290 days after an event

    lifecycle = animal['lifecycle']

    pregnant_ts = []
    for cycle in lifecycle['cycles']:

        last_calving_date = cycle.get('last_calving_date', None)
        last_calving_date = pd.to_datetime(
            last_calving_date, format=('%Y-%m-%dT%H:%M:%S'))

        frame = pd.DataFrame(cycle['events'])
        if frame.empty:
            continue

        frame.event_ts = pd.to_datetime(
            frame.event_ts, format=('%Y-%m-%dT%H:%M:%S'))

        frame.insemination_date = pd.to_datetime(
            frame.insemination_date, format=('%Y-%m-%dT%H:%M:%S'))

        calving_confirmations = frame[
            frame.event_type == 'calving_confirmation']

        confirmed_inseminations = frame[
            frame.event_type == 'pregnancy_result']

        heats = frame[frame.event_type.isin(['heat', 'insemination'])]

        min_td = pd.Timedelta(days=270)
        max_td = pd.Timedelta(days=290)

        pr_heats = []
        calving_dt_list = (
            calving_confirmations.event_ts.to_list() + [last_calving_date])

        calving_dt_list = [
            calving_dt for calving_dt in calving_dt_list if
            calving_dt is not None]

        for calving_dt in calving_dt_list:
            pr_heats += [
                x for x in heats.event_ts.to_list() if
                min_td <= calving_dt - x <= max_td]

        pregnant_ts += pr_heats
        pregnant_ts += confirmed_inseminations.insemination_date.to_list()

    if not pregnant_ts:
        return inframe

    inframe.sort_index(inplace=True)

    pregnant_ts = sorted(list(set(pregnant_ts)))

    # find indices closest to pregnant heats and set pregnant column to true
    # if no matching index is found inside tolerance of 24h, disregard
    span = pd.Timedelta(hours=12)
    start_span = pd.Timedelta(hours=2)
    end_span = pd.Timedelta(hours=8)
    for dt in pregnant_ts:
        idx = None
        from_dt = dt - span
        to_dt = dt + span
        try:
            idx = inframe.loc[
                from_dt:to_dt, 'act'].rolling(6, min_periods=1).mean().idxmax()
        except ValueError:
            continue

        start = idx - start_span
        end = idx + end_span
        inframe.loc[start:end, 'pregnant'] = True
    return inframe


def add_deleted(inframe, animal, events):
    inframe = inframe.sort_index()
    frame = pd.DataFrame(events)
    if frame.empty:
        return inframe
    frame = frame[frame.event_type == 'actincrease_704']
    detected_heats = pd.to_datetime(
        frame.event_ts, format=('%Y-%m-%dT%H:%M:%S')).to_list()

    lifecycle = animal['lifecycle']

    for cycle in lifecycle['cycles']:
        frame = pd.DataFrame(cycle['events'])

        if frame.empty:
            continue

        frame = frame[frame.event_type == 'heat']
        undeleted_heats = pd.to_datetime(
            frame.event_ts, format=('%Y-%m-%dT%H:%M:%S')).to_list()

        for undeleted in undeleted_heats:
            if undeleted in detected_heats:
                detected_heats.remove(undeleted)

    span = pd.Timedelta(hours=12)
    start_span = pd.Timedelta(hours=2)
    end_span = pd.Timedelta(hours=8)
    for dt in detected_heats:
        idx = None
        from_dt = dt - span
        to_dt = dt + span
        try:
            idx = inframe.loc[
                from_dt:to_dt, 'act'].rolling(6, min_periods=1).mean().idxmax()
        except ValueError:
            continue

        start = idx - start_span
        end = idx + end_span
        inframe.loc[start:end, 'deleted'] = True
    return inframe


def add_labels(inframe, animal, events):
    # add label columns to frame
    inframe['cyclic'] = False
    inframe['inseminated'] = False
    inframe['pregnant'] = False
    inframe['deleted'] = False

    # cyclic heat analysis
    inframe = add_cyclic(inframe, animal)

    # add inseminated heats
    inframe = add_inseminated(inframe, animal)

    # add heats that resulted in a pregnancy
    inframe = add_pregnant(inframe, animal)

    # add heats deleted by the user
    inframe = add_deleted(inframe, animal, events)

    return inframe


def add_features(inframe, organisation, animal):
    # shorten string fields (hdf5 serde has limits on string length)
    # race = animal.get('race', 'N/A')
    # if race is None:
    #     race = 'N/A'

    # if len(race) > 10:
    #     race = [word[0] + '_' for word in animal['race'].split('_')]
    #     race = ''.join(race)

    # partner_id = organisation.get('partner_id', 'N/A')
    # if partner_id is None:
    #     partner_id = 'N/A'

    # if len(partner_id) > 10:
    #     partner_id = partner_id[:10]

    group_id = animal.get('group_id', 'N/A')
    if group_id is None:
        group_id = 'N/A'

    # country field may be unavailable
    # country = animal.get('metadata', {}).get('country', 'N/A')

    inframe['organisation_id'] = organisation['_id']
    inframe['group_id'] = group_id
    inframe['animal_id'] = animal['_id']
    # inframe['race'] = race
    # inframe['country'] = country
    # inframe['partner_id'] = partner_id
    inframe['DIM'] = calculate_dims(inframe.index, animal)
    inframe['temp_filtered'] = calc_temp_without_drink_spikes(inframe)

    # inframe.race = inframe.race.astype('category')
    # inframe.country = inframe.country.astype('category')
    # inframe.partner_id = inframe.partner_id.astype('category')

    return inframe


def add_group_feature(inframe):
    frame = inframe['act'].drop(
        index='N/A', level=0, errors='ignore').copy(deep=True)

    if frame.empty:
        inframe.loc[:, 'act_group_mean'] = float('nan')
        return inframe

    # make animal id a column index
    frame = frame.unstack('animal_id').sort_index()

    # include only if there are enough values available for mean
    group_mean = frame[frame.count(axis=1) >= 5].mean(axis=1)

    # create multilevel column index and add group mean to each animal
    frame.columns = [frame.columns, ['act'] * len(frame.columns)]
    frame = frame.stack(level=0, dropna=False)
    frame['act_group_mean'] = float('nan')
    frame = frame.unstack('animal_id')
    frame = frame.swaplevel(axis=1).sort_index().sort_index(axis=1)
    frame.loc[(slice(None), slice(None)),
              (slice(None), 'act_group_mean')] = group_mean
    del group_mean

    # recreate original dataframe with all animals in one column
    frame = frame.stack('animal_id', dropna=False)
    frame = frame.swaplevel().dropna(subset=['act']).sort_index()
    inframe['act_group_mean'] = frame.act_group_mean

    return inframe


def calc_temp_without_drink_spikes(inframe):
    frame = pd.DataFrame()
    frame['temp'] = inframe.temp
    frame['temp_median'] = frame.temp.rolling(
        288, min_periods=1, center=True).median()

    frame['temp_var'] = frame.temp.rolling(288, min_periods=1).var()
    frame['temp_mean'] = frame.temp.rolling(
        72, min_periods=1, center=True).mean()

    frame['lower_bound'] = frame.temp_median - frame.temp_var * 0.55
    frame['temp_filtered'] = frame.temp
    frame.loc[
        frame.temp < frame.lower_bound,
        'temp_filtered'
    ] = frame.temp_mean

    frame['filler'] = frame.temp_filtered.rolling(
        36, min_periods=1, center=True).mean()

    frame.loc[
        frame.temp < frame.lower_bound,
        'temp_filtered'
    ] = frame.filler

    return frame['temp_filtered']


def transform_animal(organisation, animal_id, readpath, readfile):
    organisation_id = organisation['_id']
    try:
        data = pd.read_hdf(
            readpath,
            key=f'data/{organisation_id}/{animal_id}/sensordata')
    except KeyError:
        return None

    if data.empty:
        return None

    animal = json.loads(
        readfile[f'data/{organisation_id}/{animal_id}/animal'][()])

    if animal.get('group_id') is None:
        return None

    events = json.loads(
        readfile[f'data/{organisation_id}/{animal_id}/events'][()])

    data.act = pd.to_numeric(data.act, downcast='float')
    data.temp = pd.to_numeric(data.temp, downcast='float')

    # remove localization -> index is localtime without tzinfo
    # needed so we can have all animal indices in one column
    data = data.tz_localize(None)

    # normalize all timestamps to 10minute base
    data.index = data.index.round('10T')

    # drop duplicates in index resulting from DST switch
    data = data[~data.index.duplicated(keep='first')]

    # calculate all features and add them as columns
    data = add_features(data, organisation, animal)

    # calculate labels and add them as columns
    data = add_labels(data, animal, events)

    return data


def transform_organisation(organisation_id, readpath, temp_path):
    with h5.File(readpath, 'r') as readfile:

        organisation = json.loads(
            readfile[f'data/{organisation_id}/organisation'][()])

        animal_ids = list(readfile[f'data/{organisation_id}'].keys())
        animal_ids = list(filter(
            lambda x: x != 'organisation', animal_ids))

        # out of ram constraints, we cannot process such big organisations
        # if len(animal_ids) > 1500:
        #     return organisation_id

        framelist = []
        position = multiprocessing.current_process()._identity[0] + 1
        if position > os.cpu_count():
            position = 1
        for animal_id in tqdm(
                animal_ids,
                leave=False,
                desc=f'Thread {position - 1} animal loop',
                position=position):

            frame = transform_animal(
                organisation, animal_id, readpath, readfile)

            if frame is None:
                continue
            framelist.append(frame)

    if not framelist:
        return organisation_id

    frame = pd.concat(framelist, sort=False)
    frame.index.names = ['datetime']

    frame = frame.drop('organisation_id', axis=1).set_index(
        ['group_id', 'animal_id', frame.index])

    frame = frame.sort_index()

    groups = frame.index.unique(level='group_id')
    reslist = []
    for group in tqdm(
            groups,
            leave=False,
            desc=f'Thread {position - 1} group loop',
            position=position):

        subframe = frame.loc[
            (group, slice(None), slice(None)), slice(None)]
        reslist.append(add_group_feature(subframe))

    del frame
    frame = pd.concat(reslist, sort=False)
    frame = frame.dropna(subset=['DIM', 'act', 'act_group_mean'])
    frame['organisation_id'] = organisation_id
    frame = frame.set_index(['organisation_id', frame.index])
    frame = frame.sort_index()

    animal_ids = frame.index.unique(level='animal_id')
    for animal_id in tqdm(
            animal_ids,
            leave=False,
            desc=f'Thread {position - 1} storage loop',
            position=position):

        animal = frame.loc[
            (slice(None), slice(None), animal_id, slice(None)), slice(None)]

        animal = animal[animal.temp > 30]

        if animal.shape[0] < 30 * 144:
            continue

        has_heat = np.any(
            np.logical_or(
                np.logical_or(animal.pregnant, animal.cyclic),
                animal.inseminated)
        )

        if has_heat is False:
            continue

        animal = add_heat_feature(animal)

        store_animal(animal, temp_path, animal_id)

    return organisation_id


class DataTransformer(object):
    def __init__(self, organisation_ids=None, update=False):
        self._organisation_ids = organisation_ids

        self.store_path = os.path.join(
            os.getcwd(), 'ml_heat', '__data_store__')
        if not os.path.exists(self.store_path):
            os.mkdir(self.store_path)

        self.update = update

        self.feather_store = os.path.join(self.store_path, 'feather_store')
        if not os.path.exists(self.feather_store):
            os.mkdir(self.feather_store)

        zarr_folder = os.path.join(self.store_path, 'zarr_store')
        if not os.path.exists(zarr_folder):
            os.mkdir(zarr_folder)
        self.zarr_store = os.path.join(zarr_folder, 'store.zarr')

        self.raw_store_path = os.path.join(self.store_path, 'rawdata.hdf5')
        self.model_store = os.path.join(self.store_path, 'models')
        if not os.path.exists(self.model_store):
            os.mkdir(self.model_store)

        self._animal_orga_map = None

    def readfile(self, how='r'):
        return h5.File(self.raw_store_path, how)

    @property
    def organisation_ids(self):
        if self._organisation_ids is None:
            with self.readfile() as file:
                self._organisation_ids = list(file['data'].keys())
        return self._organisation_ids

    def animal_ids_for_organisations(self, organisation_ids):
        animal_ids = []
        with self.readfile() as file:
            for organisation_id in organisation_ids:
                ids = list(file[f'data/{organisation_id}'].keys())
                filtered = [x for x in ids if x != 'organisation']
                animal_ids += filtered
        return animal_ids

    def animal_count_per_orga(self, organisation_ids=None):
        if organisation_ids is None:
            organisation_ids = self.organisation_ids

        data = {}
        for organisation_id in organisation_ids:
            animal_ids = self.animal_ids_for_organisations([organisation_id])
            data[organisation_id] = len(animal_ids)
        return pd.Series(data)

    def arrange_data(self):
        print('Transforming data...')

        temp_path = self.feather_store
        if not os.path.exists(temp_path):
            os.mkdir(temp_path)
            filtered_orga_ids = self.organisation_ids
        else:
            files = os.listdir(temp_path)
            loaded_orgas = list(
                set([self.organisation_id_for_animal_id(file)
                     for file in files])
            )
            filtered_orga_ids = [
                x for x in self.organisation_ids if x not in loaded_orgas]

        if len(filtered_orga_ids) > 1:
            orga_sizes = self.animal_count_per_orga(filtered_orga_ids)
            filtered_orga_ids = orga_sizes.sort_values(
                ascending=False).index.to_list()

        # for organisation_id in tqdm(
        #         filtered_orga_ids, desc='Total progress'):
        #     transform_organisation(
        #         organisation_id,
        #         self.raw_store_path,
        #         temp_path)
        with ProcessPoolExecutor(os.cpu_count()) as process_pool:
            results = [
                process_pool.submit(
                    transform_organisation,
                    organisation_id,
                    self.raw_store_path,
                    temp_path
                ) for organisation_id in filtered_orga_ids
            ]

            kwargs = {
                'total': len(filtered_orga_ids),
                'unit': 'organisation',
                'unit_scale': True,
                'leave': True,
                'desc': 'Total progress',
                'position': 1,
                'smoothing': 0.01
            }

            for f in tqdm(as_completed(results), **kwargs):
                try:
                    f.result()
                except Exception as e:
                    print(e)

        os.system('tput ed')

    def clear_data(self):
        if self.update is True:

            os.system(f'rm -rf {self.zarr_store}')

            for animal_id in tqdm(os.listdir(self.feather_store)):
                file = os.path.join(self.feather_store, animal_id)
                if os.path.exists(file):
                    os.remove(file)

    def organisation_id_for_animal_id(self, animal_id):
        if self._animal_orga_map is None:
            with self.readfile() as file:
                self._animal_orga_map = json.loads(
                    file['lookup/animal_to_orga'][()])

        try:
            return self._animal_orga_map[animal_id]
        except KeyError:
            return None

    def sanitize_data(self):
        organisation_ids = self.organisation_ids

        for organisation_id in tqdm(organisation_ids):
            with self.readfile() as file:
                organisation = json.loads(
                    file[f'data/{organisation_id}/organisation'][()])

            organisation.pop('account', None)
            organisation.pop('name', None)
            organisation.pop('partner_id', None)

            with self.readfile('a') as file:
                orga_group = file[f'data/{organisation_id}']
                del orga_group['organisation']

                orga_group.create_dataset(
                    name='organisation',
                    data=json.dumps(organisation))

    def normalize_numeric_cols(self):
        animal_ids = os.listdir(self.feather_store)

        scaler = StandardScaler()

        numeric_cols = [
            'act',
            'temp',
            'DIM',
            'temp_filtered',
            'act_group_mean',
            'heat_feature'
        ]

        # fit scaler to data
        kwargs = {
            'desc': 'Fitting scaler parameters',
            'smoothing': 0.01
        }

        for animal_id in tqdm(animal_ids, **kwargs):
            frame = load_animal(self.feather_store, animal_id)

            scaler.partial_fit(frame[numeric_cols])

        scaler_store_path = os.path.join(self.model_store, 'scaler')
        joblib.dump(scaler, scaler_store_path)

        scaler = joblib.load(scaler_store_path)

        kwargs = {
            'total': len(animal_ids),
            'unit': 'animals',
            'unit_scale': True,
            'leave': True,
            'desc': 'Scaling numeric columns',
            'smoothing': 0.01
        }

        with ProcessPoolExecutor(os.cpu_count()) as process_pool:
            results = [
                process_pool.submit(
                    scale_worker,
                    animal_id,
                    numeric_cols,
                    scaler,
                    self.feather_store
                ) for animal_id in animal_ids]

            for f in tqdm(as_completed(results), **kwargs):
                pass

        return None

    def store_to_zarr(self):
        store = zarr.DirectoryStore(self.zarr_store)
        group = zarr.hierarchy.group(store=store)
        z_arr = group.require_dataset(
            'origin',
            shape=(0, 290),
            chunks=(50000, None),
            dtype='f4'
        )

        animal_ids = os.listdir(self.feather_store)

        kwargs = {
            'total': len(animal_ids),
            'unit': 'animals',
            'unit_scale': True,
            'leave': True,
            'desc': 'Writing to zarr store',
            'smoothing': 0.01
        }

        for animal_id in tqdm(animal_ids, **kwargs):
            fnn_worker(
                self.feather_store,
                z_arr,
                animal_id,
                fix_imbalance=True
            )

        # with ProcessPoolExecutor(os.cpu_count()) as process_pool:
        #     results = [
        #         process_pool.submit(
        #             fnn_worker,
        #             self.feather_store,
        #             z_arr,
        #             animal_id
        #         ) for animal_id in animal_ids]

        #     for f in tqdm(as_completed(results), **kwargs):
        #         pass

        return None

    def validation_split(self):
        pbar = ProgressBar()
        pbar.register()

        array = dd.array.from_zarr(
            os.path.join(self.zarr_store, 'origin'))

        trainset, testset = train_test_split(
            array,
            shuffle=True,
            test_size=0.3,
        )

        testset = dd.array.rechunk(testset, chunks=(50000, 290))
        trainset = dd.array.rechunk(trainset, chunks=(50000, 290))

        testset.to_zarr(
            os.path.join(self.zarr_store, 'testset'),
            overwrite=True
        )

        trainset = dd.array.random.permutation(trainset)

        trainset.to_zarr(
            os.path.join(self.zarr_store, 'trainset1'),
            overwrite=True
        )

        # trainset = dd.array.random.permutation(trainset)

        # trainset.to_zarr(
        #     os.path.join(self.zarr_store, 'trainset2'),
        #     overwrite=True
        # )

        # trainset = dd.array.random.permutation(trainset)

        # trainset.to_zarr(
        #     os.path.join(self.zarr_store, 'trainset3'),
        #     overwrite=True
        # )

        pbar.unregister()

    def test(self):
        plt = plot_setup()

        frame = load_animal(self.feather_store, '59e75f2b9e182f68cf25721d')
        frame = frame.reset_index().set_index('datetime')

        ax = frame.loc[
            :, ('act',
                'temp',
                'temp_filtered',
                'act_group_mean',
                'DIM',
                'heat_feature')
        ].plot()

        ymin, ymax = ax.get_ylim()

        ax.fill_between(
            frame.index.to_numpy(), ymin, ymax,
            where=frame.cyclic.to_numpy(),
            facecolor='g',
            alpha=0.5,
            label='cyclic heats'
        )

        ax.fill_between(
            frame.index.to_numpy(), ymin, ymax,
            where=frame.inseminated.to_numpy(),
            facecolor='y',
            alpha=0.5,
            label='inseminated heats'
        )

        ax.fill_between(
            frame.index.to_numpy(), ymin, ymax,
            where=frame.pregnant.to_numpy(),
            facecolor='b',
            alpha=0.5,
            label='pregnant heats'
        )

        ax.fill_between(
            frame.index.to_numpy(), ymin, ymax,
            where=frame.deleted.to_numpy(),
            facecolor='r',
            alpha=0.5,
            label='deleted heats'
        )

        plt.legend()

        plt.grid()
        plt.show()

    def test_feature(self):
        plt = plot_setup()

        frame = load_animal(self.zarr_store, '59e75f2b9e182f68cf25721d')
        frame = frame.reset_index().set_index('datetime')

        ax = frame[['act', 'act_group_mean', 'heat_feature']].plot()

        ymin, ymax = ax.get_ylim()

        ax.fill_between(
            frame.index.to_numpy(), ymin, ymax,
            where=frame.cyclic.to_numpy(),
            facecolor='g',
            alpha=0.5,
            label='cyclic heats'
        )

        ax.fill_between(
            frame.index.to_numpy(), ymin, ymax,
            where=frame.inseminated.to_numpy(),
            facecolor='y',
            alpha=0.5,
            label='inseminated heats'
        )

        ax.fill_between(
            frame.index.to_numpy(), ymin, ymax,
            where=frame.pregnant.to_numpy(),
            facecolor='b',
            alpha=0.5,
            label='pregnant heats'
        )

        ax.fill_between(
            frame.index.to_numpy(), ymin, ymax,
            where=frame.deleted.to_numpy(),
            facecolor='r',
            alpha=0.5,
            label='deleted heats'
        )

        plt.legend()

        plt.grid()
        plt.show()

    def test_zarr(self):
        store = zarr.DirectoryStore(self.zarr_store)
        group = zarr.open(store)
        arr1 = group['trainset1']
        # arr2 = group['trainset2']

        print(arr1[0])
        print(arr1.info)
        # print(arr2[0])

    def run(self):
        self.clear_data()
        self.arrange_data()
        self.normalize_numeric_cols()
        self.store_to_zarr()
        self.validation_split()
        # self.test()
        # self.test_feature()
        # self.test_zarr()


def main():
    transformer = DataTransformer(['59e7515edb84e482acce8339'], update=True)
    # transformer = DataTransformer()
    transformer.run()


if __name__ == '__main__':
    main()
