#!/usr/bin/python3
# coding: utf8

import os
import json
import pickle
import vaex as vx
import vaex.ml.transformations as tf
import h5py as h5
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from sxutils.models.animal.cycle import AnimalLifecycle
from sxutils.munch.cyclemunch import cycle_munchify
from ml_heat.preprocessing.animal_serde import AnimalSchemaV2

from ml_heat.helper import (
    load_vxframe,
    pandas_to_vaex,
    plot_setup,
    load_data
)


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
    for dt in cyclic_dts:
        idx = None
        from_dt = dt - span
        to_dt = dt + span
        try:
            idx = inframe.loc[
                from_dt:to_dt, 'act'].rolling(6, min_periods=1).mean().idxmax()
        except ValueError:
            continue

        inframe.cyclic.at[idx] = True
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

    # find indices closest to cyclic heats and set cyclic column to true
    # if no matching index is found inside tolerance of 24h, disregard
    span = pd.Timedelta(hours=12)
    for dt in insemination_dts:
        idx = None
        from_dt = dt - span
        to_dt = dt + span
        try:
            idx = inframe.loc[
                from_dt:to_dt, 'act'].rolling(6, min_periods=1).mean().idxmax()
        except ValueError:
            continue

        inframe.inseminated.at[idx] = True
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
    for dt in pregnant_ts:
        idx = None
        from_dt = dt - span
        to_dt = dt + span
        try:
            idx = inframe.loc[
                from_dt:to_dt, 'act'].rolling(6, min_periods=1).mean().idxmax()
        except ValueError:
            continue

        inframe.pregnant.at[idx] = True
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
    for dt in detected_heats:
        idx = None
        from_dt = dt - span
        to_dt = dt + span
        try:
            idx = inframe.loc[
                from_dt:to_dt, 'act'].rolling(6, min_periods=1).mean().idxmax()
        except ValueError:
            continue

        inframe.deleted.at[idx] = True
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
    race = animal.get('race', 'N/A')
    if race is None:
        race = 'N/A'

    if len(race) > 10:
        race = [word[0] + '_' for word in animal['race'].split('_')]
        race = ''.join(race)

    # partner_id = organisation.get('partner_id', 'N/A')
    # if partner_id is None:
    #     partner_id = 'N/A'

    # if len(partner_id) > 10:
    #     partner_id = partner_id[:10]

    group_id = animal.get('group_id', 'N/A')
    if group_id is None:
        group_id = 'N/A'

    # country field may be unavailable
    country = animal.get('metadata', {}).get('country', 'N/A')

    inframe['organisation_id'] = organisation['_id']
    inframe['group_id'] = group_id
    inframe['animal_id'] = animal['_id']
    inframe['race'] = race
    inframe['country'] = country
    # inframe['partner_id'] = partner_id
    inframe['DIM'] = calculate_dims(inframe.index, animal)
    inframe['temp_filtered'] = calc_temp_without_drink_spikes(inframe)

    inframe.race = inframe.race.astype('category')
    inframe.country = inframe.country.astype('category')
    # inframe.partner_id = inframe.partner_id.astype('category')

    return inframe


def add_group_feature(inframe):
    frame = inframe['act'].drop(
        index='N/A', level=0, errors='ignore').copy(deep=True)

    if frame.empty:
        inframe['act_group_mean'] = float('nan')
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


def pickle_animal(frame, animal_id, temp_path):
    write_path = os.path.join(temp_path, animal_id)
    with open(write_path, 'wb') as writefile:
        pickle.dump(frame, writefile)

    return animal_id


def transform_organisation(organisation_id, readpath, temp_path):
    position = multiprocessing.current_process()._identity[0] + 1
    with h5.File(readpath, 'r') as readfile:

        organisation = json.loads(
            readfile[f'data/{organisation_id}/organisation'][()])

        animal_ids = list(readfile[f'data/{organisation_id}'].keys())
        animal_ids = list(filter(
            lambda x: x != 'organisation', animal_ids))

        # out of ram constraints, we cannot process such big organisations
        if len(animal_ids) > 2000:
            return organisation

        framelist = []
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

    frame = pd.concat(reslist, sort=False)

    frame['organisation_id'] = organisation_id
    frame = frame.set_index(['organisation_id', frame.index])
    frame = frame.sort_index()

    animal_ids = frame.index.unique(level='animal_id')

    reslist = []
    for animal_id in tqdm(
            animal_ids,
            leave=False,
            desc=f'Thread {position - 1} pickle loop',
            position=position):

        subframe = frame.loc[
            (slice(None), slice(None), animal_id, slice(None)), slice(None)]

        pickle_animal(subframe, animal_id, temp_path)

    return organisation_id


def store_vaex_hdf5(filepath, storepath):
    with open(filepath, 'rb') as file:
        frame = pickle.load(file)

    if frame.empty:
        os.remove(filepath)
        return filepath

    frame.race = frame.race.astype('str')
    frame.country = frame.country.astype('str')
    # frame.partner_id = frame.partner_id.astype('str')

    vxframe = pandas_to_vaex(frame)
    del frame
    vxfilepath = os.path.join(
        storepath, os.path.basename(filepath) + '.hdf5')
    vxframe.export_hdf5(vxfilepath)
    vxframe.close_files()
    os.remove(filepath)
    return filepath


class DataTransformer(object):
    def __init__(self, organisation_ids=None):
        self._organisation_ids = organisation_ids
        self.store_path = os.path.join(
            os.getcwd(), 'ml_heat', '__data_store__')
        self.raw_store_path = os.path.join(self.store_path, 'rawdata.hdf5')
        self.train_store_path = os.path.join(self.store_path, 'traindata.hdf5')
        self.vxstore = os.path.join(self.store_path, 'vaex_store')
        self.load_vxframe = load_vxframe
        if not os.path.exists(self.store_path):
            os.mkdir(self.store_path)
        if not os.path.exists(self.vxstore):
            os.mkdir(self.vxstore)

        self._animal_orga_map = None

    def readfile(self, how='r'):
        return h5.File(self.raw_store_path, how)

    @property
    def organisation_ids(self):
        if self._organisation_ids is None:
            with self.readfile() as file:
                self._organisation_ids = list(file['data'].keys())
        return self._organisation_ids

    def transform_data(self):
        print('Transforming data...')
        # create temp storage folder
        temp_path = os.path.join(self.store_path, 'preprocessing_temp')
        if not os.path.exists(temp_path):
            os.mkdir(temp_path)
            filtered_orga_ids = self.organisation_ids
        else:
            files = os.listdir(temp_path)
            processed_orgas = list(
                set([self.organisation_id_for_animal_id(file)
                     for file in files]))

            filtered_orga_ids = [
                x for x in self.organisation_ids if x not in processed_orgas]

        # TODO: try running the organisation loop in processes now that memory
        # usage is fixed
        # for organisation_id in tqdm(
        #         filtered_orga_ids, desc='organisation loop'):
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
                'position': 1
            }

            for f in tqdm(as_completed(results), **kwargs):
                pass

    def store_data(self):
        print('Writing data to hdf file...')
        temp_path = os.path.join(self.store_path, 'preprocessing_temp')
        files = os.listdir(temp_path)
        filepaths = [os.path.join(temp_path, p) for p in files]

        with ProcessPoolExecutor(os.cpu_count()) as process_pool:
            results = [
                process_pool.submit(store_vaex_hdf5, filepath, self.vxstore)
                for filepath in filepaths]

            kwargs = {
                'total': len(filepaths),
                'unit': 'animals',
                'unit_scale': True,
                'leave': True
            }

            for f in tqdm(as_completed(results), **kwargs):
                pass

        vxfiles = [
            os.path.join(self.vxstore, x) for x in
            os.listdir(self.vxstore) if x.endswith('.hdf5') and
            not x.startswith('temp')]

        # can't open too many files at once, so first chunk it :-P
        # split it into 100 chunks, remove chunks that may be empty
        chunked = [x for x in np.array_split(vxfiles, 40) if x.size > 0]

        print('unifying vaex database into chunks')

        for idx, chunk in tqdm(enumerate(chunked), total=len(chunked)):
            vxframe = vx.open(chunk[0])
            for iidx in range(1, len(chunk)):
                vxframe.concat(vx.open(chunk[iidx]))

            vxframe.drop('index').export_hdf5(
                os.path.join(self.vxstore, f'temp{idx}.hdf5'))
            vxframe.close_files()
            for file in chunk:
                os.remove(file)

        print('unifying chunks into single file')
        # unify the chunked files into one file :-P
        vxfiles = [
            os.path.join(self.vxstore, x) for x in
            os.listdir(self.vxstore) if x.endswith('.hdf5') and
            x.startswith('temp')]

        vxframe = vx.open_many(vxfiles)

        vxframe.export_hdf5(
            os.path.join(self.vxstore, 'traindata.hdf5'))

        print('Cleaning up...')
        os.rmdir(temp_path)
        for file in vxfiles:
            os.remove(file)
        print('Done!')

    def clear_data(self):
        if os.path.exists(self.train_store_path):
            os.remove(self.train_store_path)

        if os.path.exists(self.vxstore):
            vxfiles = os.listdir(self.vxstore)
            for file in vxfiles:
                os.remove(os.path.join(self.vxstore, file))

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

    def one_hot_encode(self):
        print('Performing one hot encoding...')
        frame = self.load_vxframe(self.vxstore)
        old_cols = frame.get_column_names()
        string_cols = ['race', 'country']
        encoder = tf.OneHotEncoder(
            features=string_cols)
        frame_encoded = encoder.fit_transform(frame)
        all_cols = frame_encoded.get_column_names()
        new_cols = [col for col in all_cols if col not in old_cols]
        frame_encoded[new_cols].export_hdf5(
            os.path.join(self.vxstore, 'one_hot.hdf5'),
            virtual=True)
        print('Done!')

    def scale_numeric_cols(self):
        print('Performing data scaling...')
        frame = self.load_vxframe(self.vxstore)
        old_cols = frame.get_column_names()
        numeric_cols = [
            'act', 'temp', 'temp_filtered', 'DIM', 'act_group_mean'
        ]

        encoder = tf.RobustScaler(features=numeric_cols)
        frame_scaled = encoder.fit_transform(frame)
        all_cols = frame_scaled.get_column_names()
        new_cols = [col for col in all_cols if col not in old_cols]
        frame_scaled[new_cols].export_hdf5(
            os.path.join(self.vxstore, 'scaled.hdf5'),
            virtual=True)
        print('Done!')

    def test(self):
        plt = plot_setup()
        frame = load_data(self.vxstore, ['59e7515edb84e482acce8339'])
        frame = frame.loc[(
            slice(None),
            slice(None),
            '59e75f2b9e182f68cf25721d',
            slice(None)),
            :
        ].reset_index().set_index('datetime')

        frame.loc[
            :, ('act', 'temp', 'temp_filtered', 'act_group_mean', 'DIM')
        ].plot()

        cyclic = frame[frame.cyclic == True].index.to_list()  # noqa
        inseminated = frame[
            frame.inseminated == True].index.to_list()  # noqa
        pregnant = frame[frame.pregnant == True].index.to_list()  # noqa
        deleted = frame[frame.deleted == True].index.to_list()  # noqa

        td = pd.Timedelta(hours=4)
        for x in cyclic:
            xmin = x - td
            xmax = x + td
            plt.axvspan(
                xmin, xmax, color='g', alpha=0.5,
                label='cyclic heats'
            )

        for x in inseminated:
            xmin = x - td
            xmax = x + td
            plt.axvspan(
                xmin, xmax, color='y', alpha=0.5,
                label='inseminated heats'
            )

        for x in pregnant:
            xmin = x - td
            xmax = x + td
            plt.axvspan(
                xmin, xmax, color='b', alpha=0.5,
                label='pregnant heats'
            )

        for x in deleted:
            xmin = x - td
            xmax = x + td
            plt.axvspan(
                xmin, xmax, color='r', alpha=0.5,
                label='deleted heats'
            )

        plt.legend()

        plt.grid()
        plt.show()

    def run(self):
        self.clear_data()
        self.transform_data()
        self.store_data()
        self.one_hot_encode()
        self.scale_numeric_cols()
        # self.test()


def main():
    transformer = DataTransformer(['59e7515edb84e482acce8339'])
    # transformer = DataTransformer()
    transformer.run()


if __name__ == '__main__':
    main()
