import os
import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler


def plot_setup():
    plt.style.use('dark_background')
    plt.rcParams['axes.prop_cycle'] = cycler(
        color=[u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd',
               u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf'])

    return plt


def load_animal(path, animal_id):
    frame = pd.read_feather(os.path.join(path, animal_id))
    frame = frame.set_index(
        ['organisation_id', 'group_id', 'animal_id', 'datetime'])
    return frame


def store_animal(frame, store_path, animal_id):
    filepath = os.path.join(store_path, animal_id)
    frame = frame.reset_index()
    if os.path.exists(filepath):
        os.remove(filepath)
    frame.to_feather(filepath)
    return animal_id


def duplicate_shift(series, shifts, name=None):
    n_cols = len(shifts)
    frame = pd.concat([series] * n_cols, axis=1)
    frame.columns = shifts
    frame = frame.apply(lambda x: x.shift(int(x.name)))
    if name is not None:
        frame.columns = [f'{name}{column}' for column in frame.columns]
    return frame
