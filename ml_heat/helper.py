import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import time

from pyarrow.lib import ArrowInvalid

from cycler import cycler
from matplotlib.patches import Rectangle
from itertools import count, islice

from scipy.linalg import hankel


def plot_setup():
    plt.style.use('dark_background')
    plt.rcParams['axes.prop_cycle'] = cycler(
        color=[u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd',
               u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf'])

    return plt


def load_animal(path, animal_id, set_index=True):
    try:
        frame = pd.read_feather(os.path.join(path, animal_id))
    except ArrowInvalid as e:
        print(e)
        print(os.path.join(path, animal_id))
        raise ArrowInvalid()

    if set_index is True:
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


def duplicate_shift(series, shifts, name=None, save_memory=False):
    if save_memory:
        shifts = range(shifts)
        n_cols = len(shifts)
        frame = pd.concat([series] * n_cols, axis=1)
        frame.columns = shifts
        frame = frame.apply(lambda x: x.shift(int(x.name)))
    else:
        h = np.flipud(hankel(np.flip(series))[:, :shifts])
        frame = pd.DataFrame(h, index=series.index).replace(0, np.nan)

    if name is not None:
        frame.columns = [f'{name}{column}' for column in frame.columns]
    return frame


def plot_timings(loader, n_batches, model_time=0.2, max_time=2.5):

    fig, ax = plt.subplots()
    ax.set_axisbelow(True)
    ax.yaxis.grid(which="major", color='black', linewidth=1)

    zero_time = time.time()

    worker_ids = {}
    worker_count = count()

    for result in islice(loader, n_batches):
        start = time.time()
        time.sleep(model_time)
        end = time.time()

        # check if already batched
        if isinstance(result[0], torch.Tensor):
            result = zip(*result)

        batch = []
        for item in result:
            data, worker, t1, t2 = tuple(map(scalar, item))

            # fix worker position in plot
            if worker != -1:
                if worker not in worker_ids:
                    worker_ids[worker] = next(worker_count)
                worker = worker_ids[worker]

            plot_time_box(data, worker, t1 - zero_time, t2 - zero_time, ax)
            batch.append(data)

        batch_str = ",".join(map(str, batch))
        plot_time_box(batch_str, -1, start - zero_time,
                      end - zero_time, ax, color='firebrick')

    max_worker = len(worker_ids) - 1

    ax.set_xlim(0, max_time)
    ax.set_ylim(-1.5, max_worker + 0.5)
    ax.set_xticks(np.arange(0, max_time, 0.2))
    ax.set_yticks(np.arange(-1, max_worker + 1, 1))
    ax.set_yticklabels([])
    ax.tick_params(axis='y', colors=(0, 0, 0, 0))

    fig.set_figwidth(16)
    fig.set_figheight((max_worker + 2) * 0.5)

    ax.xaxis.label.set_color('gray')
    ax.tick_params(axis='x', colors='gray')
    for spine in ax.spines.values():
        spine.set_edgecolor((0, 0, 0, 0))


def scalar(x):
    return x.item() if hasattr(x, 'item') else x


def plot_time_box(data, worker, t1, t2, ax, color='steelblue'):
    x = t1
    y = worker - 0.25
    w = t2 - t1
    h = 0.6

    rect = Rectangle((x, y), w, h, linewidth=2,
                     edgecolor='black', facecolor=color)

    ax.add_patch(rect)

    ax.text(x + (w * 0.5), y + (h * 0.5), str(data), va='center',
            ha='center', color='white', weight='bold')
