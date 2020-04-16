import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler


def plot_setup():
    plt.style.use('dark_background')
    plt.rcParams['axes.prop_cycle'] = cycler(
        color=[u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd',
               u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf'])

    return plt


def load_organisation_from_hdf5(path, organisation_id):
    frame = pd.read_hdf(
        path,
        key='dataset',
        where=f'organisation_id=="{organisation_id}"')

    return frame
