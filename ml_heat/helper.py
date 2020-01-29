import os
import vaex as vx


def load_vxframe(vxstore):
    vxfiles = [
        os.path.join(vxstore, x) for x in
        os.listdir(vxstore) if x.endswith('.hdf5')]

    frame = vx.open(vxfiles[0])

    for file in vxfiles[1:]:
        frame2 = vx.open(file)
        frame = frame.join(frame2)

    return frame


def load_data(store_path, organisation_ids=None, dtype='pandas'):
    store = load_vxframe(store_path)

    if organisation_ids is None:
        out = store
    else:
        assert type(organisation_ids) == list
        out = store[store.organisation_id.isin(organisation_ids)]

    if dtype == 'pandas':
        out = out.to_pandas_df().drop('index', axis=1)
        out = out.set_index([
            'organisation_id',
            'group_id',
            'animal_id',
            'datetime'
        ]).sort_index()

    elif dtype == 'vaex':
        pass
    else:
        raise ValueError('Unknown output datatype specified!')

    return out


def plot_setup():
    import matplotlib.pyplot as plt
    from cycler import cycler

    plt.style.use('dark_background')
    plt.rcParams['axes.prop_cycle'] = cycler(
        color=[u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd',
               u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf'])

    return plt
