
import os

from ml_heat.helper import (
    plot_setup,
    load_animal
)

from ml_heat.preprocessing.transform_data import DataTransformer

datapath = DataTransformer().feather_store

savepath = os.path.join('ml_heat', 'visualization', 'plots')


def plot_sensordata():
    plt = plot_setup()
    plt.rcParams['axes.grid'] = True
    plt.rcParams['figure.figsize'] = 12, 8

    frame = load_animal(datapath, '59e75f2b9e182f68cf25721d')
    frame = frame.reset_index().set_index('datetime')

    fig, ax = plt.subplots(4, 1, sharex=True)

    ax[0].plot(frame.index, frame.temp)
    ax[0].set_ylabel('Rumen temperature (°C)')
    ax[1].plot(frame.index, frame.act)
    ax[1].set_ylabel('Activity')
    ax[2].plot(frame.index, frame.act_group_mean)
    ax[2].set_ylabel('Group activity')
    ax[3].plot(frame.index, frame.DIM)
    ax[3].set_ylabel('Days since parturition')

    for axis in ax:
        ymin, ymax = axis.get_ylim()

        axis.fill_between(
            frame.index.to_numpy(), ymin, ymax,
            where=frame.cyclic.to_numpy(),
            facecolor='g',
            alpha=0.5,
            label='cyclic heats'
        )

        axis.fill_between(
            frame.index.to_numpy(), ymin, ymax,
            where=frame.inseminated.to_numpy(),
            facecolor='y',
            alpha=0.5,
            label='inseminated heats'
        )

        axis.fill_between(
            frame.index.to_numpy(), ymin, ymax,
            where=frame.pregnant.to_numpy(),
            facecolor='b',
            alpha=0.5,
            label='pregnant heats'
        )

        axis.fill_between(
            frame.index.to_numpy(), ymin, ymax,
            where=frame.deleted.to_numpy(),
            facecolor='r',
            alpha=0.5,
            label='deleted heats'
        )

    # plt.tight_layout()
    plt.savefig(os.path.join(savepath, 'sensordata.pdf'),
                dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1,
                frameon=None, metadata=None)


def plot_sensordata1():
    plt = plot_setup()
    plt.rcParams['axes.grid'] = True
    plt.rcParams['figure.figsize'] = 12, 8

    frame = load_animal(datapath, '59e75f2b9e182f68cf25721d')
    frame = frame.reset_index().set_index('datetime')

    frame = frame['2019-01-01':'2019-01-13']

    fig, ax = plt.subplots(4, 1, sharex=True)

    ax[0].plot(frame.index, frame.temp)
    ax[0].set_ylabel('Rumen temperature (°C)')
    ax[1].plot(frame.index, frame.act)
    ax[1].set_ylabel('Activity')
    ax[2].plot(frame.index, frame.act_group_mean)
    ax[2].set_ylabel('Group activity')
    ax[3].plot(frame.index, frame.DIM)
    ax[3].set_ylabel('Days since parturition')

    for axis in ax:
        ymin, ymax = axis.get_ylim()

        axis.fill_between(
            frame.index.to_numpy(), ymin, ymax,
            where=frame.cyclic.to_numpy(),
            facecolor='g',
            alpha=0.5,
            label='cyclic heats'
        )

        axis.fill_between(
            frame.index.to_numpy(), ymin, ymax,
            where=frame.inseminated.to_numpy(),
            facecolor='y',
            alpha=0.5,
            label='inseminated heats'
        )

        axis.fill_between(
            frame.index.to_numpy(), ymin, ymax,
            where=frame.pregnant.to_numpy(),
            facecolor='b',
            alpha=0.5,
            label='pregnant heats'
        )

        axis.fill_between(
            frame.index.to_numpy(), ymin, ymax,
            where=frame.deleted.to_numpy(),
            facecolor='r',
            alpha=0.5,
            label='deleted heats'
        )

    # plt.tight_layout()
    plt.savefig(os.path.join(savepath, 'sensordata_micro.pdf'),
                dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1,
                frameon=None, metadata=None)


def main():
    plot_sensordata()
    plot_sensordata1()


if __name__ == '__main__':
    main()
