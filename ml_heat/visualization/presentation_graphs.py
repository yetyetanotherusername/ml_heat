
import os

from ml_heat.helper import (
    plot_setup,
    load_animal
)

import numpy as np

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
                orientation='portrait', format=None, transparent=False,
                bbox_inches=None, pad_inches=0.1, metadata=None)


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
                orientation='portrait', format=None, transparent=False,
                bbox_inches=None, pad_inches=0.1, metadata=None)


def plot_activity():
    plt = plot_setup()
    plt.rcParams['axes.grid'] = True
    plt.rcParams['figure.figsize'] = 12, 8

    frame = load_animal(datapath, '59e75f2b9e182f68cf25721d')
    frame = frame.reset_index().set_index('datetime')

    frame = frame['2019-01-01':'2019-01-13']

    # fig, ax = plt.subplots(4, 1, sharex=True)

    # ax[0].plot(frame.index, frame.temp)
    # ax[0].set_ylabel('Rumen temperature (°C)')
    plt.figure()
    plt.plot(frame.index, frame.act)
    axis = plt.gca()
    axis.set_ylabel('Activity index')
    # ax[2].plot(frame.index, frame.act_group_mean)
    # ax[2].set_ylabel('Group activity')
    # ax[3].plot(frame.index, frame.DIM)
    # ax[3].set_ylabel('Days since parturition')

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

    axis.set_aspect(0.2)

    # plt.tight_layout()
    plt.savefig(os.path.join(savepath, 'act_micro.pdf'),
                dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', format=None, transparent=False,
                bbox_inches='tight', pad_inches=0.1, metadata=None)


def plot_temp():
    plt = plot_setup()
    plt.rcParams['axes.grid'] = True
    plt.rcParams['figure.figsize'] = 12, 8

    frame = load_animal(datapath, '59e75f2b9e182f68cf25721d')
    frame = frame.reset_index().set_index('datetime')

    frame = frame['2019-01-01':'2019-01-13']

    # fig, ax = plt.subplots(4, 1, sharex=True)

    plt.figure()
    plt.plot(frame.index, frame.temp)
    axis = plt.gca()
    axis.set_ylabel('Rumen temperature (°C)')
    # plt.plot(frame.index, frame.act)
    # axis.set_ylabel('Activity index')
    # ax[2].plot(frame.index, frame.act_group_mean)
    # ax[2].set_ylabel('Group activity')
    # ax[3].plot(frame.index, frame.DIM)
    # ax[3].set_ylabel('Days since parturition')

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

    axis.set_aspect(0.2)

    # plt.tight_layout()
    plt.savefig(os.path.join(savepath, 'temp_micro.pdf'),
                dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', format=None, transparent=False,
                bbox_inches='tight', pad_inches=0.1, metadata=None)


def plot_temps():
    plt = plot_setup()
    plt.rcParams['axes.grid'] = True
    plt.rcParams['figure.figsize'] = 12, 8

    frame = load_animal(datapath, '59e75f2b9e182f68cf25721d')
    frame = frame.reset_index().set_index('datetime')

    frame = frame['2019-01-01':'2019-01-13']

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(9.0, 4.0))
    axes[0].plot(frame.index, frame.temp)
    axes[0].set_ylabel('Rumen\ntemperature (°C)')
    # axes[0].set_aspect(0.2)
    axes[1] = plt.subplot(212, sharex=axes[0])
    axes[1].plot(frame.index, frame.temp_filtered)
    axes[1].set_ylabel('Filtered\nrumen\ntemperature (°C)')
    axes[1].set_xlabel('Time')
    # axes[1].set_aspect(0.2)
    axes[1].set_ylim(31, 41)

    # plt.tight_layout()
    plt.savefig(os.path.join(savepath, 'temp_vs_temp_wo.pdf'),
                dpi=None, facecolor='w', edgecolor='w',
                orientation='landscape', format=None, transparent=False,
                bbox_inches='tight', pad_inches=0.1, metadata=None)

    plt.show()


def plot_confusion_matrix():
    plt = plot_setup()
    plt.rcParams['figure.figsize'] = 4, 3

    labels = ['No heat', 'heat']

    array = np.array([[495676930, 32952514],
                      [322323, 2281286]])

    norm_array = array / array.sum(axis=1)[:, None]

    fig, ax = plt.subplots()
    im = ax.imshow(norm_array)

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Ratio', rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    # # ... and label them with the respective list entries
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # # Loop over data dimensions and create text annotations.
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j, i, array[i, j],
                           ha="center", va="center", color="w")

    # ax.set_title("Harvest of local farmers (in tons/year)")
    fig.tight_layout()
    plt.savefig(os.path.join(savepath, 'lstm_matrix2.pdf'),
                dpi=None, facecolor='w', edgecolor='w',
                orientation='landscape', format=None, transparent=False,
                bbox_inches='tight', pad_inches=0.1, metadata=None)
    plt.show()


def main():
    # plot_sensordata()
    # plot_sensordata1()
    # plot_activity()
    # plot_temp()
    # plot_temps()
    plot_confusion_matrix()


if __name__ == '__main__':
    main()
