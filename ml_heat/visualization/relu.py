import os
from ml_heat.helper import plot_setup

savepath = os.path.join('ml_heat', 'visualization', 'plots')

plt = plot_setup()
plt.rcParams['axes.grid'] = True
plt.rcParams['figure.figsize'] = 6, 4

x = range(-3, 4)
y = [max(0, v) for v in x]

plt.plot(x, y)

ax = plt.gca()

ax.set_xlabel('z')
ax.set_ylabel('g(z)')

plt.savefig(os.path.join(savepath, 'relu.pdf'),
            dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', format=None, transparent=False,
            bbox_inches=None, pad_inches=0.1, metadata=None)

plt.show()

savepath = os.path.join('ml_heat', 'visualization', 'plots')

plt = plot_setup()
plt.rcParams['axes.grid'] = True
plt.rcParams['figure.figsize'] = 6, 4

x = np.arange(-6, 6, 0.1)
y = np.exp(x) / (np.exp(x) + 1)

plt.plot(x, y)

ax = plt.gca()

ax.set_xlabel('z')
ax.set_ylabel('g(z)')

plt.savefig(os.path.join(savepath, 'logistic.pdf'),
            dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', format=None, transparent=False,
            bbox_inches=None, pad_inches=0.1, metadata=None)

plt.show()
