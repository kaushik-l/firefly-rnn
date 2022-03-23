from train import train
import numpy as np
import sys
import torch
from matplotlib import pyplot as plt
from model import Network, Task, Plant, Algorithm

modelname = sys.argv[1]
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1

if modelname == 'firefly-baseline':
    # initialize
    net = np.empty(1, dtype=object)
    task = np.empty(1, dtype=object)
    algo = np.empty(1, dtype=object)
    plant = np.empty(1, dtype=object)
    learning = np.empty(1, dtype=object)
    # train using bptt
    net[0], task[0], algo[0], plant[0], learning[0] = \
        train(arch='ppc', N=100, S=4, R=2, task='firefly',
              learningsites=('wr'), algo='Adam', Nepochs=5000, lr=1e-3, seed=seed)

if modelname == 'firefly-feedpos':
    # initialize
    net = np.empty(1, dtype=object)
    task = np.empty(1, dtype=object)
    algo = np.empty(1, dtype=object)
    plant = np.empty(1, dtype=object)
    learning = np.empty(1, dtype=object)
    # train using bptt
    net[0], task[0], algo[0], plant[0], learning[0] = \
        train(arch='ppc', N=100, S=4, R=2, task='firefly', feed_pos=True,
              learningsites=('wr'), algo='Adam', Nepochs=5000, lr=1e-3, seed=seed)

if modelname == 'firefly-feedbelief':
    # initialize
    net = np.empty(1, dtype=object)
    task = np.empty(1, dtype=object)
    algo = np.empty(1, dtype=object)
    plant = np.empty(1, dtype=object)
    learning = np.empty(1, dtype=object)
    # train using bptt
    net[0], task[0], algo[0], plant[0], learning[0] = \
        train(arch='ppc', N=100, S=4, R=2, task='firefly', feed_belief=True,
              learningsites=('wr'), algo='Adam', Nepochs=5000, lr=1e-3, seed=seed)

if modelname == 'firefly-baseline-pn':
    # initialize
    nconds = 6
    net = np.empty((nconds, 1), dtype=object)
    task = np.empty((nconds, 1), dtype=object)
    algo = np.empty((nconds, 1), dtype=object)
    plant = np.empty((nconds, 1), dtype=object)
    learning = np.empty((nconds, 1), dtype=object)
    conds = np.arange(nconds)
    for i in np.arange(nconds):
        # train using bptt
        net[i], task[i], algo[i], plant[i], learning[i] = \
            train(arch='ppc', N=100, S=4, R=2, task='firefly', db=(1, 6), sn=1, pn=.2 * i,
                  learningsites=('J'), algo='Adam', Nepochs=5000, lr=1e-3, seed=seed)

if modelname == 'firefly-baseline-sn':
    # initialize
    nconds = 6
    net = np.empty((nconds, 1), dtype=object)
    task = np.empty((nconds, 1), dtype=object)
    algo = np.empty((nconds, 1), dtype=object)
    plant = np.empty((nconds, 1), dtype=object)
    learning = np.empty((nconds, 1), dtype=object)
    conds = np.arange(nconds)
    for i in np.arange(nconds):
        # train using bptt
        net[i], task[i], algo[i], plant[i], learning[i] = \
            train(arch='ppc', N=100, S=4, R=2, task='firefly', db=(1, 6), sn=i, pn=.2,
                  learningsites=('J'), algo='Adam', Nepochs=5000, lr=1e-3, seed=seed)

if modelname == 'firefly-baseline-db':
    # initialize
    nconds = 3
    net = np.empty((nconds, 1), dtype=object)
    task = np.empty((nconds, 1), dtype=object)
    algo = np.empty((nconds, 1), dtype=object)
    plant = np.empty((nconds, 1), dtype=object)
    learning = np.empty((nconds, 1), dtype=object)
    conds = np.arange(nconds)
    for i in np.arange(nconds):
        # train using bptt
        net[i], task[i], algo[i], plant[i], learning[i] = \
            train(arch='ppc', N=100, S=4, R=2, task='firefly', db=(1, 2 * (i + 1)), sn=1, pn=.2,
                  learningsites=('J'), algo='Adam', Nepochs=5000, lr=1e-3, seed=seed)

if modelname == 'firefly-feedpos-pn':
    # initialize
    nconds = 6
    net = np.empty((nconds, 1), dtype=object)
    task = np.empty((nconds, 1), dtype=object)
    algo = np.empty((nconds, 1), dtype=object)
    plant = np.empty((nconds, 1), dtype=object)
    learning = np.empty((nconds, 1), dtype=object)
    conds = np.arange(nconds)
    for i in np.arange(nconds):
        # train using bptt
        net[i], task[i], algo[i], plant[i], learning[i] = \
            train(arch='ppc', N=100, S=4, R=2, task='firefly', db=(1, 6), sn=1, pn=.2 * i,
                  feed_pos=True, learningsites=('J'), algo='Adam', Nepochs=5000, lr=1e-3, seed=seed)

if modelname == 'firefly-feedpos-sn':
    # initialize
    nconds = 6
    net = np.empty((nconds, 1), dtype=object)
    task = np.empty((nconds, 1), dtype=object)
    algo = np.empty((nconds, 1), dtype=object)
    plant = np.empty((nconds, 1), dtype=object)
    learning = np.empty((nconds, 1), dtype=object)
    conds = np.arange(nconds)
    for i in np.arange(nconds):
        # train using bptt
        net[i], task[i], algo[i], plant[i], learning[i] = \
            train(arch='ppc', N=100, S=4, R=2, task='firefly', db=(1, 6), sn=i, pn=.2,
                  feed_pos=True, learningsites=('J'), algo='Adam', Nepochs=5000, lr=1e-3, seed=seed)

if modelname == 'firefly-feedpos-db':
    # initialize
    nconds = 3
    net = np.empty((nconds, 1), dtype=object)
    task = np.empty((nconds, 1), dtype=object)
    algo = np.empty((nconds, 1), dtype=object)
    plant = np.empty((nconds, 1), dtype=object)
    learning = np.empty((nconds, 1), dtype=object)
    conds = np.arange(nconds)
    for i in np.arange(nconds):
        # train using bptt
        net[i], task[i], algo[i], plant[i], learning[i] = \
            train(arch='ppc', N=100, S=4, R=2, task='firefly', db=(1, 2 * (i + 1)), sn=1, pn=.2,
                  feed_pos=True, learningsites=('J'), algo='Adam', Nepochs=5000, lr=1e-3, seed=seed)

# save
torch.save({'net': net, 'task': task, 'algo': algo, 'plant': plant, 'learning': learning},
           '//burg//theory//users//jl5649//firefly-rnn//' + modelname + '//' + str(seed) + '.pt')
