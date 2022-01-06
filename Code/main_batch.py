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
        train(arch='ppc', N=100, S=4, R=2, task='firefly', algo='Adam', Nepochs=2500, lr=1e-3, seed=seed)

# save
torch.save({'net': net, 'task': task, 'algo': algo, 'plant': plant, 'learning': learning},
           '//burg//theory//users//jl5649//firefly-rnn//' + modelname + '//' + str(seed) + '.pt')
