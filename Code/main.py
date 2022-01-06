from train import train
from test import test
import numpy as np
import torch
from matplotlib import pyplot as plt
from model import Network, Task, Algorithm

do_training, do_testing = True, False
k = 6

if do_training:
    net, task, algo, plant, learning = \
        train(arch='ppc', N=100, S=4, R=2, task='firefly', algo='Adam', Nepochs=2500, lr=1e-2, seed=1)
    # save
    torch.save({'net': net, 'task': task, 'algo': algo, 'plant': plant, 'learning': learning},
               '..//Data//firefly-baseline.pt')
elif do_testing:
    data = torch.load('..//Data//firefly-baseline.pt')
    net, task, plant, algo, learning = data['net'], data['task'], data['plant'], data['algo'], data['learning']
    tars, stops, prob_correct, prob_shuffled, auc, NTlist, sa, ha, ua, va, xa = \
        test(net[k, 0], task[k, 0], plant[k, 0], algo[k, 0], learning[k, 0], Nepochs=1000, seed=1)

zz = 1
