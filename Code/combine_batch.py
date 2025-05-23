import os
import sys
import torch
import numpy as np

modelname = sys.argv[1]

fnames = [f for f in os.listdir('..//Data//' + modelname) if not f.startswith('.')]
nbatches = len(fnames)
net = np.empty((nbatches), dtype=object)
task = np.empty((nbatches), dtype=object)
algo = np.empty((nbatches), dtype=object)
plant = np.empty((nbatches), dtype=object)
learning = np.empty((nbatches), dtype=object)

count = 0
for f in fnames:
    data = torch.load('..//Data//' + modelname + '//' + f)
    net[count] = data['net'].flatten()
    task[count] = data['task'].flatten()
    algo[count] = data['algo'].flatten()
    plant[count] = data['plant'].flatten()
    learning[count] = data['learning'].flatten()
    count += 1

net = np.vstack(net)
task = np.vstack(task)
algo = np.vstack(algo)
plant = np.vstack(plant)
learning = np.vstack(learning)

torch.save({'net': net, 'task': task, 'algo': algo, 'plant': plant, 'learning': learning}, '..//Data//' + modelname + '.pt')