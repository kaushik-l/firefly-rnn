from train import train
import numpy as np
import torch
from matplotlib import pyplot as plt
from model import Network, Task, Algorithm

net, task, algo, plant, learning = \
    train(arch='ppc', N=256, S=4, R=2, task='firefly', algo='Adam', Nepochs=10000, lr=1e-3, seed=1)
