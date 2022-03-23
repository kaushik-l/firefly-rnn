from train import train
from test import test
from plot import plot
import numpy as np
import torch
from matplotlib import pyplot as plt
from model import Network, Task, Algorithm

do_training, do_testing = False, False
Nepochs = 100

test_baseline_pn, test_baseline_sn, test_baseline_db, test_feedpos_pn, test_feedpos_sn, test_feedpos_db = \
    False, False, False, False, False, False

test_baseline_gaingen, test_baseline_processgen, test_baseline_sensorygen, \
test_feedpos_gaingen, test_feedpos_processgen, test_feedpos_sensorygen = False, False, False, False, False, False

plot_behavior = True
plot_processnoise, plot_sensorynoise = False, False
plot_distgen, plot_gaingen, plot_processgen, plot_sensorygen = False, False, False, False

if do_training:
    net, task, algo, plant, learning = \
        train(arch='ppc', N=100, S=4, R=2, task='firefly', db=(1, 6), sn=.6, pn=.6,
              feed_pos=False, feed_belief=False, learningsites=('J'), algo='Adam', Nepochs=5000, lr=1e-3, seed=1)
    # save
    torch.save({'net': net, 'task': task, 'algo': algo, 'plant': plant, 'learning': learning},
               '..//Data//firefly-baseline.pt')

if do_testing:
    data = torch.load('..//Data//firefly-baseline.pt')
    net, task, plant, algo, learning = data['net'], data['task'], data['plant'], data['algo'], data['learning']
    tars, stops, prob_correct, prob_shuffled, auc, NTlist, sa, ha, ua, va, xa, ba = \
        test(net, task, plant, algo, learning, feed_pos=True, feed_belief=False, Nepochs=Nepochs, seed=1)

    data = torch.load('..//Data//firefly-feedpos.pt')
    net, task, plant, algo, learning = data['net'], data['task'], data['plant'], data['algo'], data['learning']
    for idx in range(20):
        tars, stops, prob_correct[1,idx,:], prob_shuffled[1,idx,:], auc[1,idx], NTlist, sa, ha, ua, va, xa, ba = \
            test(net[idx,0], task[idx,0], plant[idx,0], algo[idx,0], learning[idx,0],
                 feed_pos=True, feed_belief=False, Nepochs=Nepochs, seed=idx)

if test_baseline_pn:

    data = torch.load('..//Data//firefly-baseline-pn.pt')
    net, task, plant, algo, learning = data['net'], data['task'], data['plant'], data['algo'], data['learning']
    nrepeats, nconds = np.shape(net)
    tars, stops = np.empty((nrepeats, nconds, Nepochs, 4)), np.empty((nrepeats, nconds, Nepochs, 4))
    prob_correct, prob_shuffled = np.empty((nrepeats, nconds, 25)), np.empty((nrepeats, nconds, 25))
    auc, NTlist = np.empty((nrepeats, nconds)), np.empty((nrepeats, nconds, Nepochs))
    xa, va = np.empty((nrepeats, nconds, Nepochs, 600, 2)), np.empty((nrepeats, nconds, Nepochs, 600, 2))

    for i in range(nrepeats):
        for j in range(nconds):
            print('\n Net: ' + str(i+1) + '/' + str(nrepeats) + '\t Condition: ' + str(j+1) + '/' + str(nconds), end='')
            tars[i, j], stops[i, j], prob_correct[i, j], prob_shuffled[i, j], auc[i, j], NTlist[i, j], \
            _, _, _, va[i, j], xa[i, j], _ = test(net[i, j], task[i, j], plant[i, j], algo[i, j], learning[i, j],
                                                  feed_pos=False, feed_belief=False, Nepochs=Nepochs, seed=i)

    # save
    torch.save({'tars': tars, 'stops': stops, 'prob_correct': prob_correct, 'prob_shuffled': prob_shuffled, 'auc': auc,
                'NTlist': NTlist, 'va': va, 'xa': xa, }, '..//Data//firefly-baseline-pn-results.pt')

if test_baseline_sn:

    data = torch.load('..//Data//firefly-baseline-sn.pt')
    net, task, plant, algo, learning = data['net'], data['task'], data['plant'], data['algo'], data['learning']
    nrepeats, nconds = np.shape(net)
    tars, stops = np.empty((nrepeats, nconds, Nepochs, 4)), np.empty((nrepeats, nconds, Nepochs, 4))
    prob_correct, prob_shuffled = np.empty((nrepeats, nconds, 25)), np.empty((nrepeats, nconds, 25))
    auc, NTlist = np.empty((nrepeats, nconds)), np.empty((nrepeats, nconds, Nepochs))
    xa, va = np.empty((nrepeats, nconds, Nepochs, 600, 2)), np.empty((nrepeats, nconds, Nepochs, 600, 2))

    for i in range(nrepeats):
        for j in range(nconds):
            print('\n Net: ' + str(i+1) + '/' + str(nrepeats) + '\t Condition: ' + str(j+1) + '/' + str(nconds), end='')
            tars[i, j], stops[i, j], prob_correct[i, j], prob_shuffled[i, j], auc[i, j], NTlist[i, j], \
            _, _, _, va[i, j], xa[i, j], _ = test(net[i, j], task[i, j], plant[i, j], algo[i, j], learning[i, j],
                                                  feed_pos=False, feed_belief=False, Nepochs=Nepochs, seed=i)

    # save
    torch.save({'tars': tars, 'stops': stops, 'prob_correct': prob_correct, 'prob_shuffled': prob_shuffled, 'auc': auc,
                'NTlist': NTlist, 'va': va, 'xa': xa, }, '..//Data//firefly-baseline-sn-results.pt')

if test_baseline_db:

    data = torch.load('..//Data//firefly-baseline-db.pt')
    net, task, plant, algo, learning = data['net'], data['task'], data['plant'], data['algo'], data['learning']
    nrepeats, nconds = np.shape(net)
    tars, stops = np.empty((nrepeats, nconds, Nepochs, 4)), np.empty((nrepeats, nconds, Nepochs, 4))
    prob_correct, prob_shuffled = np.empty((nrepeats, nconds, 25)), np.empty((nrepeats, nconds, 25))
    auc, NTlist = np.empty((nrepeats, nconds)), np.empty((nrepeats, nconds, Nepochs))
    xa, va = np.empty((nrepeats, nconds, Nepochs, 600, 2)), np.empty((nrepeats, nconds, Nepochs, 600, 2))

    for i in range(nrepeats):
        for j in range(nconds):
            print('\n Net: ' + str(i+1) + '/' + str(nrepeats) + '\t Condition: ' + str(j+1) + '/' + str(nconds), end='')
            tars[i, j], stops[i, j], prob_correct[i, j], prob_shuffled[i, j], auc[i, j], NTlist[i, j], \
            _, _, _, va[i, j], xa[i, j], _ = test(net[i, j], task[i, j], plant[i, j], algo[i, j], learning[i, j],
                                                  feed_pos=False, feed_belief=False, Nepochs=Nepochs,
                                                  db=(1, 6), seed=i)

    # save
    torch.save({'tars': tars, 'stops': stops, 'prob_correct': prob_correct, 'prob_shuffled': prob_shuffled, 'auc': auc,
                'NTlist': NTlist, 'va': va, 'xa': xa, }, '..//Data//firefly-baseline-db-results.pt')

if test_feedpos_pn:

    data = torch.load('..//Data//firefly-feedpos-pn.pt')
    net, task, plant, algo, learning = data['net'], data['task'], data['plant'], data['algo'], data['learning']
    nrepeats, nconds = np.shape(net)
    tars, stops = np.empty((nrepeats, nconds, Nepochs, 4)), np.empty((nrepeats, nconds, Nepochs, 4))
    prob_correct, prob_shuffled = np.empty((nrepeats, nconds, 25)), np.empty((nrepeats, nconds, 25))
    auc, NTlist = np.empty((nrepeats, nconds)), np.empty((nrepeats, nconds, Nepochs))
    xa, va = np.empty((nrepeats, nconds, Nepochs, 600, 2)), np.empty((nrepeats, nconds, Nepochs, 600, 2))

    for i in range(nrepeats):
        for j in range(nconds):
            print('\n Net: ' + str(i+1) + '/' + str(nrepeats) + '\t Condition: ' + str(j+1) + '/' + str(nconds), end='')
            tars[i, j], stops[i, j], prob_correct[i, j], prob_shuffled[i, j], auc[i, j], NTlist[i, j], \
            _, _, _, va[i, j], xa[i, j], _ = test(net[i, j], task[i, j], plant[i, j], algo[i, j], learning[i, j],
                                                  feed_pos=True, feed_belief=False, Nepochs=Nepochs, seed=i)

    # save
    torch.save({'tars': tars, 'stops': stops, 'prob_correct': prob_correct, 'prob_shuffled': prob_shuffled, 'auc': auc,
                'NTlist': NTlist, 'va': va, 'xa': xa, }, '..//Data//firefly-feedpos-pn-results.pt')

if test_feedpos_sn:

    data = torch.load('..//Data//firefly-feedpos-sn.pt')
    net, task, plant, algo, learning = data['net'], data['task'], data['plant'], data['algo'], data['learning']
    nrepeats, nconds = np.shape(net)
    tars, stops = np.empty((nrepeats, nconds, Nepochs, 4)), np.empty((nrepeats, nconds, Nepochs, 4))
    prob_correct, prob_shuffled = np.empty((nrepeats, nconds, 25)), np.empty((nrepeats, nconds, 25))
    auc, NTlist = np.empty((nrepeats, nconds)), np.empty((nrepeats, nconds, Nepochs))
    xa, va = np.empty((nrepeats, nconds, Nepochs, 600, 2)), np.empty((nrepeats, nconds, Nepochs, 600, 2))

    for i in range(nrepeats):
        for j in range(nconds):
            print('\n Net: ' + str(i+1) + '/' + str(nrepeats) + '\t Condition: ' + str(j+1) + '/' + str(nconds), end='')
            tars[i, j], stops[i, j], prob_correct[i, j], prob_shuffled[i, j], auc[i, j], NTlist[i, j], \
            _, _, _, va[i, j], xa[i, j], _ = test(net[i, j], task[i, j], plant[i, j], algo[i, j], learning[i, j],
                                                  feed_pos=True, feed_belief=False, Nepochs=Nepochs, seed=i)

    # save
    torch.save({'tars': tars, 'stops': stops, 'prob_correct': prob_correct, 'prob_shuffled': prob_shuffled, 'auc': auc,
                'NTlist': NTlist, 'va': va, 'xa': xa, }, '..//Data//firefly-feedpos-sn-results.pt')

if test_feedpos_db:

    data = torch.load('..//Data//firefly-feedpos-db.pt')
    net, task, plant, algo, learning = data['net'], data['task'], data['plant'], data['algo'], data['learning']
    nrepeats, nconds = np.shape(net)
    tars, stops = np.empty((nrepeats, nconds, Nepochs, 4)), np.empty((nrepeats, nconds, Nepochs, 4))
    prob_correct, prob_shuffled = np.empty((nrepeats, nconds, 25)), np.empty((nrepeats, nconds, 25))
    auc, NTlist = np.empty((nrepeats, nconds)), np.empty((nrepeats, nconds, Nepochs))
    xa, va = np.empty((nrepeats, nconds, Nepochs, 600, 2)), np.empty((nrepeats, nconds, Nepochs, 600, 2))

    for i in range(nrepeats):
        for j in range(nconds):
            print('\n Net: ' + str(i+1) + '/' + str(nrepeats) + '\t Condition: ' + str(j+1) + '/' + str(nconds), end='')
            tars[i, j], stops[i, j], prob_correct[i, j], prob_shuffled[i, j], auc[i, j], NTlist[i, j], \
            _, _, _, va[i, j], xa[i, j], _ = test(net[i, j], task[i, j], plant[i, j], algo[i, j], learning[i, j],
                                                  feed_pos=True, feed_belief=False, Nepochs=Nepochs,
                                                  db=(1, 6), seed=i)

    # save
    torch.save({'tars': tars, 'stops': stops, 'prob_correct': prob_correct, 'prob_shuffled': prob_shuffled, 'auc': auc,
                'NTlist': NTlist, 'va': va, 'xa': xa, }, '..//Data//firefly-feedpos-db-results.pt')

if test_baseline_gaingen:

    data = torch.load('..//Data//firefly-baseline-db.pt')
    net, task, plant, algo, learning = data['net'], data['task'], data['plant'], data['algo'], data['learning']
    nrepeats, nconds = np.shape(net)
    tars, stops = np.empty((nrepeats, nconds, Nepochs, 4)), np.empty((nrepeats, nconds, Nepochs, 4))
    prob_correct, prob_shuffled = np.empty((nrepeats, nconds, 25)), np.empty((nrepeats, nconds, 25))
    auc, NTlist = np.empty((nrepeats, nconds)), np.empty((nrepeats, nconds, Nepochs))
    xa, va = np.empty((nrepeats, nconds, Nepochs, 600, 2)), np.empty((nrepeats, nconds, Nepochs, 600, 2))

    for i in range(nrepeats):
        for j in range(nconds):
            print('\n Net: ' + str(i+1) + '/' + str(nrepeats) + '\t Condition: ' + str(j+1) + '/' + str(nconds), end='')
            tars[i, j], stops[i, j], prob_correct[i, j], prob_shuffled[i, j], auc[i, j], NTlist[i, j], \
            _, _, _, va[i, j], xa[i, j], _ = test(net[i, j], task[i, j], plant[i, j], algo[i, j], learning[i, j],
                                                  feed_pos=False, feed_belief=False, Nepochs=Nepochs,
                                                  gain=2, seed=i)

    # save
    torch.save({'tars': tars, 'stops': stops, 'prob_correct': prob_correct, 'prob_shuffled': prob_shuffled, 'auc': auc,
                'NTlist': NTlist, 'va': va, 'xa': xa, }, '..//Data//firefly-baseline-gaingen-results.pt')

if test_feedpos_gaingen:

    data = torch.load('..//Data//firefly-feedpos-db.pt')
    net, task, plant, algo, learning = data['net'], data['task'], data['plant'], data['algo'], data['learning']
    nrepeats, nconds = np.shape(net)
    tars, stops = np.empty((nrepeats, nconds, Nepochs, 4)), np.empty((nrepeats, nconds, Nepochs, 4))
    prob_correct, prob_shuffled = np.empty((nrepeats, nconds, 25)), np.empty((nrepeats, nconds, 25))
    auc, NTlist = np.empty((nrepeats, nconds)), np.empty((nrepeats, nconds, Nepochs))
    xa, va = np.empty((nrepeats, nconds, Nepochs, 600, 2)), np.empty((nrepeats, nconds, Nepochs, 600, 2))

    for i in range(nrepeats):
        for j in range(nconds):
            print('\n Net: ' + str(i+1) + '/' + str(nrepeats) + '\t Condition: ' + str(j+1) + '/' + str(nconds), end='')
            tars[i, j], stops[i, j], prob_correct[i, j], prob_shuffled[i, j], auc[i, j], NTlist[i, j], \
            _, _, _, va[i, j], xa[i, j], _ = test(net[i, j], task[i, j], plant[i, j], algo[i, j], learning[i, j],
                                                  feed_pos=True, feed_belief=False, Nepochs=Nepochs,
                                                  gain=2, seed=i)

    # save
    torch.save({'tars': tars, 'stops': stops, 'prob_correct': prob_correct, 'prob_shuffled': prob_shuffled, 'auc': auc,
                'NTlist': NTlist, 'va': va, 'xa': xa, }, '..//Data//firefly-feedpos-gaingen-results.pt')

if test_baseline_sensorygen:

    data = torch.load('..//Data//firefly-baseline-db.pt')
    net, task, plant, algo, learning = data['net'], data['task'], data['plant'], data['algo'], data['learning']
    nrepeats, nconds = np.shape(net)
    tars, stops = np.empty((nrepeats, nconds, Nepochs, 4)), np.empty((nrepeats, nconds, Nepochs, 4))
    prob_correct, prob_shuffled = np.empty((nrepeats, nconds, 25)), np.empty((nrepeats, nconds, 25))
    auc, NTlist = np.empty((nrepeats, nconds)), np.empty((nrepeats, nconds, Nepochs))
    xa, va = np.empty((nrepeats, nconds, Nepochs, 600, 2)), np.empty((nrepeats, nconds, Nepochs, 600, 2))

    for i in range(nrepeats):
        for j in range(nconds):
            print('\n Net: ' + str(i+1) + '/' + str(nrepeats) + '\t Condition: ' + str(j+1) + '/' + str(nconds), end='')
            tars[i, j], stops[i, j], prob_correct[i, j], prob_shuffled[i, j], auc[i, j], NTlist[i, j], \
            _, _, _, va[i, j], xa[i, j], _ = test(net[i, j], task[i, j], plant[i, j], algo[i, j], learning[i, j],
                                                  feed_pos=False, feed_belief=False, Nepochs=Nepochs,
                                                  sn=5, seed=i)

    # save
    torch.save({'tars': tars, 'stops': stops, 'prob_correct': prob_correct, 'prob_shuffled': prob_shuffled, 'auc': auc,
                'NTlist': NTlist, 'va': va, 'xa': xa, }, '..//Data//firefly-baseline-sensorygen-results.pt')

if test_feedpos_sensorygen:

    data = torch.load('..//Data//firefly-feedpos-db.pt')
    net, task, plant, algo, learning = data['net'], data['task'], data['plant'], data['algo'], data['learning']
    nrepeats, nconds = np.shape(net)
    tars, stops = np.empty((nrepeats, nconds, Nepochs, 4)), np.empty((nrepeats, nconds, Nepochs, 4))
    prob_correct, prob_shuffled = np.empty((nrepeats, nconds, 25)), np.empty((nrepeats, nconds, 25))
    auc, NTlist = np.empty((nrepeats, nconds)), np.empty((nrepeats, nconds, Nepochs))
    xa, va = np.empty((nrepeats, nconds, Nepochs, 600, 2)), np.empty((nrepeats, nconds, Nepochs, 600, 2))

    for i in range(nrepeats):
        for j in range(nconds):
            print('\n Net: ' + str(i+1) + '/' + str(nrepeats) + '\t Condition: ' + str(j+1) + '/' + str(nconds), end='')
            tars[i, j], stops[i, j], prob_correct[i, j], prob_shuffled[i, j], auc[i, j], NTlist[i, j], \
            _, _, _, va[i, j], xa[i, j], _ = test(net[i, j], task[i, j], plant[i, j], algo[i, j], learning[i, j],
                                                  feed_pos=True, feed_belief=False, Nepochs=Nepochs,
                                                  sn=5, seed=i)

    # save
    torch.save({'tars': tars, 'stops': stops, 'prob_correct': prob_correct, 'prob_shuffled': prob_shuffled, 'auc': auc,
                'NTlist': NTlist, 'va': va, 'xa': xa, }, '..//Data//firefly-feedpos-sensorygen-results.pt')

if test_baseline_processgen:

    data = torch.load('..//Data//firefly-baseline-db.pt')
    net, task, plant, algo, learning = data['net'], data['task'], data['plant'], data['algo'], data['learning']
    nrepeats, nconds = np.shape(net)
    tars, stops = np.empty((nrepeats, nconds, Nepochs, 4)), np.empty((nrepeats, nconds, Nepochs, 4))
    prob_correct, prob_shuffled = np.empty((nrepeats, nconds, 25)), np.empty((nrepeats, nconds, 25))
    auc, NTlist = np.empty((nrepeats, nconds)), np.empty((nrepeats, nconds, Nepochs))
    xa, va = np.empty((nrepeats, nconds, Nepochs, 600, 2)), np.empty((nrepeats, nconds, Nepochs, 600, 2))

    for i in range(nrepeats):
        for j in range(nconds):
            print('\n Net: ' + str(i + 1) + '/' + str(nrepeats) + '\t Condition: ' + str(j + 1) + '/' + str(nconds),
                  end='')
            tars[i, j], stops[i, j], prob_correct[i, j], prob_shuffled[i, j], auc[i, j], NTlist[i, j], \
            _, _, _, va[i, j], xa[i, j], _ = test(net[i, j], task[i, j], plant[i, j], algo[i, j], learning[i, j],
                                                  feed_pos=False, feed_belief=False, Nepochs=Nepochs,
                                                  pn=1, seed=i)

    # save
    torch.save({'tars': tars, 'stops': stops, 'prob_correct': prob_correct, 'prob_shuffled': prob_shuffled, 'auc': auc,
                'NTlist': NTlist, 'va': va, 'xa': xa, }, '..//Data//firefly-baseline-processgen-results.pt')

if test_feedpos_processgen:

    data = torch.load('..//Data//firefly-feedpos-db.pt')
    net, task, plant, algo, learning = data['net'], data['task'], data['plant'], data['algo'], data['learning']
    nrepeats, nconds = np.shape(net)
    tars, stops = np.empty((nrepeats, nconds, Nepochs, 4)), np.empty((nrepeats, nconds, Nepochs, 4))
    prob_correct, prob_shuffled = np.empty((nrepeats, nconds, 25)), np.empty((nrepeats, nconds, 25))
    auc, NTlist = np.empty((nrepeats, nconds)), np.empty((nrepeats, nconds, Nepochs))
    xa, va = np.empty((nrepeats, nconds, Nepochs, 600, 2)), np.empty((nrepeats, nconds, Nepochs, 600, 2))

    for i in range(nrepeats):
        for j in range(nconds):
            print('\n Net: ' + str(i + 1) + '/' + str(nrepeats) + '\t Condition: ' + str(j + 1) + '/' + str(nconds),
                  end='')
            tars[i, j], stops[i, j], prob_correct[i, j], prob_shuffled[i, j], auc[i, j], NTlist[i, j], \
            _, _, _, va[i, j], xa[i, j], _ = test(net[i, j], task[i, j], plant[i, j], algo[i, j], learning[i, j],
                                                  feed_pos=True, feed_belief=False, Nepochs=Nepochs,
                                                  pn=1, seed=i)

    # save
    torch.save({'tars': tars, 'stops': stops, 'prob_correct': prob_correct, 'prob_shuffled': prob_shuffled, 'auc': auc,
                'NTlist': NTlist, 'va': va, 'xa': xa, }, '..//Data//firefly-feedpos-processgen-results.pt')

if plot_behavior:
    data = {'baseline': torch.load('..//Data//firefly-baseline-pn-results.pt'),
            'feedpos': torch.load('..//Data//firefly-feedpos-pn-results.pt')}
    plot(data, 'behavior')

if plot_processnoise:
    data = {'baseline': torch.load('..//Data//firefly-baseline-pn-results.pt'),
            'feedpos': torch.load('..//Data//firefly-feedpos-pn-results.pt')}
    plot(data, 'processnoise')

if plot_sensorynoise:
    data = {'baseline': torch.load('..//Data//firefly-baseline-sn-results.pt'),
            'feedpos': torch.load('..//Data//firefly-feedpos-sn-results.pt')}
    plot(data, 'sensorynoise')

if plot_distgen:
    data = {'baseline': torch.load('..//Data//firefly-baseline-db-results.pt'),
            'feedpos': torch.load('..//Data//firefly-feedpos-db-results.pt')}
    plot(data, 'distgen')

if plot_gaingen:
    data = {'baseline_train': torch.load('..//Data//firefly-baseline-db-results.pt'),
            'baseline_test': torch.load('..//Data//firefly-baseline-gaingen-results.pt'),
            'feedpos_train': torch.load('..//Data//firefly-feedpos-db-results.pt'),
            'feedpos_test': torch.load('..//Data//firefly-feedpos-gaingen-results.pt')}
    plot(data, 'gaingen')

if plot_processgen:
    data = {'baseline_train': torch.load('..//Data//firefly-baseline-db-results.pt'),
            'baseline_test': torch.load('..//Data//firefly-baseline-processgen-results.pt'),
            'feedpos_train': torch.load('..//Data//firefly-feedpos-db-results.pt'),
            'feedpos_test': torch.load('..//Data//firefly-feedpos-processgen-results.pt')}
    plot(data, 'processgen')

if plot_sensorygen:
    data = {'baseline_train': torch.load('..//Data//firefly-baseline-db-results.pt'),
            'baseline_test': torch.load('..//Data//firefly-baseline-sensorygen-results.pt'),
            'feedpos_train': torch.load('..//Data//firefly-feedpos-db-results.pt'),
            'feedpos_test': torch.load('..//Data//firefly-feedpos-sensorygen-results.pt')}
    plot(data, 'sensorygen')
