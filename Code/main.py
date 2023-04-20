from train import train
from test import test
from plot import plot
import numpy as np
import torch
from matplotlib import pyplot as plt
from model import Network, Task, Algorithm
from sklearn.linear_model import LinearRegression
import seaborn
import pandas as pd
import scipy.io

do_training, do_testing, do_plotting = True, True, True
Nepochs = 500

# test_baseline_pn, test_baseline_sn, test_baseline_db, test_feedpos_pn, test_feedpos_sn, test_feedpos_db, \
# test_feedbelief_pn, test_feedbelief_sn, test_feedbelief_db = False, False, False, False, False, False, False, False, False
#
# test_baseline_gaingen, test_baseline_processgen, test_baseline_sensorygen, \
# test_feedpos_gaingen, test_feedpos_processgen, test_feedpos_sensorygen = False, False, False, False, False, False
#
# plot_behavior = False
# plot_processnoise, plot_sensorynoise = False, False
# plot_distgen, plot_gaingen, plot_processgen, plot_sensorygen = False, False, False, False
# plot_iocurve = False

if do_training:
    net, task, algo, plant, learning = \
        train(arch='ppc', N=100, S=4, R=2, task='firefly', db=(1, 6), sn=10, pn=0.2,
              feed_pos=False, feed_belief=False, learningsites=({'wr'}), algo='Adam', Nepochs=10000, lr=1e-4, seed=1)
    # save
    torch.save({'net': net, 'task': task, 'algo': algo, 'plant': plant, 'learning': learning},
               '..//Data//m1-train.pt')

    # net, task, algo, plant, learning = \
    #     train(arch='ppc', N=100, S=4, R=2, task='firefly', db=(1, 6), sn=10, pn=0.2,
    #           feed_pos=False, feed_belief=False, learningsites=({'wr', 'J'}), algo='Adam', Nepochs=10000, lr=1e-4, seed=1)
    # # save
    # torch.save({'net': net, 'task': task, 'algo': algo, 'plant': plant, 'learning': learning},
    #            '..//Data//m2-train.pt')

    # net, task, algo, plant, learning = \
    #     train(arch='ppc', N=100, S=4, R=2, task='firefly', db=(1, 6), sn=10, pn=0.2,
    #           feed_pos=False, feed_belief=True, learningsites=({'wr'}), algo='Adam', Nepochs=10000, lr=1e-4, seed=1)
    # # save
    # torch.save({'net': net, 'task': task, 'algo': algo, 'plant': plant, 'learning': learning},
    #            '..//Data//m3-train.pt')

    # net, task, algo, plant, learning = \
    #     train(arch='ppc', N=100, S=4, R=2, task='firefly', db=(1, 6), sn=10, pn=0.2,
    #           feed_pos=False, feed_belief=True, learningsites=({'wr', 'J'}), algo='Adam', Nepochs=10000, lr=1e-4, seed=1)
    # # save
    # torch.save({'net': net, 'task': task, 'algo': algo, 'plant': plant, 'learning': learning},
    #            '..//Data//m4-train.pt')

if do_testing:
    data = torch.load('..//Data//m1-train.pt')
    net, task, plant, algo, learning = data['net'], data['task'], data['plant'], data['algo'], data['learning']
    tars, stops, prob_correct, prob_shuffled, auc, NTlist, sa, da, ha, ua, va, xa, ba = \
        test(net, task, plant, algo, learning, feed_pos=False, feed_belief=False, Nepochs=Nepochs, seed=1)
    torch.save({'tars': tars, 'stops': stops, 'prob_correct': prob_correct, 'prob_shuffled': prob_shuffled, 'auc': auc,
                'NTlist': NTlist, 'da': da, 'ha': ha, 'ua': ua, 'va': va, 'xa': xa}, '..//Data//m1-test.pt')

if do_plotting:
    data = torch.load('..//Data//m1-test.pt')
    NTlist, xa, tars, stops = data['NTlist'], data['xa'], data['tars'], data['stops']
    plt.subplot(1, 3, 1)
    for ei in range(Nepochs):
        plt.plot(xa[ei, :int(NTlist[ei]), 0], xa[ei, :int(NTlist[ei]), 1], color='black', linewidth=0.2)
    plt.xlim((-3, 3)), plt.ylim((0, 6))
    plt.subplot(1, 3, 2)
    plt.plot(np.array(tars)[:, 2], np.array(stops)[:, 2], '.')
    plt.xlim((0, 6)), plt.ylim((0, 6))
    plt.subplot(1, 3, 3)
    plt.plot(np.array(tars)[:, 3], np.array(stops)[:, 3], '.')
    plt.xlim((-.6, .6)), plt.ylim((-.6, .6))

    data = torch.load('..//Data//m1-train.pt')
    plt.figure()
    # plot learning curve (moving average over 20 trials)
    plt.plot(np.convolve(np.array(data['learning']['mses']).flatten(), np.ones(20) / 20, mode='same'))
    plt.yscale('log')


#
# tx = np.array(tars)[:, 1]
# ty = np.array(tars)[:, 2]
# bx = [np.array(ua.detach().numpy())[idx, :, 2].mean(axis=0) for idx in range(100)]
# by = [np.array(ua.detach().numpy())[idx, :, 3].mean(axis=0) for idx in range(100)]
#
# # regression
# xb = [np.array(ua.detach().numpy())[idx, 30:NTlist[idx], 2] for idx in range(100)]
# yb = [np.array(ua.detach().numpy())[idx, 30:NTlist[idx], 3] for idx in range(100)]
# xt = [tars[idx][0] - np.array(xa)[idx, 30:NTlist[idx], 0] for idx in range(100)]
# yt = [tars[idx][1] - np.array(xa)[idx, 30:NTlist[idx], 1] for idx in range(100)]
# xs = [stops[idx][0] - np.array(xa)[idx, 30:NTlist[idx], 0] for idx in range(100)]
# ys = [stops[idx][1] - np.array(xa)[idx, 30:NTlist[idx], 1] for idx in range(100)]
#
# xb1, yb1, xt1, yt1, xs1, ys1 = [], [], [], [], [], []
# for idx in range(100):
#     nt = len(xb[idx])
#     x = np.linspace(0, nt-1, num=50)
#     xb1.append(np.interp(x, range(nt), xb[idx]))
#     yb1.append(np.interp(x, range(nt), yb[idx]))
#     xt1.append(np.interp(x, range(nt), xt[idx]))
#     yt1.append(np.interp(x, range(nt), yt[idx]))
#     xs1.append(np.interp(x, range(nt), xs[idx]))
#     ys1.append(np.interp(x, range(nt), ys[idx]))
#
# reg = LinearRegression(fit_intercept=True)
# betas_t, betas_s = [], []
# for nboots in range(100):
#     trials = np.random.choice(100, 100)
#     beta_x, beta_y = [], []
#     for idx in range(50):
#         reg.fit(np.vstack((np.array(xt1)[trials, idx], np.array(xs1)[trials, idx])).T, np.array(xb1)[trials, idx])
#         beta_x.append(reg.coef_)
#         reg.fit(np.vstack((np.array(yt1)[trials, idx], np.array(ys1)[trials, idx])).T, np.array(yb1)[trials, idx])
#         beta_y.append(reg.coef_)
#     beta_x, beta_y = np.array(beta_x), np.array(beta_y)
#     beta_t, beta_s = np.abs(beta_x[:, 0]), np.abs(beta_y[:, 1])
#     beta_t = beta_t / np.sqrt(beta_t[0] ** 2 + beta_s[0] ** 2)
#     beta_s = beta_s / np.sqrt(beta_t[48] ** 2 + beta_s[48] ** 2)
#     betas_t.append(beta_t), betas_s.append(beta_s)
#
# plt.plot(range(50), np.mean(np.array(betas_t), axis=0), color='cyan')
# plt.plot(range(50), np.mean(np.array(betas_s), axis=0), color='blue')
# plt.fill_between(range(50), np.mean(np.array(betas_t), axis=0) - np.std(np.array(betas_t), axis=0),
#                  np.mean(np.array(betas_t), axis=0) + np.std(np.array(betas_t), axis=0), color='cyan', alpha=0.2)
# plt.fill_between(range(50), np.mean(np.array(betas_s), axis=0) - np.std(np.array(betas_s), axis=0),
#                  np.mean(np.array(betas_s), axis=0) + np.std(np.array(betas_s), axis=0), color='blue', alpha=0.8)
# plt.xlim(0, 48)

# if test_baseline_pn:
#
#     data = torch.load('..//Data//firefly-baseline-pn.pt')
#     net, task, plant, algo, learning = data['net'], data['task'], data['plant'], data['algo'], data['learning']
#     nrepeats, nconds = np.shape(net)
#     tars, stops = np.empty((nrepeats, nconds, Nepochs, 4)), np.empty((nrepeats, nconds, Nepochs, 4))
#     prob_correct, prob_shuffled = np.empty((nrepeats, nconds, 25)), np.empty((nrepeats, nconds, 25))
#     auc, NTlist = np.empty((nrepeats, nconds)), np.empty((nrepeats, nconds, Nepochs))
#     xa, va = np.empty((nrepeats, nconds, Nepochs, 600, 2)), np.empty((nrepeats, nconds, Nepochs, 600, 2))
#
#     for i in range(nrepeats):
#         for j in range(nconds):
#             print('\n Net: ' + str(i+1) + '/' + str(nrepeats) + '\t Condition: ' + str(j+1) + '/' + str(nconds), end='')
#             tars[i, j], stops[i, j], prob_correct[i, j], prob_shuffled[i, j], auc[i, j], NTlist[i, j], \
#             _, _, _, va[i, j], xa[i, j], _ = test(net[i, j], task[i, j], plant[i, j], algo[i, j], learning[i, j],
#                                                   feed_pos=False, feed_belief=False, Nepochs=Nepochs, seed=i)
#
#     # save
#     torch.save({'tars': tars, 'stops': stops, 'prob_correct': prob_correct, 'prob_shuffled': prob_shuffled, 'auc': auc,
#                 'NTlist': NTlist, 'va': va, 'xa': xa, }, '..//Data//firefly-baseline-pn-results.pt')
#
# if test_baseline_sn:
#
#     data = torch.load('..//Data//firefly-baseline-sn.pt')
#     net, task, plant, algo, learning = data['net'], data['task'], data['plant'], data['algo'], data['learning']
#     nrepeats, nconds = np.shape(net)
#     tars, stops = np.empty((nrepeats, nconds, Nepochs, 4)), np.empty((nrepeats, nconds, Nepochs, 4))
#     prob_correct, prob_shuffled = np.empty((nrepeats, nconds, 25)), np.empty((nrepeats, nconds, 25))
#     auc, NTlist = np.empty((nrepeats, nconds)), np.empty((nrepeats, nconds, Nepochs))
#     xa, va = np.empty((nrepeats, nconds, Nepochs, 600, 2)), np.empty((nrepeats, nconds, Nepochs, 600, 2))
#
#     for i in range(nrepeats):
#         for j in range(nconds):
#             print('\n Net: ' + str(i+1) + '/' + str(nrepeats) + '\t Condition: ' + str(j+1) + '/' + str(nconds), end='')
#             tars[i, j], stops[i, j], prob_correct[i, j], prob_shuffled[i, j], auc[i, j], NTlist[i, j], \
#             _, _, _, va[i, j], xa[i, j], _ = test(net[i, j], task[i, j], plant[i, j], algo[i, j], learning[i, j],
#                                                   feed_pos=False, feed_belief=False, Nepochs=Nepochs, seed=i)
#
#     # save
#     torch.save({'tars': tars, 'stops': stops, 'prob_correct': prob_correct, 'prob_shuffled': prob_shuffled, 'auc': auc,
#                 'NTlist': NTlist, 'va': va, 'xa': xa, }, '..//Data//firefly-baseline-sn-results.pt')
#
# if test_baseline_db:
#
#     data = torch.load('..//Data//firefly-baseline-db.pt')
#     net, task, plant, algo, learning = data['net'], data['task'], data['plant'], data['algo'], data['learning']
#     nrepeats, nconds = np.shape(net)
#     tars, stops = np.empty((nrepeats, nconds, Nepochs, 4)), np.empty((nrepeats, nconds, Nepochs, 4))
#     prob_correct, prob_shuffled = np.empty((nrepeats, nconds, 25)), np.empty((nrepeats, nconds, 25))
#     auc, NTlist = np.empty((nrepeats, nconds)), np.empty((nrepeats, nconds, Nepochs))
#     xa, va = np.empty((nrepeats, nconds, Nepochs, 600, 2)), np.empty((nrepeats, nconds, Nepochs, 600, 2))
#
#     for i in range(nrepeats):
#         for j in range(nconds):
#             print('\n Net: ' + str(i+1) + '/' + str(nrepeats) + '\t Condition: ' + str(j+1) + '/' + str(nconds), end='')
#             tars[i, j], stops[i, j], prob_correct[i, j], prob_shuffled[i, j], auc[i, j], NTlist[i, j], \
#             _, _, _, va[i, j], xa[i, j], _ = test(net[i, j], task[i, j], plant[i, j], algo[i, j], learning[i, j],
#                                                   feed_pos=False, feed_belief=False, Nepochs=Nepochs,
#                                                   db=(1, 6), seed=i)
#
#     # save
#     torch.save({'tars': tars, 'stops': stops, 'prob_correct': prob_correct, 'prob_shuffled': prob_shuffled, 'auc': auc,
#                 'NTlist': NTlist, 'va': va, 'xa': xa, }, '..//Data//firefly-baseline-db-results.pt')
#
# if test_feedpos_pn:
#
#     data = torch.load('..//Data//firefly-feedpos-pn.pt')
#     net, task, plant, algo, learning = data['net'], data['task'], data['plant'], data['algo'], data['learning']
#     nrepeats, nconds = np.shape(net)
#     tars, stops = np.empty((nrepeats, nconds, Nepochs, 4)), np.empty((nrepeats, nconds, Nepochs, 4))
#     prob_correct, prob_shuffled = np.empty((nrepeats, nconds, 25)), np.empty((nrepeats, nconds, 25))
#     auc, NTlist = np.empty((nrepeats, nconds)), np.empty((nrepeats, nconds, Nepochs))
#     xa, va = np.empty((nrepeats, nconds, Nepochs, 600, 2)), np.empty((nrepeats, nconds, Nepochs, 600, 2))
#
#     for i in range(nrepeats):
#         for j in range(nconds):
#             print('\n Net: ' + str(i+1) + '/' + str(nrepeats) + '\t Condition: ' + str(j+1) + '/' + str(nconds), end='')
#             tars[i, j], stops[i, j], prob_correct[i, j], prob_shuffled[i, j], auc[i, j], NTlist[i, j], \
#             _, _, _, va[i, j], xa[i, j], _ = test(net[i, j], task[i, j], plant[i, j], algo[i, j], learning[i, j],
#                                                   feed_pos=True, feed_belief=False, Nepochs=Nepochs, seed=i)
#
#     # save
#     torch.save({'tars': tars, 'stops': stops, 'prob_correct': prob_correct, 'prob_shuffled': prob_shuffled, 'auc': auc,
#                 'NTlist': NTlist, 'va': va, 'xa': xa, }, '..//Data//firefly-feedpos-pn-results.pt')
#
# if test_feedpos_sn:
#
#     data = torch.load('..//Data//firefly-feedpos-sn.pt')
#     net, task, plant, algo, learning = data['net'], data['task'], data['plant'], data['algo'], data['learning']
#     nrepeats, nconds = np.shape(net)
#     tars, stops = np.empty((nrepeats, nconds, Nepochs, 4)), np.empty((nrepeats, nconds, Nepochs, 4))
#     prob_correct, prob_shuffled = np.empty((nrepeats, nconds, 25)), np.empty((nrepeats, nconds, 25))
#     auc, NTlist = np.empty((nrepeats, nconds)), np.empty((nrepeats, nconds, Nepochs))
#     xa, va = np.empty((nrepeats, nconds, Nepochs, 600, 2)), np.empty((nrepeats, nconds, Nepochs, 600, 2))
#
#     for i in range(nrepeats):
#         for j in range(nconds):
#             print('\n Net: ' + str(i+1) + '/' + str(nrepeats) + '\t Condition: ' + str(j+1) + '/' + str(nconds), end='')
#             tars[i, j], stops[i, j], prob_correct[i, j], prob_shuffled[i, j], auc[i, j], NTlist[i, j], \
#             _, _, _, va[i, j], xa[i, j], _ = test(net[i, j], task[i, j], plant[i, j], algo[i, j], learning[i, j],
#                                                   feed_pos=True, feed_belief=False, Nepochs=Nepochs, seed=i)
#
#     # save
#     torch.save({'tars': tars, 'stops': stops, 'prob_correct': prob_correct, 'prob_shuffled': prob_shuffled, 'auc': auc,
#                 'NTlist': NTlist, 'va': va, 'xa': xa, }, '..//Data//firefly-feedpos-sn-results.pt')
#
# if test_feedpos_db:
#
#     data = torch.load('..//Data//firefly-feedpos-db.pt')
#     net, task, plant, algo, learning = data['net'], data['task'], data['plant'], data['algo'], data['learning']
#     nrepeats, nconds = np.shape(net)
#     tars, stops = np.empty((nrepeats, nconds, Nepochs, 4)), np.empty((nrepeats, nconds, Nepochs, 4))
#     prob_correct, prob_shuffled = np.empty((nrepeats, nconds, 25)), np.empty((nrepeats, nconds, 25))
#     auc, NTlist = np.empty((nrepeats, nconds)), np.empty((nrepeats, nconds, Nepochs))
#     xa, va = np.empty((nrepeats, nconds, Nepochs, 600, 2)), np.empty((nrepeats, nconds, Nepochs, 600, 2))
#
#     for i in range(nrepeats):
#         for j in range(nconds):
#             print('\n Net: ' + str(i+1) + '/' + str(nrepeats) + '\t Condition: ' + str(j+1) + '/' + str(nconds), end='')
#             tars[i, j], stops[i, j], prob_correct[i, j], prob_shuffled[i, j], auc[i, j], NTlist[i, j], \
#             _, _, _, va[i, j], xa[i, j], _ = test(net[i, j], task[i, j], plant[i, j], algo[i, j], learning[i, j],
#                                                   feed_pos=True, feed_belief=False, Nepochs=Nepochs,
#                                                   db=(1, 6), seed=i)
#
#     # save
#     torch.save({'tars': tars, 'stops': stops, 'prob_correct': prob_correct, 'prob_shuffled': prob_shuffled, 'auc': auc,
#                 'NTlist': NTlist, 'va': va, 'xa': xa, }, '..//Data//firefly-feedpos-db-results.pt')
#
# if test_feedbelief_pn:
#
#     data = torch.load('..//Data//firefly-feedbelief-pn.pt')
#     net, task, plant, algo, learning = data['net'], data['task'], data['plant'], data['algo'], data['learning']
#     nrepeats, nconds = np.shape(net)
#     tars, stops = np.empty((nrepeats, nconds, Nepochs, 4)), np.empty((nrepeats, nconds, Nepochs, 4))
#     prob_correct, prob_shuffled = np.empty((nrepeats, nconds, 25)), np.empty((nrepeats, nconds, 25))
#     auc, NTlist = np.empty((nrepeats, nconds)), np.empty((nrepeats, nconds, Nepochs))
#     xa, va = np.empty((nrepeats, nconds, Nepochs, 600, 2)), np.empty((nrepeats, nconds, Nepochs, 600, 2))
#
#     for i in range(nrepeats):
#         for j in range(nconds):
#             print('\n Net: ' + str(i+1) + '/' + str(nrepeats) + '\t Condition: ' + str(j+1) + '/' + str(nconds), end='')
#             tars[i, j], stops[i, j], prob_correct[i, j], prob_shuffled[i, j], auc[i, j], NTlist[i, j], \
#             _, _, _, va[i, j], xa[i, j], _ = test(net[i, j], task[i, j], plant[i, j], algo[i, j], learning[i, j],
#                                                   feed_pos=False, feed_belief=True, Nepochs=Nepochs, seed=i)
#
#     # save
#     torch.save({'tars': tars, 'stops': stops, 'prob_correct': prob_correct, 'prob_shuffled': prob_shuffled, 'auc': auc,
#                 'NTlist': NTlist, 'va': va, 'xa': xa, }, '..//Data//firefly-feedbelief-pn-results.pt')
#
# if test_feedbelief_sn:
#
#     data = torch.load('..//Data//firefly-feedbelief-sn.pt')
#     net, task, plant, algo, learning = data['net'], data['task'], data['plant'], data['algo'], data['learning']
#     nrepeats, nconds = np.shape(net)
#     tars, stops = np.empty((nrepeats, nconds, Nepochs, 4)), np.empty((nrepeats, nconds, Nepochs, 4))
#     prob_correct, prob_shuffled = np.empty((nrepeats, nconds, 25)), np.empty((nrepeats, nconds, 25))
#     auc, NTlist = np.empty((nrepeats, nconds)), np.empty((nrepeats, nconds, Nepochs))
#     xa, va = np.empty((nrepeats, nconds, Nepochs, 600, 2)), np.empty((nrepeats, nconds, Nepochs, 600, 2))
#
#     for i in range(nrepeats):
#         for j in range(nconds):
#             print('\n Net: ' + str(i+1) + '/' + str(nrepeats) + '\t Condition: ' + str(j+1) + '/' + str(nconds), end='')
#             tars[i, j], stops[i, j], prob_correct[i, j], prob_shuffled[i, j], auc[i, j], NTlist[i, j], \
#             _, _, _, va[i, j], xa[i, j], _ = test(net[i, j], task[i, j], plant[i, j], algo[i, j], learning[i, j],
#                                                   feed_pos=False, feed_belief=True, Nepochs=Nepochs, seed=i)
#
#     # save
#     torch.save({'tars': tars, 'stops': stops, 'prob_correct': prob_correct, 'prob_shuffled': prob_shuffled, 'auc': auc,
#                 'NTlist': NTlist, 'va': va, 'xa': xa, }, '..//Data//firefly-feedbelief-sn-results.pt')
#
# if test_feedbelief_db:
#
#     data = torch.load('..//Data//firefly-feedbelief-db.pt')
#     net, task, plant, algo, learning = data['net'], data['task'], data['plant'], data['algo'], data['learning']
#     nrepeats, nconds = np.shape(net)
#     tars, stops = np.empty((nrepeats, nconds, Nepochs, 4)), np.empty((nrepeats, nconds, Nepochs, 4))
#     prob_correct, prob_shuffled = np.empty((nrepeats, nconds, 25)), np.empty((nrepeats, nconds, 25))
#     auc, NTlist = np.empty((nrepeats, nconds)), np.empty((nrepeats, nconds, Nepochs))
#     xa, va = np.empty((nrepeats, nconds, Nepochs, 600, 2)), np.empty((nrepeats, nconds, Nepochs, 600, 2))
#
#     for i in range(nrepeats):
#         for j in range(nconds):
#             print('\n Net: ' + str(i+1) + '/' + str(nrepeats) + '\t Condition: ' + str(j+1) + '/' + str(nconds), end='')
#             tars[i, j], stops[i, j], prob_correct[i, j], prob_shuffled[i, j], auc[i, j], NTlist[i, j], \
#             _, _, _, va[i, j], xa[i, j], _ = test(net[i, j], task[i, j], plant[i, j], algo[i, j], learning[i, j],
#                                                   feed_pos=False, feed_belief=True, Nepochs=Nepochs,
#                                                   db=(1, 6), seed=i)
#
#     # save
#     torch.save({'tars': tars, 'stops': stops, 'prob_correct': prob_correct, 'prob_shuffled': prob_shuffled, 'auc': auc,
#                 'NTlist': NTlist, 'va': va, 'xa': xa, }, '..//Data//firefly-feedbelief-db-results.pt')
#
# if test_baseline_gaingen:
#
#     data = torch.load('..//Data//firefly-baseline-db.pt')
#     net, task, plant, algo, learning = data['net'], data['task'], data['plant'], data['algo'], data['learning']
#     nrepeats, nconds = np.shape(net)
#     tars, stops = np.empty((nrepeats, nconds, Nepochs, 4)), np.empty((nrepeats, nconds, Nepochs, 4))
#     prob_correct, prob_shuffled = np.empty((nrepeats, nconds, 25)), np.empty((nrepeats, nconds, 25))
#     auc, NTlist = np.empty((nrepeats, nconds)), np.empty((nrepeats, nconds, Nepochs))
#     xa, va = np.empty((nrepeats, nconds, Nepochs, 600, 2)), np.empty((nrepeats, nconds, Nepochs, 600, 2))
#
#     for i in range(nrepeats):
#         for j in range(nconds):
#             print('\n Net: ' + str(i+1) + '/' + str(nrepeats) + '\t Condition: ' + str(j+1) + '/' + str(nconds), end='')
#             tars[i, j], stops[i, j], prob_correct[i, j], prob_shuffled[i, j], auc[i, j], NTlist[i, j], \
#             _, _, _, va[i, j], xa[i, j], _ = test(net[i, j], task[i, j], plant[i, j], algo[i, j], learning[i, j],
#                                                   feed_pos=False, feed_belief=False, Nepochs=Nepochs,
#                                                   gain=2, seed=i)
#
#     # save
#     torch.save({'tars': tars, 'stops': stops, 'prob_correct': prob_correct, 'prob_shuffled': prob_shuffled, 'auc': auc,
#                 'NTlist': NTlist, 'va': va, 'xa': xa, }, '..//Data//firefly-baseline-gaingen-results.pt')
#
# if test_feedpos_gaingen:
#
#     data = torch.load('..//Data//firefly-feedpos-db.pt')
#     net, task, plant, algo, learning = data['net'], data['task'], data['plant'], data['algo'], data['learning']
#     nrepeats, nconds = np.shape(net)
#     tars, stops = np.empty((nrepeats, nconds, Nepochs, 4)), np.empty((nrepeats, nconds, Nepochs, 4))
#     prob_correct, prob_shuffled = np.empty((nrepeats, nconds, 25)), np.empty((nrepeats, nconds, 25))
#     auc, NTlist = np.empty((nrepeats, nconds)), np.empty((nrepeats, nconds, Nepochs))
#     xa, va = np.empty((nrepeats, nconds, Nepochs, 600, 2)), np.empty((nrepeats, nconds, Nepochs, 600, 2))
#
#     for i in range(nrepeats):
#         for j in range(nconds):
#             print('\n Net: ' + str(i+1) + '/' + str(nrepeats) + '\t Condition: ' + str(j+1) + '/' + str(nconds), end='')
#             tars[i, j], stops[i, j], prob_correct[i, j], prob_shuffled[i, j], auc[i, j], NTlist[i, j], \
#             _, _, _, va[i, j], xa[i, j], _ = test(net[i, j], task[i, j], plant[i, j], algo[i, j], learning[i, j],
#                                                   feed_pos=True, feed_belief=False, Nepochs=Nepochs,
#                                                   gain=2, seed=i)
#
#     # save
#     torch.save({'tars': tars, 'stops': stops, 'prob_correct': prob_correct, 'prob_shuffled': prob_shuffled, 'auc': auc,
#                 'NTlist': NTlist, 'va': va, 'xa': xa, }, '..//Data//firefly-feedpos-gaingen-results.pt')
#
# if test_baseline_sensorygen:
#
#     data = torch.load('..//Data//firefly-baseline-db.pt')
#     net, task, plant, algo, learning = data['net'], data['task'], data['plant'], data['algo'], data['learning']
#     nrepeats, nconds = np.shape(net)
#     tars, stops = np.empty((nrepeats, nconds, Nepochs, 4)), np.empty((nrepeats, nconds, Nepochs, 4))
#     prob_correct, prob_shuffled = np.empty((nrepeats, nconds, 25)), np.empty((nrepeats, nconds, 25))
#     auc, NTlist = np.empty((nrepeats, nconds)), np.empty((nrepeats, nconds, Nepochs))
#     xa, va = np.empty((nrepeats, nconds, Nepochs, 600, 2)), np.empty((nrepeats, nconds, Nepochs, 600, 2))
#
#     for i in range(nrepeats):
#         for j in range(nconds):
#             print('\n Net: ' + str(i+1) + '/' + str(nrepeats) + '\t Condition: ' + str(j+1) + '/' + str(nconds), end='')
#             tars[i, j], stops[i, j], prob_correct[i, j], prob_shuffled[i, j], auc[i, j], NTlist[i, j], \
#             _, _, _, va[i, j], xa[i, j], _ = test(net[i, j], task[i, j], plant[i, j], algo[i, j], learning[i, j],
#                                                   feed_pos=False, feed_belief=False, Nepochs=Nepochs,
#                                                   sn=5, seed=i)
#
#     # save
#     torch.save({'tars': tars, 'stops': stops, 'prob_correct': prob_correct, 'prob_shuffled': prob_shuffled, 'auc': auc,
#                 'NTlist': NTlist, 'va': va, 'xa': xa, }, '..//Data//firefly-baseline-sensorygen-results.pt')
#
# if test_feedpos_sensorygen:
#
#     data = torch.load('..//Data//firefly-feedpos-db.pt')
#     net, task, plant, algo, learning = data['net'], data['task'], data['plant'], data['algo'], data['learning']
#     nrepeats, nconds = np.shape(net)
#     tars, stops = np.empty((nrepeats, nconds, Nepochs, 4)), np.empty((nrepeats, nconds, Nepochs, 4))
#     prob_correct, prob_shuffled = np.empty((nrepeats, nconds, 25)), np.empty((nrepeats, nconds, 25))
#     auc, NTlist = np.empty((nrepeats, nconds)), np.empty((nrepeats, nconds, Nepochs))
#     xa, va = np.empty((nrepeats, nconds, Nepochs, 600, 2)), np.empty((nrepeats, nconds, Nepochs, 600, 2))
#
#     for i in range(nrepeats):
#         for j in range(nconds):
#             print('\n Net: ' + str(i+1) + '/' + str(nrepeats) + '\t Condition: ' + str(j+1) + '/' + str(nconds), end='')
#             tars[i, j], stops[i, j], prob_correct[i, j], prob_shuffled[i, j], auc[i, j], NTlist[i, j], \
#             _, _, _, va[i, j], xa[i, j], _ = test(net[i, j], task[i, j], plant[i, j], algo[i, j], learning[i, j],
#                                                   feed_pos=True, feed_belief=False, Nepochs=Nepochs,
#                                                   sn=5, seed=i)
#
#     # save
#     torch.save({'tars': tars, 'stops': stops, 'prob_correct': prob_correct, 'prob_shuffled': prob_shuffled, 'auc': auc,
#                 'NTlist': NTlist, 'va': va, 'xa': xa, }, '..//Data//firefly-feedpos-sensorygen-results.pt')
#
# if test_baseline_processgen:
#
#     data = torch.load('..//Data//firefly-baseline-db.pt')
#     net, task, plant, algo, learning = data['net'], data['task'], data['plant'], data['algo'], data['learning']
#     nrepeats, nconds = np.shape(net)
#     tars, stops = np.empty((nrepeats, nconds, Nepochs, 4)), np.empty((nrepeats, nconds, Nepochs, 4))
#     prob_correct, prob_shuffled = np.empty((nrepeats, nconds, 25)), np.empty((nrepeats, nconds, 25))
#     auc, NTlist = np.empty((nrepeats, nconds)), np.empty((nrepeats, nconds, Nepochs))
#     xa, va = np.empty((nrepeats, nconds, Nepochs, 600, 2)), np.empty((nrepeats, nconds, Nepochs, 600, 2))
#
#     for i in range(nrepeats):
#         for j in range(nconds):
#             print('\n Net: ' + str(i + 1) + '/' + str(nrepeats) + '\t Condition: ' + str(j + 1) + '/' + str(nconds),
#                   end='')
#             tars[i, j], stops[i, j], prob_correct[i, j], prob_shuffled[i, j], auc[i, j], NTlist[i, j], \
#             _, _, _, va[i, j], xa[i, j], _ = test(net[i, j], task[i, j], plant[i, j], algo[i, j], learning[i, j],
#                                                   feed_pos=False, feed_belief=False, Nepochs=Nepochs,
#                                                   pn=1, seed=i)
#
#     # save
#     torch.save({'tars': tars, 'stops': stops, 'prob_correct': prob_correct, 'prob_shuffled': prob_shuffled, 'auc': auc,
#                 'NTlist': NTlist, 'va': va, 'xa': xa, }, '..//Data//firefly-baseline-processgen-results.pt')
#
# if test_feedpos_processgen:
#
#     data = torch.load('..//Data//firefly-feedpos-db.pt')
#     net, task, plant, algo, learning = data['net'], data['task'], data['plant'], data['algo'], data['learning']
#     nrepeats, nconds = np.shape(net)
#     tars, stops = np.empty((nrepeats, nconds, Nepochs, 4)), np.empty((nrepeats, nconds, Nepochs, 4))
#     prob_correct, prob_shuffled = np.empty((nrepeats, nconds, 25)), np.empty((nrepeats, nconds, 25))
#     auc, NTlist = np.empty((nrepeats, nconds)), np.empty((nrepeats, nconds, Nepochs))
#     xa, va = np.empty((nrepeats, nconds, Nepochs, 600, 2)), np.empty((nrepeats, nconds, Nepochs, 600, 2))
#
#     for i in range(nrepeats):
#         for j in range(nconds):
#             print('\n Net: ' + str(i + 1) + '/' + str(nrepeats) + '\t Condition: ' + str(j + 1) + '/' + str(nconds),
#                   end='')
#             tars[i, j], stops[i, j], prob_correct[i, j], prob_shuffled[i, j], auc[i, j], NTlist[i, j], \
#             _, _, _, va[i, j], xa[i, j], _ = test(net[i, j], task[i, j], plant[i, j], algo[i, j], learning[i, j],
#                                                   feed_pos=True, feed_belief=False, Nepochs=Nepochs,
#                                                   pn=1, seed=i)
#
#     # save
#     torch.save({'tars': tars, 'stops': stops, 'prob_correct': prob_correct, 'prob_shuffled': prob_shuffled, 'auc': auc,
#                 'NTlist': NTlist, 'va': va, 'xa': xa, }, '..//Data//firefly-feedpos-processgen-results.pt')
#
# if plot_behavior:
#     data = {'baseline': torch.load('..//Data//firefly-baseline-pn-results.pt'),
#             'feedpos': torch.load('..//Data//firefly-feedpos-pn-results.pt'),
#             'feedbelief': torch.load('..//Data//firefly-feedbelief-results.pt')}
#     plot(data, 'behavior')
#
# if plot_processnoise:
#     data = {'baseline': torch.load('..//Data//firefly-baseline-pn-results.pt'),
#             'feedpos': torch.load('..//Data//firefly-feedpos-pn-results.pt'),
#             'feedbelief': torch.load('..//Data//firefly-feedbelief-pn-results.pt')}
#     plot(data, 'processnoise')
#
# if plot_sensorynoise:
#     data = {'baseline': torch.load('..//Data//firefly-baseline-sn-results.pt'),
#             'feedpos': torch.load('..//Data//firefly-feedpos-sn-results.pt'),
#             'feedbelief': torch.load('..//Data//firefly-feedbelief-sn-results.pt')}
#     plot(data, 'sensorynoise')
#
# if plot_distgen:
#     data = {'baseline': torch.load('..//Data//firefly-baseline-db-results.pt'),
#             'feedpos': torch.load('..//Data//firefly-feedpos-db-results.pt'),
#             'feedbelief': torch.load('..//Data//firefly-feedbelief-db-results.pt')}
#     plot(data, 'distgen')
#
# if plot_gaingen:
#     data = {'baseline_train': torch.load('..//Data//firefly-baseline-db-results.pt'),
#             'baseline_test': torch.load('..//Data//firefly-baseline-gaingen-results.pt'),
#             'feedpos_train': torch.load('..//Data//firefly-feedpos-db-results.pt'),
#             'feedpos_test': torch.load('..//Data//firefly-feedpos-gaingen-results.pt')}
#     plot(data, 'gaingen')
#
# if plot_processgen:
#     data = {'baseline_train': torch.load('..//Data//firefly-baseline-db-results.pt'),
#             'baseline_test': torch.load('..//Data//firefly-baseline-processgen-results.pt'),
#             'feedpos_train': torch.load('..//Data//firefly-feedpos-db-results.pt'),
#             'feedpos_test': torch.load('..//Data//firefly-feedpos-processgen-results.pt')}
#     plot(data, 'processgen')
#
# if plot_sensorygen:
#     data = {'baseline_train': torch.load('..//Data//firefly-baseline-db-results.pt'),
#             'baseline_test': torch.load('..//Data//firefly-baseline-sensorygen-results.pt'),
#             'feedpos_train': torch.load('..//Data//firefly-feedpos-db-results.pt'),
#             'feedpos_test': torch.load('..//Data//firefly-feedpos-sensorygen-results.pt')}
#     plot(data, 'sensorygen')
#
# if plot_iocurve:
#     data = {'test1': torch.load('..//Data//firefly-test-sn1.pt'),
#             'test2': torch.load('..//Data//firefly-test-sn2.pt')}
#     ntrl, nt, nunits = np.shape(data['test1']['Ia'])
#     NTlist1, NTlist2 = data['test1']['NTlist'], data['test2']['NTlist']
#     Ia1, Ia2 = data['test1']['Ia'], data['test2']['Ia']
#     ha1, ha2 = data['test1']['ha'], data['test2']['ha']
#     I1, I2, h1, h2 = [], [], [], []
#     for unit in range(nunits):
#         I1.append([])
#         I2.append([])
#         h1.append([])
#         h2.append([])
#     for unit in range(nunits):
#         for trial in range(ntrl):
#             I1[unit].append(Ia1[trial, 30:NTlist1[trial], unit].detach().numpy())
#             h1[unit].append(ha1[trial, 30:NTlist1[trial], unit].detach().numpy())
#             I2[unit].append(Ia2[trial, 30:NTlist2[trial], unit].detach().numpy())
#             h2[unit].append(ha2[trial, 30:NTlist2[trial], unit].detach().numpy())
#         I1[unit] = np.hstack(I1[unit])
#         h1[unit] = np.hstack(h1[unit])
#         I2[unit] = np.hstack(I2[unit])
#         h2[unit] = np.hstack(h2[unit])
#
#     plt.plot(I1[0], h1[0], '.k')
