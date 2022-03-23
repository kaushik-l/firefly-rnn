from matplotlib import pyplot as plt
import numpy as np
import numpy.random as npr


def bootstrap(x, median=True, nboots=100):
    nrepeats, nx = np.shape(x)
    x_m = np.empty((nboots, nx))
    for boot in range(nboots):
        idx = npr.choice(nrepeats, nrepeats)
        x_m[boot] = np.mean(np.array(x)[idx], axis=0)
    return np.median(x_m, axis=0), np.std(x_m, axis=0)


def plot(data, plottype):
    colors = ['xkcd:mustard green', 'xkcd:purplish', 'xkcd:very light purple']

    if plottype == 'behavior':

        for idx, key in enumerate(data.keys()):
            Nmodels, Nconds, Nepochs = np.shape(data[key]['xa'])[:3]
            model_id = npr.choice(Nmodels)
            xa = data[key]['xa'][model_id, -3]
            NTlist = data[key]['NTlist'][model_id, -3]
            prob_shuffled = data[key]['prob_shuffled'][model_id, -3]
            prob_correct = data[key]['prob_correct'][model_id, -3]
            tars = data[key]['tars'][model_id, -3]
            stops = data[key]['stops'][model_id, -3]

            fig, axs = plt.subplots(2, 2)
            plt.subplot(2, 2, 1)
            for ei in range(Nepochs):
                plt.plot(xa[ei, :int(NTlist[ei]), 0], xa[ei, :int(NTlist[ei]), 1], color='black', linewidth=0.2)
            plt.xlim((-3, 3)), plt.ylim((0, 6))

            plt.subplot(2, 2, 2)
            plt.plot(prob_shuffled, prob_correct, color=colors[idx])

            plt.subplot(2, 2, 3)
            plt.plot(np.array(tars)[:, 0], np.array(stops)[:, 0], '.', color=colors[idx])
            plt.xlim((-2, 2)), plt.ylim((-2, 2))

            plt.subplot(2, 2, 4)
            plt.plot(np.array(tars)[:, 1], np.array(stops)[:, 1], '.', color=colors[idx])
            plt.xlim((0, 6)), plt.ylim((0, 6))
        plt.show()

    elif plottype == 'processnoise':
        fig, axs = plt.subplots(1, 3)

        for idx, key in enumerate(data.keys()):
            nconds = np.shape(data[key]['auc'])[1]
            prob_correct = data[key]['prob_correct'].mean(axis=0)
            prob_shuffled = data[key]['prob_shuffled'].mean(axis=0)
            plt.subplot(1, 3, idx + 1)
            for cond in range(nconds):
                plt.plot(prob_shuffled[cond], prob_correct[cond], alpha=(cond+1)/nconds, color=colors[idx])
            plt.xlabel('Prob shuffled'), plt.ylabel('Prob correct')

            auc = data[key]['auc']
            auc_mu, auc_sem = bootstrap(auc)
            pn = np.linspace(0, 1, 6)
            plt.subplot(1, 3, 3)
            plt.plot(pn, auc_mu, 'k-')
            plt.fill_between(pn, auc_mu - auc_sem, auc_mu + auc_sem, color=colors[idx])
            plt.ylim((0.5, 1))
            plt.xlabel('Noise level'), plt.ylabel('AUC')

        plt.suptitle('Effect of process noise')
        plt.show()

    elif plottype == 'sensorynoise':
        fig, axs = plt.subplots(1, 3)

        for idx, key in enumerate(data.keys()):
            nconds = np.shape(data[key]['auc'])[1]
            prob_correct = data[key]['prob_correct'].mean(axis=0)
            prob_shuffled = data[key]['prob_shuffled'].mean(axis=0)
            plt.subplot(1, 3, idx + 1)
            for cond in range(nconds):
                plt.plot(prob_shuffled[cond], prob_correct[cond], alpha=(cond+1)/nconds, color=colors[idx])
            plt.xlabel('Prob shuffled'), plt.ylabel('Prob correct')

            auc = data[key]['auc']
            auc_mu, auc_sem = bootstrap(auc)
            pn = np.linspace(0, 5, 6)
            plt.subplot(1, 3, 3)
            plt.plot(pn, auc_mu, 'k-')
            plt.fill_between(pn, auc_mu - auc_sem, auc_mu + auc_sem, color=colors[idx])
            plt.ylim((0.5, 1))
            plt.xlabel('Noise level'), plt.ylabel('AUC')

        plt.suptitle('Effect of sensory noise')
        plt.show()

    elif plottype == 'distgen':
        fig, axs = plt.subplots(1, 2)

        for idx, key in enumerate(data.keys()):
            prob_correct = data[key]['prob_correct'].mean(axis=0)
            prob_shuffled = data[key]['prob_shuffled'].mean(axis=0)
            plt.subplot(1, 2, 1)
            plt.plot(prob_shuffled[1], prob_correct[1], color=colors[idx])
            plt.xlabel('Prob shuffled'), plt.ylabel('Prob correct')

            auc = data[key]['auc']
            auc_mu, auc_sem = bootstrap(auc)
            plt.subplot(1, 2, 2)
            plt.errorbar(auc_mu[1], auc_mu[0], xerr=auc_sem[1], yerr=auc_sem[0], color=colors[idx])
            plt.plot([0, 1], [0, 1], color='k', linewidth=.5)
            plt.xlim((0.5, 1)), plt.ylim((0.5, 1))
            plt.xlabel('Train'), plt.ylabel('Test')

        plt.suptitle('Generalization to longer distances')
        plt.show()

    elif plottype == 'gaingen':
        fig, axs = plt.subplots(1, 2)

        prob_correct, prob_shuffled, auc_mu, auc_sem = [], [], [], []
        for idx, key in enumerate(data.keys()):
            prob_correct.append(data[key]['prob_correct'].mean(axis=0))
            prob_shuffled.append(data[key]['prob_shuffled'].mean(axis=0))
            mu, sem = bootstrap(data[key]['auc'])
            auc_mu.append(mu), auc_sem.append(sem)
        prob_correct, prob_shuffled, auc_mu, auc_sem = \
            np.array(prob_correct), np.array(prob_shuffled), np.array(auc_mu), np.array(auc_sem)

        plt.subplot(1, 2, 1)
        plt.plot(prob_shuffled[1, -1], prob_correct[1, -1], color=colors[0])
        plt.plot(prob_shuffled[3, -1], prob_correct[3, -1], color=colors[1])
        plt.xlabel('Prob shuffled'), plt.ylabel('Prob correct')
        plt.subplot(1, 2, 2)
        plt.errorbar(auc_mu[0, -1], auc_mu[1, -1], xerr=auc_sem[0, -1], yerr=auc_sem[1, -1], color=colors[0])
        plt.errorbar(auc_mu[2, -1], auc_mu[3, -1], xerr=auc_sem[2, -1], yerr=auc_sem[3, -1], color=colors[1])
        plt.plot([0, 1], [0, 1], color='k', linewidth=.5)
        plt.xlim((0.5, 1)), plt.ylim((0.5, 1))
        plt.xlabel('Train'), plt.ylabel('Test')

        plt.suptitle('Generalization to larger gains')
        plt.show()

    elif plottype == 'processgen':
        fig, axs = plt.subplots(1, 2)

        prob_correct, prob_shuffled, auc_mu, auc_sem = [], [], [], []
        for idx, key in enumerate(data.keys()):
            prob_correct.append(data[key]['prob_correct'].mean(axis=0))
            prob_shuffled.append(data[key]['prob_shuffled'].mean(axis=0))
            mu, sem = bootstrap(data[key]['auc'])
            auc_mu.append(mu), auc_sem.append(sem)
        prob_correct, prob_shuffled, auc_mu, auc_sem = \
            np.array(prob_correct), np.array(prob_shuffled), np.array(auc_mu), np.array(auc_sem)

        plt.subplot(1, 2, 1)
        plt.plot(prob_shuffled[1, -1], prob_correct[1, -1], color=colors[0])
        plt.plot(prob_shuffled[3, -1], prob_correct[3, -1], color=colors[1])
        plt.xlabel('Prob shuffled'), plt.ylabel('Prob correct')
        plt.subplot(1, 2, 2)
        plt.errorbar(auc_mu[0, -1], auc_mu[1, -1], xerr=auc_sem[0, -1], yerr=auc_sem[1, -1], color=colors[0])
        plt.errorbar(auc_mu[2, -1], auc_mu[3, -1], xerr=auc_sem[2, -1], yerr=auc_sem[3, -1], color=colors[1])
        plt.plot([0, 1], [0, 1], color='k', linewidth=.5)
        plt.xlim((0.5, 1)), plt.ylim((0.5, 1))
        plt.xlabel('Train'), plt.ylabel('Test')

        plt.suptitle('Generalization to larger process noise')
        plt.show()

    elif plottype == 'sensorygen':
        fig, axs = plt.subplots(1, 2)

        prob_correct, prob_shuffled, auc_mu, auc_sem = [], [], [], []
        for idx, key in enumerate(data.keys()):
            prob_correct.append(data[key]['prob_correct'].mean(axis=0))
            prob_shuffled.append(data[key]['prob_shuffled'].mean(axis=0))
            mu, sem = bootstrap(data[key]['auc'])
            auc_mu.append(mu), auc_sem.append(sem)
        prob_correct, prob_shuffled, auc_mu, auc_sem = \
            np.array(prob_correct), np.array(prob_shuffled), np.array(auc_mu), np.array(auc_sem)

        plt.subplot(1, 2, 1)
        plt.plot(prob_shuffled[1, -1], prob_correct[1, -1], color=colors[0])
        plt.plot(prob_shuffled[3, -1], prob_correct[3, -1], color=colors[1])
        plt.xlabel('Prob shuffled'), plt.ylabel('Prob correct')
        plt.subplot(1, 2, 2)
        plt.errorbar(auc_mu[0, -1], auc_mu[1, -1], xerr=auc_sem[0, -1], yerr=auc_sem[1, -1], color=colors[0])
        plt.errorbar(auc_mu[2, -1], auc_mu[3, -1], xerr=auc_sem[2, -1], yerr=auc_sem[3, -1], color=colors[1])
        plt.plot([0, 1], [0, 1], color='k', linewidth=.5)
        plt.xlim((0.5, 1)), plt.ylim((0.5, 1))
        plt.xlabel('Train'), plt.ylabel('Test')

        plt.suptitle('Generalization to larger sensory noise')
        plt.show()

# # plot of target location vs stopping location
# plt.figure()
# plt.plot(np.array(tars)[:, 2], np.array(stops)[:, 2], '.')
# plt.xticks(range(0,5), fontsize=16), plt.yticks(range(0,5), fontsize=16)
# plt.xlim(.5, 4.5), plt.ylim(.5, 4.5)
#
# plt.figure()
# plt.plot(np.array(tars)[:, 3]*180/np.pi, np.array(stops)[:, 3]*180/np.pi, '.')
# plt.xticks(range(-40,41,20), fontsize=16), plt.yticks(range(-40,41,20), fontsize=16)
#
#
# # linear velocity
# plt.figure()
# for trial in range(100, 200):
#     plt.plot(va[trial, :NTlist[trial], 0].detach().numpy(), color='Blue')
#
# # angular velocity
# plt.figure()
# for trial in range(100, 200):
#     plt.plot(va[trial, :NTlist[trial], 1].detach().numpy(), color='Red')
#
# # trajectories
# plt.figure()
# for trial in range(100, 200):
#     plt.plot(xa[trial, :NTlist[trial], 0].detach().numpy(), xa[trial, :NTlist[trial], 1].detach().numpy(), color='k')
#
# # roc curve
# # plt.plot(prob_shuffled, prob_correct)
#
# for idx in range(20):
#     plt.plot(prob_shuffled[0, idx, :], prob_correct[0, idx, :])
#
# Nepochs, maxNT, N = 1000, np.max(NTlist), 100
# # small distance trials
# sd = np.argsort(np.array(tars)[:, 2])[:int(Nepochs/10)]
# # large distance trials
# ld = np.argsort(np.array(tars)[:, 2])[int(9*Nepochs/10):]
# # small angle trials
# st = np.argsort((np.array(tars))[:, 3])[:int(Nepochs/10)]
# # large angle trials
# lt = np.argsort((np.array(tars))[:, 3])[int(9*Nepochs/10):]
#
# # make trials equal length
# for trial in range(Nepochs):
#     for unit in range(N):
#         ha[trial, :maxNT, unit] = \
#             torch.tensor(np.interp(np.linspace(0, 1, maxNT),
#                                    np.linspace(0, 1, NTlist[trial]), ha[trial, :NTlist[trial], unit]))
#
# # average small_dist trial
# ha_mean_sd = torch.mean(ha[sd, :, :], dim=0)
# for unit in range(N):
#     ha_mean_sd[:, unit] = ha_mean_sd[:, unit] / torch.max(abs(ha_mean_sd[:, unit]))
# peaks = np.argmax(ha_mean_sd, axis=0)
# order = np.argsort(peaks)
# plt.subplot(2,2,1)
# plt.imshow(ha_mean_sd[:, order].T, vmin=-1, vmax=1)
# plt.xlim(0, 200), plt.ylim(0, 100)
#
# # average small_dist trial (sorted by other)
# ha_mean_sd = torch.mean(ha[sd, :, :], dim=0)
# ha_mean_ld = torch.mean(ha[ld, :, :], dim=0)
# for unit in range(N):
#     ha_mean_sd[:, unit] = ha_mean_sd[:, unit] / torch.max(abs(ha_mean_ld[:, unit]))
# peaks = np.argmax(ha_mean_ld, axis=0)
# order = np.argsort(peaks)
# plt.subplot(2,2,2)
# plt.imshow(ha_mean_sd[:, order].T, vmin=-1, vmax=1)
# plt.xlim(0, 200), plt.ylim(0, 100)
#
# # average large_dist trial
# ha_mean_ld = torch.mean(ha[ld, :, :], dim=0)
# for unit in range(N):
#     ha_mean_ld[:, unit] = ha_mean_ld[:, unit] / torch.max(abs(ha_mean_ld[:, unit]))
# peaks = np.argmax(ha_mean_ld, axis=0)
# order = np.argsort(peaks)
# plt.subplot(2,2,4)
# plt.imshow(ha_mean_ld[:, order].T, vmin=-1, vmax=1)
# plt.xlim(0, 200), plt.ylim(0, 100)
#
# # average large_dist trial (sorted by other)
# ha_mean_ld = torch.mean(ha[ld, :, :], dim=0)
# ha_mean_sd = torch.mean(ha[sd, :, :], dim=0)
# for unit in range(N):
#     ha_mean_ld[:, unit] = ha_mean_ld[:, unit] / torch.max(abs(ha_mean_sd[:, unit]))
# peaks = np.argmax(ha_mean_sd, axis=0)
# order = np.argsort(peaks)
# plt.subplot(2,2,3)
# plt.imshow(ha_mean_ld[:, order].T, vmin=-1, vmax=1)
# plt.xlim(0, 200), plt.ylim(0, 100)
#
# # average small_angle trial
# ha_mean_st = torch.mean(ha[st, :, :], dim=0)
# for unit in range(N):
#     ha_mean_st[:, unit] = ha_mean_st[:, unit] / torch.max(abs(ha_mean_st[:, unit]))
# peaks = np.argmax(ha_mean_st, axis=0)
# order = np.argsort(peaks)
# plt.subplot(2,2,1)
# plt.imshow(ha_mean_st[:, order].T, vmin=-1, vmax=1)
# plt.ylim(70, 256)
#
# # average small_angle trial (sorted by other)
# ha_mean_st = torch.mean(ha[st, :, :], dim=0)
# ha_mean_lt = torch.mean(ha[lt, :, :], dim=0)
# for unit in range(N):
#     ha_mean_st[:, unit] = ha_mean_st[:, unit] / torch.max(abs(ha_mean_lt[:, unit]))
# peaks = np.argmax(ha_mean_lt, axis=0)
# order = np.argsort(peaks)
# plt.subplot(2,2,2)
# plt.imshow(ha_mean_st[:, order].T, vmin=-1, vmax=1)
# plt.ylim(70, 256)
#
# # average large_angle trial
# ha_mean_lt = torch.mean(ha[lt, :, :], dim=0)
# for unit in range(N):
#     ha_mean_lt[:, unit] = ha_mean_lt[:, unit] / torch.max(abs(ha_mean_lt[:, unit]))
# peaks = np.argmax(ha_mean_lt, axis=0)
# order = np.argsort(peaks)
# plt.subplot(2,2,3)
# plt.imshow(ha_mean_lt[:, order].T, vmin=-1, vmax=1)
# plt.ylim(70, 256)
#
# # average large_angle trial (sorted by other)
# ha_mean_st = torch.mean(ha[st, :, :], dim=0)
# ha_mean_lt = torch.mean(ha[lt, :, :], dim=0)
# for unit in range(N):
#     ha_mean_lt[:, unit] = ha_mean_lt[:, unit] / torch.max(abs(ha_mean_st[:, unit]))
# peaks = np.argmax(ha_mean_st, axis=0)
# order = np.argsort(peaks)
# plt.subplot(2,2,4)
# plt.imshow(ha_mean_lt[:, order].T, vmin=-1, vmax=1)
# plt.ylim(70, 256)
#
# #