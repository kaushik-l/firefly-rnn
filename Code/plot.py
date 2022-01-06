from matplotlib import pyplot as plt

# plot of target location vs stopping location
plt.figure()
plt.plot(np.array(tars)[:, 2], np.array(stops)[:, 2], '.')
plt.xticks(range(0,5), fontsize=16), plt.yticks(range(0,5), fontsize=16)
plt.xlim(.5, 4.5), plt.ylim(.5, 4.5)

plt.figure()
plt.plot(np.array(tars)[:, 3]*180/np.pi, np.array(stops)[:, 3]*180/np.pi, '.')
plt.xticks(range(-40,41,20), fontsize=16), plt.yticks(range(-40,41,20), fontsize=16)


# velocity
for trial in range(400, 600):
    plt.plot(va[trial, :NTlist[trial], 0].detach().numpy())

# trajectories
for trial in range(400, 600):
    plt.plot(xa[trial, :NTlist[trial], 0].detach().numpy(), xa[trial, :NTlist[trial], 1].detach().numpy(), color='k')

# roc curve
# plt.plot(prob_shuffled, prob_correct)

Nepochs, maxNT, N = 1000, np.max(NTlist), 256
# small distance trials
sd = np.argsort(np.array(tars)[:, 2])[:int(Nepochs/10)]
# large distance trials
ld = np.argsort(np.array(tars)[:, 2])[int(9*Nepochs/10):]
# small angle trials
st = np.argsort((np.array(tars))[:, 3])[:int(Nepochs/10)]
# large angle trials
lt = np.argsort((np.array(tars))[:, 3])[int(9*Nepochs/10):]

# make trials equal length
for trial in range(Nepochs):
    for unit in range(N):
        ha[trial, :maxNT, unit] = \
            torch.tensor(np.interp(np.linspace(0, 1, maxNT),
                                   np.linspace(0, 1, NTlist[trial]), ha[trial, :NTlist[trial], unit]))

# average small_dist trial
ha_mean_sd = torch.mean(ha[sd, :, :], dim=0)
for unit in range(N):
    ha_mean_sd[:, unit] = ha_mean_sd[:, unit] / torch.max(abs(ha_mean_sd[:, unit]))
peaks = np.argmax(ha_mean_sd, axis=0)
order = np.argsort(peaks)
plt.subplot(2,2,1)
plt.imshow(ha_mean_sd[:, order].T, vmin=-1, vmax=1)
plt.xlim(0, 200), plt.ylim(128, 256)

# average small_dist trial (sorted by other)
ha_mean_sd = torch.mean(ha[sd, :, :], dim=0)
ha_mean_ld = torch.mean(ha[ld, :, :], dim=0)
for unit in range(N):
    ha_mean_sd[:, unit] = ha_mean_sd[:, unit] / torch.max(abs(ha_mean_ld[:, unit]))
peaks = np.argmax(ha_mean_ld, axis=0)
order = np.argsort(peaks)
plt.subplot(2,2,2)
plt.imshow(ha_mean_sd[:, order].T, vmin=-1, vmax=1)
plt.xlim(0, 200), plt.ylim(128, 256)

# average large_dist trial
ha_mean_ld = torch.mean(ha[ld, :, :], dim=0)
for unit in range(N):
    ha_mean_ld[:, unit] = ha_mean_ld[:, unit] / torch.max(abs(ha_mean_ld[:, unit]))
peaks = np.argmax(ha_mean_ld, axis=0)
order = np.argsort(peaks)
plt.subplot(2,2,4)
plt.imshow(ha_mean_ld[:, order].T, vmin=-1, vmax=1)
plt.xlim(0, 200), plt.ylim(128, 256)

# average large_dist trial (sorted by other)
ha_mean_ld = torch.mean(ha[ld, :, :], dim=0)
ha_mean_sd = torch.mean(ha[sd, :, :], dim=0)
for unit in range(N):
    ha_mean_ld[:, unit] = ha_mean_ld[:, unit] / torch.max(abs(ha_mean_sd[:, unit]))
peaks = np.argmax(ha_mean_sd, axis=0)
order = np.argsort(peaks)
plt.subplot(2,2,3)
plt.imshow(ha_mean_ld[:, order].T, vmin=-1, vmax=1)
plt.xlim(0, 200), plt.ylim(128, 256)

# average small_angle trial
ha_mean_st = torch.mean(ha[st, :, :], dim=0)
for unit in range(N):
    ha_mean_st[:, unit] = ha_mean_st[:, unit] / torch.max(abs(ha_mean_st[:, unit]))
peaks = np.argmax(ha_mean_st, axis=0)
order = np.argsort(peaks)
plt.subplot(2,2,1)
plt.imshow(ha_mean_st[:, order].T, vmin=-1, vmax=1)
plt.ylim(70, 256)

# average small_angle trial (sorted by other)
ha_mean_st = torch.mean(ha[st, :, :], dim=0)
ha_mean_lt = torch.mean(ha[lt, :, :], dim=0)
for unit in range(N):
    ha_mean_st[:, unit] = ha_mean_st[:, unit] / torch.max(abs(ha_mean_lt[:, unit]))
peaks = np.argmax(ha_mean_lt, axis=0)
order = np.argsort(peaks)
plt.subplot(2,2,2)
plt.imshow(ha_mean_st[:, order].T, vmin=-1, vmax=1)
plt.ylim(70, 256)

# average large_angle trial
ha_mean_lt = torch.mean(ha[lt, :, :], dim=0)
for unit in range(N):
    ha_mean_lt[:, unit] = ha_mean_lt[:, unit] / torch.max(abs(ha_mean_lt[:, unit]))
peaks = np.argmax(ha_mean_lt, axis=0)
order = np.argsort(peaks)
plt.subplot(2,2,3)
plt.imshow(ha_mean_lt[:, order].T, vmin=-1, vmax=1)
plt.ylim(70, 256)

# average large_angle trial (sorted by other)
ha_mean_st = torch.mean(ha[st, :, :], dim=0)
ha_mean_lt = torch.mean(ha[lt, :, :], dim=0)
for unit in range(N):
    ha_mean_lt[:, unit] = ha_mean_lt[:, unit] / torch.max(abs(ha_mean_st[:, unit]))
peaks = np.argmax(ha_mean_st, axis=0)
order = np.argsort(peaks)
plt.subplot(2,2,4)
plt.imshow(ha_mean_lt[:, order].T, vmin=-1, vmax=1)
plt.ylim(70, 256)