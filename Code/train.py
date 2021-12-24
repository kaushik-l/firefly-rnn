import torch
from time import time
import numpy as np
import numpy.random as npr
from model import Network, Task, Algorithm, Plant
from scipy import stats
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from scipy.stats.stats import pearsonr
from scipy.linalg import subspace_angles
from matplotlib import pyplot as plt


def gen_tar(dist_bounds, angle_bounds):
    dmin, dmax = dist_bounds
    amin, amax = angle_bounds
    r = -1
    while r < dmin:
        r = dmax * np.sqrt(npr.rand())
    a = (amax - amin) * npr.rand() + amin
    x, y = r * np.sin(a), r * np.cos(a)
    return x, y


def plan_traj(x_tar, y_tar, vmax, dt, nr):
    R = (x_tar ** 2 + y_tar ** 2) / (2 * abs(x_tar))
    r = np.sqrt(x_tar ** 2 + y_tar ** 2)
    theta = np.arctan2(abs(y_tar), abs(x_tar))
    phi = np.arctan2(r * np.sin(theta), R - r * np.cos(theta))
    t_d = (R * phi) / vmax
    v = vmax * np.ones((round(t_d / dt)))
    v[:nr] = np.linspace(0, vmax, nr)
    v = np.append(v, np.linspace(vmax, 0, nr))      # m per tau
    w = np.sign(x_tar) * (v / R)
    a = 10 * np.diff(v) / dt                        # m per tau^2 scaled by 10 so network output need not be too small
    b = 10 * np.diff(w) / dt
    t = np.arange(0, (dt * (len(v))), dt) + dt
    return a, b, v, w


def gen_traj(v, dt):
    v_lin, v_ang = v.T
    NT = v.size()[0]
    x, y, phi = np.zeros(NT), np.zeros(NT), np.zeros(NT)
    for idx in range(NT-1):
        x[idx+1] = x[idx] + v_lin[idx] * np.sin(phi[idx]) * dt
        y[idx+1] = y[idx] + v_lin[idx] * np.cos(phi[idx]) * dt
        phi[idx+1] = phi[idx] + v_ang[idx] * dt
        phi[idx+1] = ((phi[idx+1] > -np.pi) and (phi[idx+1] <= np.pi)) * phi[idx+1] + \
                     (phi[idx+1] > np.pi) * (phi[idx+1] - 2 * np.pi) + \
                     (phi[idx+1] <= -np.pi) * (phi[idx+1] + 2 * np.pi)
    return x, y, phi


def compute_roc(X_tars, X_stops, maxrewardwin=4, npermutations=100):

    # initialise
    ntrials = len(X_tars)
    binedges = np.linspace(0, maxrewardwin, 21)
    nbins = len(binedges) - 1
    prob_correct = np.zeros(nbins+1)
    prob_shuffled = np.zeros((npermutations, nbins+1))

    # actual reward probability
    err = np.sqrt(np.sum((X_tars - X_stops) ** 2, axis=1))
    for bin in range(nbins+1):
        prob_correct[bin] = np.sum(err < binedges[bin]) / ntrials

    # shuffled reward probability
    for permutation in range(npermutations):
        trials = npr.choice(ntrials, size=ntrials)
        err_shuffled = np.sqrt(np.sum((X_tars - X_stops[trials]) ** 2, axis=1))
        for bin in range(nbins+1):
            prob_shuffled[permutation, bin] = np.sum(err_shuffled < binedges[bin]) / ntrials
    prob_shuffled = prob_shuffled.mean(axis=0)

    # area under the curve
    auc = np.sum(0.5 * (prob_correct[1:] + prob_correct[:-1]) * np.diff(prob_shuffled))

    return prob_correct, prob_shuffled, auc


def train(arch='ThCtx', N=256, S=4, R=2, task='firefly', num_tar=None,
          rand_tar=True, algo='Adam', Nepochs=10000, lr=1e-3, learningsites=('J'), seed=1, minerr=1e-15):

    # set random seed
    npr.seed(seed)

    # sites
    sites = ('ws', 'J', 'wr')

    # instantiate model
    net = Network(arch, N, S, seed=seed)
    task = Task(task, num_tar=num_tar, rand_tar=rand_tar, seed=seed)
    plant = Plant('joystick', seed=seed)
    algo = Algorithm(algo, Nepochs, lr)

    # convert to tensor
    for site in sites:
        if site in learningsites:
            setattr(net, site, torch.tensor(getattr(net, site), requires_grad=True))
        else:
            setattr(net, site, torch.tensor(getattr(net, site), requires_grad=False))

    # frequently used vars
    dt, NT, NT_on, NT_ramp, N, S, R = net.dt, task.NT, task.duration_tar, task.duration_ramp, net.N, net.S, net.R
    t = dt * np.arange(NT)

    # OU noise parameters
    ou_param_1 = np.exp(-dt / plant.noise_corr_time)
    ou_param_2 = np.sqrt(1 - ou_param_1 ** 2)

    # track variables during learning
    learning = {'epoch': [], 'J0': [], 'mses': [], 'snr_mse': [], 'lr': []}

    # optimizer
    opt = None
    if algo.name == 'Adam':
        opt = torch.optim.Adam([getattr(net, site) for site in sites], lr=algo.lr)

    # random initialization of hidden state
    z0 = npr.randn(N, 1)  # hidden state (potential)
    h0 = net.f(z0)  # hidden state (rate)
    stoplearning = False

    # initial noise
    noise = torch.as_tensor(npr.randn(1, R).astype(np.float32)) * plant.process_noise

    # save initial weights
    learning['J0'] = net.J.detach().clone()
    reg = LinearRegression()

    # save target and stopping positions
    tars, stops = [], []

    # train
    for ei in range(algo.Nepochs):

        # pick a condition
        if task.rand_tar:
            x_tar, y_tar = gen_tar(task.dist_bounds, task.angle_bounds)
            a, b, v, w = plan_traj(x_tar, y_tar, task.v_bounds[-1], dt, NT_ramp)
            task.s[0, :NT_on] = x_tar
            task.s[1, :NT_on] = y_tar
            task.ustar[:, :] = 0
            task.ustar[NT_on:NT_on+len(a)] = np.array([a, b]).T

        # initialize activity
        z, h = torch.as_tensor(z0), torch.as_tensor(h0)
        v_true, v_sense = torch.zeros(R, 1), torch.zeros(R, 1)
        s = torch.tensor(task.s)  # input

        # save tensors for plotting
        sa = torch.zeros(NT, S)  # save the inputs for each time bin for plotting
        ha = torch.zeros(NT, N)  # save the hidden states for each time bin for plotting
        ua = torch.zeros(NT, R)  # acceleration
        va = torch.zeros(NT, R)  # velocity
        na = torch.as_tensor(npr.randn(NT, R).astype(np.float32)) * plant.process_noise

        # errors
        err = torch.zeros(NT, R)     # error in angular acceleration

        for ti in range(NT):
            # network update
            s[(S-R):, ti] = v_sense.flatten()        # sensory feedback in the last two channels
            if net.name == 'ppc':
                Iin = torch.matmul(net.ws, s[:, ti][:, None])
                Irec = torch.matmul(net.J, h)  #np.matmul(net.J, h)
                z = Iin + Irec      # potential

            # update activity
            h = (1 - dt) * h + dt * (net.f(z))  # cortex
            u = net.wr.mm(h)  # output

            noise = ou_param_1 * noise + ou_param_2 * na[ti]  # noise is OU process
            v_true, v_sense = plant.forward(u, v_true, noise.T, dt)  # actual state

            # save values for plotting
            ha[ti], ua[ti], na[ti], sa[ti], va[ti] = h.T, u.T, noise, s.T[ti], v_true.T

            # error
            err[ti] = torch.as_tensor(task.ustar[ti]) - u.squeeze(dim=1)

        # target trajectory
        # xstar, ystar, phistar = gen_traj(torch.as_tensor([v, w]).T, dt)

        # actual trajectory
        xa, ya, phia = gen_traj(va, dt)

        # save target and stopping positions
        tars.append([x_tar, y_tar, np.sqrt(x_tar ** 2 + y_tar ** 2), np.arctan2(x_tar, y_tar)])     # x, y, dist, angle
        stops.append([xa[-1], ya[-1], np.sqrt(xa[-1] ** 2 + ya[-1] ** 2), np.arctan2(xa[-1], ya[-1])])

        # print loss
        loss = task.loss(err)
        print('\r' + str(ei + 1) + '/' + str(algo.Nepochs) + '\t Err:' + str(loss.item()), end='')

        # save mse list and cond list
        learning['mses'].append(loss.item())

        # do BPTT
        loss.backward()
        opt.step()
        opt.zero_grad()

        # update learning rate
        update_every = int(1e3)
        if ei+1 in np.arange(1, Nepochs, update_every) or ei == Nepochs:
            if ei >= update_every:
                reg.fit(np.arange(update_every).reshape((-1, 1)), np.log(learning['mses'][-update_every:]))
                learning['snr_mse'].append(reg.score(np.arange(update_every).reshape((-1, 1)),
                                                     np.log(learning['mses'][-update_every:])))
                if learning['snr_mse'][-1] < 0.05: opt.param_groups[0]['lr'] *= .5   # reduce learning rate
                _, _, auc = compute_roc(np.array(tars)[-update_every:, :2], np.array(stops)[-update_every:, :2])
                # stoplearning = np.max(learning['mses'][-10:]) < minerr
                stoplearning = auc > 0.85
            learning['lr'].append(opt.param_groups[0]['lr'])
            learning['epoch'].append(ei)

        # stop learning if criterion is met
        if stoplearning:
            break

    return net, task, algo, plant, learning
