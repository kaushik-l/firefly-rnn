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
    x, y, phi = torch.zeros(NT), torch.zeros(NT), torch.zeros(NT)
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


def test(net, task, plant, algo, learning, Nepochs=1000, seed=1):

    # set random seed
    npr.seed(seed)

    # set grad to False
    sites = ('ws', 'J', 'wr')
    net.J.requires_grad = False

    # frequently used vars
    dt, NT, NT_on, NT_ramp, NT_stop, N, S, R = \
        net.dt, task.NT, task.duration_tar, task.duration_ramp, task.duration_stop, net.N, net.S, net.R
    t = dt * np.arange(NT)

    # OU noise parameters
    ou_param_1 = np.exp(-dt / plant.noise_corr_time)
    ou_param_2 = np.sqrt(1 - ou_param_1 ** 2)

    # random initialization of hidden state
    z0 = net.z0     # hidden state (potential)
    h0 = net.f(z0)  # hidden state (rate)

    # initial noise
    noise = torch.as_tensor(npr.randn(1, R).astype(np.float32)) * plant.process_noise

    # save target and stopping positions
    tars, stops = [], []

    # save tensors for plotting
    sa = torch.zeros(Nepochs, NT, S)  # save the inputs for each time bin for plotting
    ha = torch.zeros(Nepochs, NT, N)  # save the hidden states for each time bin for plotting
    ua = torch.zeros(Nepochs, NT, R)  # acceleration
    va = torch.zeros(Nepochs, NT, R)  # velocity
    xa = torch.zeros(Nepochs, NT, R)  # position
    maxNT, NTlist = 0, []

    # train
    for ei in range(Nepochs):

        # pick a condition
        if task.rand_tar:
            x_tar, y_tar = gen_tar(task.dist_bounds, task.angle_bounds)
            a, b, v, w = plan_traj(x_tar, y_tar, task.v_bounds[-1], dt, NT_ramp)
            task.s[0, :NT_on] = x_tar
            task.s[1, :NT_on] = y_tar
            task.ustar = np.zeros_like(task.s[:2, :].T)
            task.ustar[NT_on:NT_on+len(a)] = np.array([a, b]).T
            task.ustar = task.ustar[:NT, :]

        # target trajectory
        xstar, ystar, phistar = gen_traj(torch.as_tensor([v, w]).T, dt)
        xstar = torch.tensor(np.pad(xstar, NT_stop, 'edge'))
        ystar = torch.tensor(np.pad(ystar, NT_stop, 'edge'))
        NT = len(xstar)
        maxNT = np.max((NT, maxNT))

        # initialize activity
        z, h = torch.as_tensor(z0), torch.as_tensor(h0)
        v_true, v_sense, x, phi = torch.zeros(R, 1), torch.zeros(R, 1), torch.zeros(R, 1), torch.zeros(1)
        s = torch.tensor(task.s)  # input

        # save tensors for plotting
        na = torch.as_tensor(npr.randn(NT, R).astype(np.float32)) * plant.process_noise

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
            v_true, v_sense, x, phi = plant.forward(u, v_true, x, phi, noise.T, dt)  # actual state

            # save values for plotting
            ha[ei, ti], ua[ei, ti], na[ti], sa[ei, ti], va[ei, ti], xa[ei, ti] = h.T, u.T, noise, s.T[ti], v_true.T, x.T

        # save target and stopping positions
        x_stop, y_stop = xa[ei, ti, 0].item(), xa[ei, ti, 1].item()
        tars.append([x_tar, y_tar, np.sqrt(x_tar ** 2 + y_tar ** 2), np.arctan2(x_tar, y_tar)])     # x, y, dist, angle
        stops.append([x_stop, y_stop, np.sqrt(x_stop ** 2 + y_stop ** 2), np.arctan2(x_stop, y_stop)])
        prob_correct, prob_shuffled, auc = compute_roc(np.array(tars)[:, :2], np.array(stops)[:, :2])
        NTlist.append(NT)

        print('\r' + str(ei + 1) + '/' + str(Nepochs), end='')

    return tars, stops, prob_correct, prob_shuffled, auc, NTlist, sa, ha, ua, va, xa
