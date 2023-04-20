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
# torch.autograd.set_detect_anomaly(True)


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


def compute_roc(X_tars, X_stops, maxrewardwin=6, npermutations=100):

    # initialise
    ntrials = len(X_tars)
    binedges = np.linspace(0, maxrewardwin, 25)
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


def train(arch='ppc', N=256, S=4, R=2, task='firefly', num_tar=None, rand_tar=True, db=(1, 6), sn=.2, pn=.2,
          feed_pos=False, feed_belief=False, algo='Adam', Nepochs=10000, lr=1e-3, learningsites=('J'), seed=1, minerr=1e-15):

    # set random seed
    npr.seed(seed)

    # sites
    sites = ('ws', 'J', 'wr')

    # instantiate model
    if feed_pos:
        net = Network(arch, N, S+2, R, seed=seed)
    elif feed_belief:
        net = Network(arch, N, S+2, R+2, seed=seed)
    else:
        net = Network(arch, N, S, R, seed=seed)
    task = Task(task, num_tar=num_tar, rand_tar=rand_tar, dist_bounds=db, feed_pos=feed_pos, feed_belief=feed_belief,
                lambda_b=0, lambda_u=1e-1, lambda_du=1e-0, lambda_v=10, lambda_h=0, lambda_dh=0, seed=seed)
    plant = Plant('joystick', sensory_noise=sn, process_noise=pn, seed=seed)
    algo = Algorithm(algo, Nepochs, lr)

    # convert to tensor
    for site in sites:
        if site in learningsites:
            setattr(net, site, torch.tensor(getattr(net, site), requires_grad=True))
        else:
            setattr(net, site, torch.tensor(getattr(net, site), requires_grad=False))

    # frequently used vars
    dt, NT, NT_on, NT_ramp, NT_stop, N, S, R = \
        net.dt, task.NT, task.duration_tar, task.duration_ramp, task.duration_stop, net.N, net.S, net.R
    t = dt * np.arange(NT)

    # OU noise parameters
    ou_param_1 = np.exp(-dt / plant.noise_corr_time)
    ou_param_2 = np.sqrt(1 - ou_param_1 ** 2)

    # track variables during learning
    learning = {'epoch': [], 'J0': [], 'mses': [], 'u_regs': [], 'du_regs': [],
                'v_regs': [], 'h_regs': [],  'dh_regs': [], 'b_regs': [], 'snr_mse': [], 'lr': [], 'auc': []}

    # optimizer
    opt = None
    if algo.name == 'Adam':
        opt = torch.optim.Adam([getattr(net, site) for site in learningsites], lr=algo.lr)

    # random initialization of hidden state
    z0 = np.zeros((N, 1))  # hidden state (potential)
    net.z0 = z0  # save
    h0 = net.f(z0)  # hidden state (rate)
    stoplearning = False
    wr0 = net.wr.detach().clone()

    # initial noise
    noise = torch.as_tensor(npr.randn(1, 2).astype(np.float32)) * plant.process_noise

    # save initial weights
    learning['J0'] = net.J.detach().clone()
    reg = LinearRegression()

    # save target and stopping positions
    tars, stops = [], []
    # save loss, mse, and regularization terms
    losses, mses, u_regs, du_regs, v_regs, h_regs, dh_regs, b_regs = [], [], [], [], [], [], [], []

    # train
    for ei in range(algo.Nepochs):

        # pick a condition
        if task.rand_tar:
            x_tar, y_tar = gen_tar(task.dist_bounds, task.angle_bounds)
            _, _, v, w = plan_traj(x_tar, y_tar, task.v_bounds[-1], dt, NT_ramp)
            task.s[0, :NT_on] = x_tar
            task.s[1, :NT_on] = y_tar

        # target trajectory
        xstar, ystar, phistar = gen_traj(torch.as_tensor([v, w]).T, dt)
        xstar = torch.tensor(np.pad(xstar, NT_stop, 'edge'))
        ystar = torch.tensor(np.pad(ystar, NT_stop, 'edge'))
        NT = len(xstar)

        # initialize activity
        z, h = torch.as_tensor(z0), torch.as_tensor(h0)
        v_true, v_sense, x, phi, x_err, u = \
            torch.zeros(2, 1), torch.zeros(2, 1), torch.zeros(2, 1), torch.zeros(1), torch.zeros(2, 1), torch.zeros(R, 1)
        s = torch.tensor(task.s)  # input

        # save tensors for plotting
        sa = torch.zeros(NT, S)  # save the inputs for each time bin for plotting
        ha = torch.zeros(NT, N)  # save the hidden states for each time bin for plotting
        ua = torch.zeros(NT, R)  # acceleration
        va = torch.zeros(NT, 2)  # velocity
        xa = torch.zeros(NT, 2)  # position
        ba = torch.zeros(NT, 2)  # belief
        na = torch.as_tensor(npr.randn(NT, 2).astype(np.float32)) * plant.process_noise

        # errors
        err = torch.zeros(NT, R)     # error

        for ti in range(NT):
            # network update
            if feed_pos:
                s[2:4, ti] = v_sense.flatten()                                              # sensory feedback
                s[4:, ti] = (torch.tensor((x_tar, y_tar)) - x.squeeze(dim=1))               # position feedback
            elif feed_belief:
                s[2:4, ti] = v_sense.flatten()                                          # sensory feedback
                s[4:, ti] = u[2:].T.flatten()                                      # belief feedback
            else:
                s[2:, ti] = v_sense.flatten()                                           # sensory feedback
            s2 = s.clone()
            if net.name == 'ppc':
                Iin = torch.matmul(net.ws, s2[:, ti][:, None])
                Irec = torch.matmul(net.J, h)
                z = Iin + Irec      # potential

            # update activity
            h = (1 - dt) * h + dt * (net.f(z))  # cortex
            u = net.wr.mm(h)  # output

            noise = ou_param_1 * noise + ou_param_2 * na[ti]  # noise is OU process
            v_true, v_sense, x, phi = plant.forward(u[:2], v_true, x, phi, noise.T, dt)  # actual state
            x_err = (torch.tensor((x_tar, y_tar)) - x.squeeze(dim=1)) if feed_belief else 0

            # save values for plotting
            ha[ti], ua[ti], na[ti], sa[ti], va[ti], xa[ti], ba[ti] = h.T, u.T, noise, s.T[ti], v_true.T, x.T, x_err

            # error
            if feed_belief:
                err_pos = torch.hstack((xstar[ti], ystar[ti])) - x.squeeze(dim=1) \
                    if (ti > (NT - NT_stop) or ti < (NT_on)) else torch.zeros(1, 2).squeeze()
                err_blf = x_err - u[2:].squeeze() \
                    if (NT_on < ti < (NT - NT_stop)) else torch.zeros(1, 2).squeeze()
                err[ti] = torch.hstack((err_pos, err_blf))
            else:
                err[ti] = torch.hstack((xstar[ti], ystar[ti])) - x.squeeze(dim=1) if (ti > (NT - NT_stop) or ti < (NT_on)) \
                    else 0

        # save target and stopping positions
        x_stop, y_stop = xa[-1, 0].item(), xa[-1, 1].item()
        tars.append([x_tar, y_tar, np.sqrt(x_tar ** 2 + y_tar ** 2), np.arctan2(x_tar, y_tar)])     # x, y, dist, angle
        stops.append([x_stop, y_stop, np.sqrt(x_stop ** 2 + y_stop ** 2), np.arctan2(x_stop, y_stop)])

        # print loss
        mse, u_reg, du_reg, v_reg, h_reg, dh_reg, b_reg = task.loss(err, ua, va, ha, ba, dt)
        loss = mse + u_reg + du_reg + v_reg + h_reg + dh_reg + b_reg

        # save loss, mse, and regularization terms
        losses.append(loss.item())
        mses.append(mse.item())
        u_regs.append(u_reg.item())
        du_regs.append(du_reg.item())
        v_regs.append(v_reg.item())
        h_regs.append(h_reg.item())
        dh_regs.append(dh_reg.item())
        b_regs.append(b_reg.item())

        print('\r' + str(ei + 1) + '/' + str(algo.Nepochs) + '\t Err:' + str(loss.item())
              + '\t MSE:' + str(mse.item()), end='')

        # do BPTT
        loss.backward()
        opt.step()
        opt.zero_grad()

        # reset weights
        # net.wr.data[:2] = wr0[:2]

        # delta rule
        # dwr = np.array([torch.outer(ha[idx], err[idx]).detach().numpy() for idx in range(NT)]).mean(axis=0) * 1e-20
        # net.wr -= dwr.T

        # update learning rate
        update_every = int(1e3)
        if ei+1 in np.arange(1, Nepochs, update_every) or ei+1 == Nepochs:
            if ei >= update_every:
                reg.fit(np.arange(update_every).reshape((-1, 1)), np.log(mses[-update_every:]))
                learning['snr_mse'].append(reg.score(np.arange(update_every).reshape((-1, 1)),
                                                     np.log(mses[-update_every:])))
                # if learning['snr_mse'][-1] < 0.05: opt.param_groups[0]['lr'] *= .5   # reduce learning rate
                _, _, auc = compute_roc(np.array(tars)[-update_every:, :2], np.array(stops)[-update_every:, :2])
                learning['auc'].append(auc)
                stoplearning = np.max(mses[-10:]) < minerr
                # stoplearning = auc > 0.85
            learning['lr'].append(opt.param_groups[0]['lr'])
            learning['epoch'].append(ei)
        opt.param_groups[0]['lr'] *= np.exp(np.log(np.minimum(1e-4, opt.param_groups[0]['lr'])
                                                   / opt.param_groups[0]['lr']) / Nepochs)
        learning['lr'].append(opt.param_groups[0]['lr'])

        # stop learning if criterion is met
        if stoplearning:
            break

    # save mse list and cond list
    learning['mses'].append(mses)
    learning['u_regs'].append(u_regs)
    learning['du_regs'].append(du_regs)
    learning['v_regs'].append(v_regs)
    learning['h_regs'].append(h_regs)
    learning['dh_regs'].append(dh_regs)
    learning['b_regs'].append(b_regs)

    return net, task, algo, plant, learning
