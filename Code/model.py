import numpy as np
import math
from math import sqrt, pi
import numpy.random as npr
import torch


class Network:
    def __init__(self, name='ppc', N=256, S=4, R=2, seed=1):
        self.name = name
        npr.seed(seed)
        # network parameters
        self.N = N  # RNN units
        self.dt = .1  # time bin (in units of tau)
        self.g_in = 1.0  # initial input weight scale
        self.g_rec = 1.0  # initial recurrent weight scale
        self.g_out = 1.0  # initial output weight scale
        self.S = S  # input
        self.R = R  # readout
        self.sig = 0.001  # initial activity noise
        self.z0 = []    # initial condition
        self.xa, self.ha, self.ua = [], [], []  # input, activity, output
        self.ws = (2 * npr.random((N, S)) - 1)/sqrt(S)  # input weights
        self.J = self.g_rec * npr.standard_normal([N, N]) / np.sqrt(N)  # recurrent weights
        self.wr = (2 * npr.random((R, N)) - 1) / sqrt(N)  # readout weights

    # nlin
    def f(self, x):
        return np.tanh(x) if not torch.is_tensor(x) else torch.tanh(x)

    # derivative of nlin
    def df(self, x):
        return 1 - (np.tanh(x) ** 2) if not torch.is_tensor(x) else 1 - (torch.tanh(x) ** 2)


class Task:
    def __init__(self, name='firefly', duration=60, rand_tar=False, dist_bounds=(1, 6), num_tar=None, feed_pos=False, feed_belief=False,
                 rand_init=False, dt=0.1, lambda_b=0, lambda_u=0, lambda_du=0, lambda_v=0, lambda_vmax=0, lambda_h=0, lambda_dh=0, seed=1):
        self.name = name
        npr.seed(seed)
        self.rand_init = rand_init
        NT = int(duration / dt)
        self.lambda_b, self.lambda_u, self.lambda_du, self.lambda_v, self.lambda_vmax, self.lambda_h, self.lambda_dh = \
            lambda_b, lambda_u, lambda_du, lambda_v, lambda_vmax, lambda_h, lambda_dh
        if feed_belief: self.lambda_b = 1e-1
        # task parameters
        if self.name == 'firefly':
            self.T, self.dt, self.NT = duration, dt, NT
            self.duration_tar, self.duration_ramp, self.duration_stop = int(3 / dt), int(3 / dt), int(6 / dt)
            self.num_tar = num_tar
            self.rand_tar = rand_tar
            self.dist_bounds = dist_bounds
            self.angle_bounds = (-np.pi / 5, np.pi / 5)
            self.v_bounds = (0, 0.2)    # m per tau
            if feed_pos or feed_belief:
                self.s = 0.0 * np.ones((6, self.num_tar, NT)) if num_tar is not None else 0.0 * np.ones((6, NT))
            else:
                self.s = 0.0 * np.ones((4, self.num_tar, NT)) if num_tar is not None else 0.0 * np.ones((4, NT))
            self.ustar = 0.0 * np.ones((NT, self.num_tar, 2)) if num_tar is not None else 0.0 * np.ones((NT, 2))

    def loss(self, err, ut, vt, ht, bt, dt):
        u1 = ut[:, :2]
        u2 = ut[:, 2:] if self.lambda_b else 0
        mse = (err ** 2).mean() / 2
        u_reg = self.lambda_u * ((u1 ** 2).sum(dim=-1).mean() + (u1[0] ** 2).sum(dim=-1))
        du_reg = self.lambda_du * ((torch.diff(ut, dim=0) / dt) ** 2).sum(dim=-1).mean()
        v_reg = self.lambda_v * (vt[-self.duration_stop:, :] ** 2).sum(dim=-1).mean() + \
                1e-2 * (vt[vt[:, 0] > self.v_bounds[-1], 0]).sum() - \
                1e-2 * (vt[vt[:, 0] < self.v_bounds[0], 0]).sum()
        h_reg = self.lambda_h * ((abs(ht)).sum(dim=-1) ** 2).mean()
        dh_reg = self.lambda_dh * ((torch.diff(ht, dim=0) / dt) ** 2).sum(dim=-1).mean()
        b_reg = self.lambda_b * ((bt - u2) ** 2).mean() / 2
        return mse, u_reg, du_reg, v_reg, h_reg, dh_reg, b_reg


class Plant:
    def __init__(self, name='joystick', sensory_noise=0.2, process_noise=0.2, seed=1):
        self.name = name
        npr.seed(seed)
        if self.name == 'joystick':
            # physics parameters
            self.sensory_noise = sensory_noise  # sensory input multiplicative noise scale
            self.process_noise = process_noise  # motor output multiplicative noise scale
            self.noise_corr_time = 3  # noise correlation time (units of tau)
            self.v_init = [0.0, 0.0]  # initial position of endpoint

    # plant dynamics (actual dynamics with noise)
    def forward(self, u, v, x, phi, process_noise, dt):
        # physics
        v_true, v_sense = [], []
        if self.name == 'joystick':
            # true velocity
            accel = (1 / 10) * u * (1 + process_noise) - 0.01 * v  # to compensate for the scaling factor in plan_traj()
            v_true = v + accel * dt
            # sensed velocity
            sensory_noise = torch.as_tensor(npr.randn(len(accel), 1)) * self.sensory_noise
            v_sense = v_true * (1 + sensory_noise)
            # true position
            x = x + v_true[0] * torch.vstack((torch.sin(phi), torch.cos(phi))) * dt
            phi = phi + v_true[1] * dt
            if phi > np.pi:
                phi = phi - 2 * torch.tensor(np.pi)
            elif phi <= -np.pi:
                phi = phi + 2 * torch.tensor(np.pi)
        # return
        return v_true, v_sense, x, phi

    # predicted plant dynamics (predicted dynamics unaware of noise)
    def forward_predict(self, u, v, w, dt):
        x_predict = None
        # physics
        if self.name == 'TwoLink':
            accel = u
            v_new = v + accel * dt
            w = w + v * dt + 0.5 * accel * dt ** 2
            v = v_new
            # hand location
            ang1, ang2 = w[:, 0], w.sum(dim=-1)
            x_predict = np.stack((np.cos(ang1) + np.cos(ang2), np.sin(ang1) + np.sin(ang2)), dim=-1)
        # return
        return x_predict


class Algorithm:
    def __init__(self, name='Adam', Nepochs=10000, lr=1e-3):
        self.name = name
        # learning parameters
        self.Nepochs = Nepochs
        self.Nstart_anneal = 30000
        self.lr = lr  # learning rate
        self.annealed_lr = 1e-6
