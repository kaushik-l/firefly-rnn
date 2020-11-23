function [network_params, learning_params, initial_cond, task_params] = ...
    init_rnn(network_params, learning_params, initial_cond, task_params)

%% network parameters
n_in = 1;
n_c = 100;
n_out = 1;
g_cc = 1.2;
if exist('network_params','var') && isstruct(network_params), fields = fieldnames(network_params);
    for i=1:numel(fields), eval([fields{i} '= network_params.(fields{i});']); end; end
tau_c = 10*ones(n_c,1);

%% learning_parameters
train_in = false;
train_cc = false;
train_out = false;
fb_type = 'random';
eta_out = 0.1;
eta_cc = 0.1;
eta_in = 0.1;
ntrls = 30000;
algorithm = 'rflo';
online_learning = true;
if exist('learning_params','var') && isstruct(learning_params), fields = fieldnames(learning_params);
    for i=1:numel(fields), eval([fields{i} '= learning_params.(fields{i});']); end; end

%% initial condition
h0 = rand(n_c,1) - 0.5;     % initial state of the cortex
if exist('initial_cond','var') && isstruct(initial_cond), fields = fieldnames(initial_cond);
    for i=1:numel(fields), eval([fields{i} '= initial_cond.(fields{i});']); end; end

%% output
% network params
network_params.n_in = n_in;
network_params.n_c = n_c;
network_params.n_out = n_out;
network_params.tau_c = tau_c;
network_params.g_cc = g_cc;

% learning params
learning_params.train_in = train_in;
learning_params.train_cc = train_cc;
learning_params.train_out = train_out;
learning_params.fb_type = fb_type;
learning_params.eta_out = eta_out;
learning_params.eta_cc = eta_cc;
learning_params.eta_in = eta_in;
learning_params.ntrls = ntrls;
learning_params.algorithm = algorithm;
learning_params.online_learning = online_learning;

% initial condition
initial_cond.h0 = h0;

% task parameters
name = 'firefly';
if exist('task_params','var') && isstruct(task_params), fields = fieldnames(task_params);
    for i=1:numel(fields), eval([fields{i} '= task_params.(fields{i});']); end; end
[n_in, n_out, x_in, y_out] = gen_task(name);
network_params.n_in = n_in;
network_params.n_out = n_out;
task_params.x_in = x_in;
task_params.y_out = y_out;

function [n_in, n_out, x_in, y_out] = gen_task(taskname)
% function to generate inputs and outputs for various tasks
switch taskname
    case 'sinewave'
        n_in = 1; n_out = 1;
        duration = 600;  % number of timesteps in one period
        cycles = 0.25;
        x_in = 0.0*ones(duration, n_in);
        y_out = (sin(2*pi*(1:duration)/(duration*cycles)) + ...
            0.5*sin(2*2*pi*(1:duration)/(duration*cycles)) + ...
            0.25*sin(4*2*pi*(1:duration)/(duration*cycles)))'*ones(1,n_out);  
    case 'firefly'
        n_in = 4; n_out = 2;
        duration = 700; pulseduration = 50; rampduration = 50;
        nconds = 6;
        x_in = 0.0*ones(duration, n_in, nconds);
        y_out = 0.0*ones(duration, n_out, nconds);
        
        % define inputs
        x_amp = [0.0 0.5 -0.5 0.0 1.0 -1.0]; y_amp = [0.5 0.5 0.5 1.0 1.0 1.0];
        for k=1:nconds
            x_in(1:pulseduration,1,k) = x_amp(k); % firefly x-coord
            x_in(1:pulseduration,2,k) = y_amp(k); % firefly y-coord
        end
        
        % define output
        for k=1:nconds
            delay_x = 0; delay_y = 0;
            if abs(x_amp(k))>0, delay_x = 500*(abs(x_amp(k)) - 0.1); end
            if abs(y_amp(k))>0, delay_y = 500*(abs(y_amp(k)) - 0.1); end
            
            y_out(pulseduration + (1:rampduration),1,k) = 2.0*sign(x_amp(k));
            y_out(pulseduration + (1:rampduration),2,k) = 2.0*sign(y_amp(k));
            y_out((pulseduration + rampduration + delay_x):...
                (pulseduration + rampduration + delay_x + pulseduration),1,k) = -2.0*sign(x_amp(k));
            y_out((pulseduration + rampduration + delay_y):...
                (pulseduration + rampduration + delay_y + pulseduration),2,k) = -2.0*sign(y_amp(k));
        end
        
end