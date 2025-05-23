function [y, h, u] = run_trial(this, x, y_, eta, learning_rule, online_learning)
%{
        Run the RNN for a single trial.

        Parameters:
        -----------
        x : The input to the network. x[t,i] is input from unit i at timestep t.
        y_ : The target RNN output, where y_[t,i] is output i at timestep t.
        eta : A list of 3 learning rates for w_in, w_cc, and w_out.
        learning : Specify the learning algorithm with one of the following
            strings: 'bptt' or 'rflo'. If None, run the network without
            learning.
        online_learning : If True (and learning is on), update weights at each
            timestep. If False (and learning is on), update weights only at the
            end of each trial. Online learning cannot be used with BPTT.

        Returns:
        --------
        y : The time-dependent network output. y[t,i] is output i at timestep t.
        h : The time-dependent cortical state vector. h(t,i) is unit i at timestep t.
        u : The inputs to cortical units (feedforward plus recurrent) at each
            timestep.
%}

% neural nonlinearity f and f'
f = @(x) tanh(x);                           
df = @(x) 1./(cosh(10*tanh(x/10)).^2);      % the tanh prevents overflow
% f = @(x) x;                           
% df = @(x) x./x;      % the tanh prevents overflow

% Boolean shorthands to specify learning algorithm:
bptt = strcmp(learning_rule, 'bptt');
rflo = strcmp(learning_rule, 'rflo');

% learning rates for w_in, w_cc, and w_out
[eta3, eta2, eta1] = deal(eta(1), eta(2), eta(3)); 
t_max = size(x,1);                                  % number of timesteps
train_in = this.learning_params.train_in;
train_cc = this.learning_params.train_cc;
train_out = this.learning_params.train_out;
fb_type = this.learning_params.fb_type;

% network parameters
n_c = this.network_params.n_c;
n_in = this.network_params.n_in; n_out = this.network_params.n_out;
tau_c = this.network_params.tau_c;
w_in = this.network_params.w_in; 
w_cc = this.network_params.w_cc; 
w_out = this.network_params.w_out;
b = this.network_params.b;

% initial conditions
h0 = this.initial_cond.h0;

% initialize
[dw_in, dw_cc, dw_out] = deal(0, 0, 0);                       % changes to weights
u = zeros(t_max, n_c);                                        % cortical input (feedforward plus recurrent)
h = zeros(t_max, n_c);                                        % time-dependent cortical activity vector
h(1,:) = h0;                                                  % initial cortical state
y = zeros(t_max, n_out);                                      % cortical output
err = zeros(t_max, n_out);                                    % readout error

%% If rflo, eligibility traces p, q, m, and n should have rank 2;
if rflo
    p = zeros(n_c, n_c);
    q = zeros(n_c, n_in);
end

%% initialize
for jj = 1:n_c
    if rflo
        q(jj, :) = df(u(1, jj))*x(1,:)/tau_c(jj);
    end
end

%%
for tt = 1:(t_max-1)
    % cortex
    u(tt+1,:) = w_cc*h(tt,:)' + w_in*x(tt+1,:)';
    h(tt+1,:) = h(tt,:) + (-h(tt,:) + f(u(tt+1,:)))./tau_c';% + normrnd(0,0.01,[1 size(h,2)]);
        
    % cortical output
    y(tt+1,:) = w_out*h(tt+1,:)';
    err(tt+1,:) = y_(tt+1,:) - y(tt+1,:);  % readout error
            
    if rflo
        if train_cc, p = (df(u(tt+1,:))'./tau_c)*h(tt,:) + repmat((1-1./tau_c),[1 n_c]).*p; end
        if train_in, q = (df(u(tt+1,:))'./tau_c)*x(tt,:) + repmat((1-1./tau_c),[1 n_in]).*q; end
    end

    if rflo && online_learning
        if train_out, dw_out = eta1/t_max*(err(tt+1,:)'*h(tt+1,:)); end
        if train_cc, dw_cc = eta4*((b*err(tt+1,:)')*ones(1,n_c)).*p/t_max; end
        if train_in, dw_in = eta5*((b*err(tt+1,:)')*ones(1,n_in)).*q/t_max; end
    elseif rflo && ~online_learning
        if train_out, dw_out = dw_out + eta1/t_max*(err(tt+1,:)'*h(tt+1,:)); end
        if train_cc, dw_cc = dw_cc + eta2*((b*err(tt+1,:)')*ones(1,n_c)).*p/t_max; end
        if train_in, dw_in = dw_in + eta3*((b*err(tt+1,:)')*ones(1,n_in)).*q/t_max; end
    end
    
    if online_learning && ~bptt
        w_out = w_out + dw_out;
        w_cc = w_cc + dw_cc;
        w_in = w_in + dw_in;
        if strcmp(fb_type, 'aligned'), b = (w_out')*(sqrt(n_c)/sqrt(n_out)); end
    end    
    
    % integrate output and feed it back as input through sensory channels
    x(:,3:4) = 20*cumsum(y)*1e-3;
    
end

if bptt  % backward pass for BPTT
    z = zeros(t_max, n_c);
    z(end,:) = (b)*err(end,:)';
    for tt = t_max:-1:2
        z(tt-1,:) = z(tt,:).*(1 - 1./tau_c');
        z(tt-1,:) = z(tt-1,:) + (b*err(tt,:)')';
        z(tt-1,:) = z(tt-1,:) + ((z(tt,:).*df(u(tt,:)))*w_cc)./tau_c';
        
        % Updates for the weights:
        if train_out, dw_out = dw_out + eta1*(err(tt,:)'*h(tt,:))/t_max; end
        if train_cc, dw_cc = dw_cc + eta2./(t_max*repmat(tau_c,[1 n_c])).*((z(tt,:).*df(u(tt,:)))'*h(tt-1,:)); end
        if train_in, dw_in = dw_in + eta3./(t_max*repmat(tau_c,[1 n_in])).*((z(tt,:).*df(u(tt,:)))'*x(tt,:)); end
    end
end
    
if ~online_learning && any(learning_rule)  % wait until end of trial to update weights
    w_out = w_out + dw_out;
    w_cc = w_cc + dw_cc;
    w_in = w_in + dw_in;
    if strcmp(fb_type, 'aligned'), b = (w_out')*(sqrt(n_c)/sqrt(n_out)); end
end

this.network_params.w_out = w_out;
this.network_params.w_cc = w_cc;
this.network_params.w_in = w_in;
this.network_params.b = b;

end

