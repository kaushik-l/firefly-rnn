function [network_params,learning_params,initial_cond,task_params,training] = ...
    run_session(this, learning_rule, online_learning)

%{
Run the RNN for a session consisting of many trials.

Parameters:
-----------
learning_rule : Specify the learning algorithm with one of the following
    strings: 'rtrl', 'bptt', or 'rflo'. If None, run the network without
    learning.
online_learning : If True (and learning is on), update weights at each
    timestep. If False (and learning is on), update weights only at the
    end of each trial. Online learning cannot be used with BPTT.

Returns:
--------
y : The RNN output.
loss_list : A list with the value of the loss function for each trial.
readout_alignment : The normalized dot product between the vectorized
error feedback matrix and the readout matrix, as in Lillicrap et al
(2016).
%}

if nargin < 3, online_learning = false; end
if nargin < 2, learning_rule = 'rflo'; online_learning = false; end

ntrls = this.learning_params.ntrls;
eta = [this.learning_params.eta_in this.learning_params.eta_cc this.learning_params.eta_out];

n_in = this.network_params.n_in;
n_c = this.network_params.n_c;
n_out = this.network_params.n_out;

x_in = this.task_params.x_in;
y_out = this.task_params.y_out;

t_max = size(this.task_params.x_in,1);  % number of timesteps
loss_list = nan(1,ntrls);
overlap.b__w_out = [];

% check if number of targets match number of input
if size(x_in,3) == size(y_out,3), num_inputs = size(x_in,3);
else, error('number of inputs and outputs'); end
indx = randi(num_inputs,[ntrls 1]);

count = 0;
for ii = 1:ntrls
    x_ = x_in(:,:,indx(ii)); y_target = y_out(:,:,indx(ii));
    [y, h] = this.run_trial(x_, y_target, eta/log10(ii+1), learning_rule, online_learning);
    
    % loss
    err = y_target - y;
    loss = 0.5*mean(sum(err.^2,2));
    loss_list(ii) = loss;
    
    % Flatten the random feedback matrix to check for feedback alignment:
    bT_flat = this.network_params.b(:);
    bT_flat = bT_flat/norm(bT_flat);
    w_out_flat = (this.network_params.w_out)'; w_out_flat = w_out_flat(:);
    w_out_flat = w_out_flat/norm(w_out_flat);
    overlap.b__w_out = [overlap.b__w_out bT_flat'*w_out_flat];
        
    % output before and after training
    for k=1:num_inputs
        if ii==find(indx==k,1,'first')
            y_{k}.pre = y;
            h_{k}.pre = h;
        elseif ii==find(indx==k,1,'last')
            y_{k}.post = y;
            h_{k}.post = h;
        end
    end
    
    %% display
    if mod(ii,10000)==0, fprintf([num2str(ii) '/' num2str(ntrls) '  Loss: ' num2str(loss) '\n']); end
end

% save
this.training.activity.y_ = y_;
this.training.activity.h_ = h_;
this.training.loss = loss_list;
this.training.overlap = overlap;

% output
network_params = this.network_params;
learning_params = this.learning_params;
initial_cond = this.initial_cond;
task_params = this.task_params;
training = this.training;