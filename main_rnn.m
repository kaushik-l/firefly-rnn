% train_tcrnn
% script to train thalamocortical recurrent neural network
% clear;

%% multiple tasks
eta = 1e-3; % 1e-5 for 2000 %bptt
task = 'firefly';
disp(['Training ' task '.....\n']);
    
% BPTT
for k=1:numel(eta)
    % bptt w_tc and w_ct
    disp('......BPTT w_tc and w_ct.....\n');
    % set network, learning, and task parameters
    network_params = []; learning_params = []; initial_cond = []; task_params = [];
    % overwrite default params (check init_tcrnn for default values)
%     learning_params.train_in = true; learning_params.eta_in = eta(k);
    learning_params.train_out = true; learning_params.train_cc = true; 
    learning_params.eta_out = eta(k); learning_params.eta_cc = eta(k);
    learning_params.algorithm = 'bptt'; learning_params.online_learning = false;
    learning_params.fb_type = 'aligned';
    task_params.name = task;
    % initialize, build and train
    try
        bptt_nets{k} = train_rnn(network_params,learning_params,initial_cond,task_params);
    catch ME_bptt
        disp(['......skipping eta = ' num2str(eta(k)) ' .....\n']);
    end
end