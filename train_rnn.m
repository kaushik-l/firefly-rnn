function nets = train_rnn(network_params,learning_params,initial_cond,task_params)

% initialize parameters
[network_params, learning_params, initial_cond, task_params] = ...
    init_rnn(network_params,learning_params,initial_cond,task_params);

% build and train
N_nets = 12;
parfor k = 1:N_nets
    % build network
    net = rnn(network_params, learning_params, initial_cond, task_params);    
    % train network
    [network_prs(k), learning_prs(k), init_cond(k), task_prs(k), training(k)] = ...
        net.run_session(learning_params.algorithm, learning_params.online_learning);
end

% save results
for k=1:N_nets
    nets(k) = rnn(network_params, learning_params, initial_cond, task_params);
    nets(k).network_params = network_prs(k);
    nets(k).learning_params = learning_prs(k);
    nets(k).initial_cond = init_cond(k);
    nets(k).task_params = task_prs(k);
    nets(k).training = training(k);
end