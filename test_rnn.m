function [r2,data] = test_rnn(net)

if nargin < 3, noiselevel = 'low'; end
p=[];
task_params = net.task_params;
x_in = task_params.x_in;
y_out = task_params.y_out;
learning_params = net.learning_params;
eta = [learning_params.eta_in learning_params.eta_cc learning_params.eta_in];
nconds = size(x_in,3);
cmap1 = summer(nconds); cmap2 = spring(nconds);

for i=1:nconds
    x_ = x_in(:,:,i); y_target = y_out(:,:,i);
    [y, h, r] = net.run_trial(x_, y_target, eta, false, false);
    err = y_target(:) - y(:);
    r2(i) = 1 - mean(sum(err.^2,2))/mean(var(y_out,[],3));
    data.x_(i,:,:) = x_; data.y_target(i,:,:) = y_target; 
    data.y(i,:,:) = y; data.h(i,:,:) = h; data.r(i,:,:) = r;
end

r2 = mean(r2);