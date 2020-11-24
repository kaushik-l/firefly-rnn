l=9;

figure; hold on;
for k=1:10
    subplot(2,10,k); hold on;
    plot(bptt_nets{1}(l).task_params.y_out(:,1,k));
    plot(bptt_nets{1}(l).training.activity.y_{k}.post(:,1));
    ylim([-3 3]);
    
    subplot(2,10,k+10); hold on;
    plot(bptt_nets{1}(l).task_params.y_out(:,2,k));
    plot(bptt_nets{1}(l).training.activity.y_{k}.post(:,2));
    ylim([-3 3]);
end

% for l=1:12, loss(l,:) = bptt_nets{1}(l).training.loss; end
% figure; hold on; plot(median(loss));
% set(gca,'XScale','Log','YScale','Log'); box off;

figure; hold on;
for k=1:10
    subplot(2,5,k); hold on;
    h_ref = bptt_nets{1}(1).training.activity.h_{5}.post;
    h = bptt_nets{1}(1).training.activity.h_{k}.post;
    [~,peaktime] = max(h_ref); [~,indx] = sort(peaktime);
    imagesc(h(:,indx)',[-1 1]);
    axis tight;
end