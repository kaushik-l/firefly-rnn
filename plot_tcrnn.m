function p = plot_tcrnn(nets,plottype,plotmedian,fig_num,cmap)

N_netsizes = 1;
% p = [];
figure(fig_num{1}); hold on; 
% if ~isempty(fig_num{2}), subplot(fig_num{2}(1),fig_num{2}(2),fig_num{2}(3)); hold on; end
if ~iscell(nets) || ~(numel(nets) == N_netsizes), nets = {nets}; N_netsizes = 1; end
if iscell(nets{1}), N_hyperparams = numel(nets{1}); else, N_hyperparams = 1; end

for j=1:N_netsizes
    
    if ~any(strcmp(plottype,{'r2hyper','r2best'}))
        
        if N_hyperparams > 1
            N_nets = numel(nets{j}{1}); clear r2best;
            for k=1:numel(nets{j})
                if ~isempty(nets{j}{k})
                    for i=1:N_nets, r2_vals(i) = test_tcrnn(nets{j}{k}(i), false); end
                    r2best(k) = median(r2_vals);
                else
                    r2best(k) = 0;
                end
            end
            [~,bestparam] = max(r2best);
            nets{j} = nets{j}{bestparam};
        else
            if iscell(nets{j}), nets{j} = nets{j}{end}; end
        end        
        N_nets = numel(nets{j});
        n_in = nets{j}(1).network_params.n_in;
        n_c = nets{j}(1).network_params.n_c;
        n_t = nets{j}(1).network_params.n_t;
        n_out = nets{j}(1).network_params.n_out;
        network_params = [nets{j}.network_params];
        learning_params = [nets{j}.learning_params];
        initial_cond = [nets{j}.initial_cond];
        task_params = [nets{j}.task_params];
        training = [nets{j}.training];
    
    end
    
    switch lower(plottype)
        
        case 'learningcurve'
            if plotmedian
                p(j) = plot(1:learning_params(1).ntrls,median(cell2mat({training.loss}')),'Linewidth',2,...
                    'Color',cmap(j,:)); grid on;
            else
                indx = randperm(N_nets,1);
                p(j) = plot(1:learning_params(1).ntrls,training(indx).loss,'Linewidth',2,...
                    'Color',cmap(j,:)); grid on;
            end
            if j == N_netsizes
                set(gca,'XScale','Linear','YScale','Log'); xlim([0 1e4]); ax = gca; ax.XAxis.Exponent = 3;
                set(gca,'TickDir','Out','Ticklength',[.03 .03],'Fontsize',14); box off;
                xlabel('Trial','Fontsize',18); ylabel('MSE','Fontsize',18);
            end
            
        case 'fbalign'
            overlap = [training.overlap];
            if plotmedian
                p(j) = plot(1:learning_params(1).ntrls,median(cell2mat({overlap.b__w_out}')),'Linewidth',2,...
                    'Color',cmap(j,:)); grid on;
            else
                indx = randperm(N_nets,1);
                p(j) = plot(1:learning_params(1).ntrls,overlap(indx).b__w_out,'Linewidth',2,...
                    'Color',cmap(j,:)); grid on;
            end
            if j == N_netsizes
                set(gca,'XScale','Linear','YScale','Linear');
                xlim([0 1e4]); ylim([-.2 1]); ax = gca; ax.XAxis.Exponent = 3;
                set(gca,'TickDir','Out','Ticklength',[.03 .03],'Fontsize',14); box off;
                xlabel('Trial','Fontsize',18); ylabel('Feedback alignment','Fontsize',18);
            end
            
        case 'fbalign_post'
            overlap = [training.overlap];
            if plotmedian
                b__w_out = median(cell2mat({overlap.b__w_out}')); 
                fbalignment(j) = median(b__w_out(end-100:end));
            else
                indx = randperm(N_nets,1); b__w_out = overlap(indx).b__w_out;
                fbalignment(j) = median(b__w_out(end-100:end));
            end
            if j == N_netsizes
                p = plot(2.^(0:N_netsizes-1)/2^(N_netsizes-1),fbalignment,'Marker','o',...
                    'Linewidth',2,'Color',cmap(4,:));
                set(p, 'markerfacecolor', get(p, 'color'));
                set(gca,'XScale','Linear','YScale','Linear');
                xlim([0 1]); ylim([-.2 1]); set(gca,'XScale','Log'); grid on; %ylim([0.75 1]);
                set(gca,'TickDir','Out','Ticklength',[.03 .03],'Fontsize',14); box off;
                xlabel('Compression ratio, N_t/N_c','Fontsize',18); 
                ylabel('Feedback alignment','Fontsize',18);
            end
            
        case 'r2'
            for i=1:N_nets, r2_vals(i) = test_tcrnn(nets{j}(i), false); end
            if plotmedian, r2(j) = median(r2_vals); r2_mu(j) = median(r2_vals); r2_sem(j) = std(r2_vals)/sqrt(N_nets);
            else, indx = randperm(N_nets,1); r2(j) = r2_vals(indx);
            end
            if j == N_netsizes
                p = plot(2.^(0:N_netsizes-1)/2^(N_netsizes-1),r2,'Marker','o',...
                    'Linewidth',2,'Color',cmap(4,:));
                set(p, 'markerfacecolor', get(p, 'color'));
                set(gca,'XScale','Linear','YScale','Linear');
                xlim([0 1]); set(gca,'XScale','Log'); grid on; %ylim([0.75 1]); 
                set(gca,'TickDir','Out','Ticklength',[.03 .03],'Fontsize',14); box off;
                xlabel('Compression ratio, N_t/N_c','Fontsize',18); ylabel('R^2','Fontsize',18);
            end
            
        case 'r2hyper'
            for k = 1:N_hyperparams
                N_nets = numel(nets{j}{k});
                network_params = [nets{j}{k}.network_params];
                learning_params = [nets{j}{k}.learning_params];
                initial_cond = [nets{j}{k}.initial_cond];
                task_params = [nets{j}{k}.task_params];
                training = [nets{j}{k}.training];
                if ~isempty(nets{j}{k})
                    for i=1:N_nets, r2_vals(i) = test_tcrnn(nets{j}{k}(i), false); end
                    r2best(k) = median(r2_vals);
                else
                    r2best(k) = nan;
                end
                eta_tc(k) = learning_params(1).eta_tc;
            end
            p = plot(eta_tc,r2best,'Marker','o','Linewidth',2,'Color',cmap);
            set(p, 'markerfacecolor', get(p, 'color'));
            set(gca,'XScale','Linear','YScale','Linear');
            xlim([0 1]); ylim([0.1 1]); set(gca,'XScale','Log'); grid on;
            set(gca,'TickDir','Out','Ticklength',[.03 .03],'Fontsize',14); box off;
            xlabel('Learning rate, \eta_{tc}','Fontsize',18); ylabel('R^2','Fontsize',18);
            
        case 'r2best'
            for k = 1:N_hyperparams
                N_nets = numel(nets{j}{k});
                network_params = [nets{j}{k}.network_params];
                learning_params = [nets{j}{k}.learning_params];
                initial_cond = [nets{j}{k}.initial_cond];
                task_params = [nets{j}{k}.task_params];
                training = [nets{j}{k}.training];
                if ~isempty(nets{j}{k})
                    for i=1:N_nets, r2_vals(i) = test_tcrnn(nets{j}{k}(i), false); end
                    r2best(k) = median(r2_vals);
                else
                    r2best(k) = nan;
                end
                eta_tc(k) = learning_params(1).eta_tc;
            end
            x = get(gca,'xlim');
            p = plot(x + 1e-6,[max(r2best) max(r2best)],'Linewidth',2,'Color',cmap);
            set(p, 'markerfacecolor', get(p, 'color'));
            set(gca,'XScale','Linear','YScale','Linear');
            xlim([0 1]); ylim([0.1 1]); set(gca,'XScale','Log'); grid on;
            set(gca,'TickDir','Out','Ticklength',[.03 .03],'Fontsize',14); box off;
            xlabel('Learning rate, \eta_{tc}','Fontsize',18); ylabel('R^2','Fontsize',18);
            
        case 'input-output'
            for i=1:N_nets, r2(i) = test_tcrnn(nets{j}(i), false); end
            [~,indx] = max(r2); [r2, data] = test_tcrnn(nets{j}(indx), false);
            switch task_params(1).name
                case 'sine_wave_simple'
                    p(1) = plot(data.y, 'Linewidth',2, 'Color', cmap(4,:)); 
                    p(2) = plot(data.y_target,'--k', 'Linewidth',2);
                    set(gca,'TickDir','Out','Ticklength',[.03 .03],'Fontsize',14); box off;
                    xlabel('Time','Fontsize',18); ylabel('Input','Fontsize',18);
                    legend(p(2),'target','Fontsize',18); legend boxoff;
                case 'vanderpol_oscillator'
                    p(1) = plot(data.y(1,:,1), data.y(1,:,2), 'Linewidth',2, 'Color', cmap(4,:)); 
                    p(2) = plot(data.y_target(1,:,1), data.y_target(1,:,2), '--k', 'Linewidth',2);
                    set(gca,'TickDir','Out','Ticklength',[.03 .03],'Fontsize',14); box off;
                    xlabel('y_1','Fontsize',18); ylabel('y_2','Fontsize',18);
                case 'lorenz_attractor'
                    p(1) = plot3(data.y(1,:,1), data.y(1,:,2), data.y(1,:,3), 'Linewidth',2, 'Color', cmap(4,:));
                    p(2) = plot3(data.y_target(1,:,1), data.y_target(1,:,2), data.y_target(1,:,3), '--k', 'Linewidth',2);
                    set(gca,'TickDir','Out','Ticklength',[.03 .03],'Fontsize',14); box off;
                    xlabel('y_1','Fontsize',18); ylabel('y_2','Fontsize',18); zlabel('y_3','Fontsize',18);
                case 'delayed_xor'
                    n_conds = size(data.x_, 1);
                    for i=1:n_conds
                        subplot(2,n_conds,i); plot(data.x_(i,:),'Linewidth',2); 
                        ylim([-1 1]); hline(0,'k');
                        set(gca,'TickDir','Out','Ticklength',[.03 .03],'Fontsize',14); box off;
                        if i==1, xlabel('Time','Fontsize',18); ylabel('Input','Fontsize',18); end
                        subplot(2,n_conds,n_conds+i); hold on;
                        plot(data.y(i,:),'Linewidth',2); 
                        plot(data.y_target(i,:),'--k','Linewidth',2);
                        set(gca,'TickDir','Out','Ticklength',[.03 .03],'Fontsize',14); box off;
                        if i==1, xlabel('Time','Fontsize',18); ylabel('Output','Fontsize',18); end
                    end
                case 'detection'
                    for i=1:N_nets, r2(i) = test_tcrnn(nets{j}(i), false, 'low'); end
                    [~,indx] = max(r2); [~, data_lownoise] = test_tcrnn(nets{j}(indx), false, 'low');
                    for i=1:N_nets, r2(i) = test_tcrnn(nets{j}(i), false, 'high'); end
                    [~,indx] = max(r2); [~, data_highnoise] = test_tcrnn(nets{j}(indx), false, 'high');
                    n_conds = size(data.x_, 1);
                    for i=1:n_conds
                        % below threshold
                        subplot(2,2*n_conds,i); plot(data_lownoise.x_(i,:),'Linewidth',2);
                        ylim([0 2]); hline(1,'k');
                        set(gca,'TickDir','Out','Ticklength',[.03 .03],'Fontsize',14); box off;
                        if i==1, xlabel('Time','Fontsize',18); ylabel('Input','Fontsize',18); end
                        subplot(2,2*n_conds,i+2); plot(data_highnoise.x_(i,:),'Linewidth',2);
                        ylim([0 2]); hline(1,'k');
                        set(gca,'TickDir','Out','Ticklength',[.03 .03],'Fontsize',14); box off;
                        % above threshold
                        subplot(2,2*n_conds,2*n_conds+i); hold on;
                        plot(data_lownoise.y(i,:),'Linewidth',2);
                        plot(data_lownoise.y_target(i,:),'--k','Linewidth',2); ylim([-.5 1.5]);
                        set(gca,'TickDir','Out','Ticklength',[.03 .03],'Fontsize',14); box off;
                        if i==1, xlabel('Time','Fontsize',18); ylabel('Output','Fontsize',18); end
                        subplot(2,2*n_conds,2*n_conds+i+2); hold on;
                        plot(data_highnoise.y(i,:),'Linewidth',2);
                        plot(data_highnoise.y_target(i,:),'--k','Linewidth',2); ylim([-.5 1.5]);
                        set(gca,'TickDir','Out','Ticklength',[.03 .03],'Fontsize',14); box off;
                    end
                case 'contextual_averaging'
                    n_conds = size(data.x_, 1);
                    for i=1:n_conds
                        % input
                        subplot(3,n_conds,i); plot(squeeze(data.x_(i,:,1:2)),'Linewidth',2);
                         ylim([-1.5 1.5]); hline(0,'k');
                        set(gca,'TickDir','Out','Ticklength',[.03 .03],'Fontsize',14); box off;
                        if i==1, xlabel('Time','Fontsize',18); ylabel('Input','Fontsize',18); end
                        % context
                        subplot(3,n_conds,n_conds+i); plot(squeeze(data.x_(i,:,3:4)),'Linewidth',2);
                         ylim([-1.5 1.5]); hline(0,'k');
                        set(gca,'TickDir','Out','Ticklength',[.03 .03],'Fontsize',14); box off;
                        if i==1, xlabel('Time','Fontsize',18); ylabel('Context','Fontsize',18); end
                        % output
                        subplot(3,n_conds,2*n_conds+i); hold on;
                        plot(data.y(i,:),'Linewidth',2);
                        plot(data.y_target(i,:),'--k','Linewidth',2); ylim([-1.5 1.5]);
                        set(gca,'TickDir','Out','Ticklength',[.03 .03],'Fontsize',14); box off;
                        if i==1, xlabel('Time','Fontsize',18); ylabel('Output','Fontsize',18); end
                    end
            end
            
        case 'pccortex'
            for i=1:N_nets, r2(i) = test_tcrnn(nets{j}(i), false); end
            [~,indx] = sort(r2,'descend');
            pcs = [training.pcs]; pc_score = {pcs.score};
            npcs = size(pc_score{1},2);
            pc_score = pc_score(indx);
            for i=1:N_nets
                for k=1:npcs
                    subplot(N_nets, npcs, npcs*(i-1)+k); hold on;
                    p(i,k) = plot(pc_score{i}(:,k),'Color',cmap(3,:),'Linewidth',2); 
                    box off; axis off;
                end
            end
            
        case 'pcoverlap'
            overlap = [training.overlap];
            % pc__w_in
            for i=1:N_nets, pc__w_in(i,:,:) = abs(squeeze(overlap(i).pc__w_in(end,:,:))); end
            pc__w_in_mu = squeeze(mean(pc__w_in)); pc__w_in_sem = squeeze(std(pc__w_in))/sqrt(N_nets);
            % pc__w_tc
            for i=1:N_nets, pc__w_tc(i,:,:) = abs(squeeze(overlap(i).pc__w_tc(end,:,:))); end
            pc__w_tc_mu = squeeze(mean(pc__w_tc)); pc__w_tc_sem = squeeze(std(pc__w_tc))/sqrt(N_nets);
            % pc__w_ct
            for i=1:N_nets, pc__w_ct(i,:,:) = abs(squeeze(overlap(i).pc__w_ct(end,:,:))); end
            pc__w_ct_mu = squeeze(mean(pc__w_ct)); pc__w_ct_sem = squeeze(std(pc__w_ct))/sqrt(N_nets);
            % pc__w_out
            for i=1:N_nets, pc__w_out(i,:,:) = abs(squeeze(overlap(i).pc__w_out(end,:,:))); end
            pc__w_out_mu = squeeze(mean(pc__w_out)); pc__w_out_sem = squeeze(std(pc__w_out))/sqrt(N_nets);

            % 1st row
            subplot(2,2,1); hold on;
            p{1} = plot(pc__w_in_mu','LineWidth',2); 
            for k=1:size(pc__w_in_mu,2), p{1}(k).Color = cmap(k,:); end
            set(gca,'TickDir','Out','Ticklength',[.03 .03],'Fontsize',14); box off;
            set(gca,'XScale','Log','YScale','Linear'); axis([1 n_c 0 0.8]);
            xlabel('Component','Fontsize',18); ylabel('Overlap','Fontsize',18);
            title('w_{in}','Fontsize',18);
            subplot(2,2,2); hold on;
            p{2} = plot(pc__w_tc_mu,'LineWidth',2);
            for k=1:size(pc__w_tc_mu,2), p{2}(k).Color = cmap(k,:); end
            set(gca,'TickDir','Out','Ticklength',[.03 .03],'Fontsize',14); box off;
            set(gca,'XScale','Log','YScale','Linear'); axis([1 n_c 0 0.8]);
            xlabel('Component','Fontsize',18); title('w_{tc}','Fontsize',18);
            % 2nd row
            subplot(2,2,3); hold on;
            p{3} = plot(pc__w_ct_mu','LineWidth',2);
            for k=1:size(pc__w_ct_mu',2), p{3}(k).Color = cmap(k,:); end
            set(gca,'TickDir','Out','Ticklength',[.03 .03],'Fontsize',14); box off;
            set(gca,'XScale','Log','YScale','Linear'); axis([1 n_c 0 0.8]);
            xlabel('Component','Fontsize',18); ylabel('Overlap','Fontsize',18);
            title('w_{ct}','Fontsize',18);
            subplot(2,2,4); hold on;
            p{4} = plot(pc__w_out_mu,'LineWidth',2);
            for k=1:size(pc__w_out_mu,2), p{4}(k).Color = cmap(k,:); end
            set(gca,'TickDir','Out','Ticklength',[.03 .03],'Fontsize',14); box off;
            set(gca,'XScale','Log','YScale','Linear'); axis([1 n_c 0 0.8]);
            xlabel('Component','Fontsize',18); title('w_{out}','Fontsize',18);
%             p = subplot(3,4,12); hold on;
%             h = pcolor(1:64,1:64,(pc__w_ct_mu*pc__w_tc_mu'));
%             h.EdgeColor = 'None';
%             set(gca,'XScale','Log','YScale','Log')
%             colormap(p,cmap3);
            
        case 'overlap'
            overlap = [training.overlap];
            % w_in__w_tc
            for i=1:N_nets, w_in__w_tc(i,:,:) = abs(squeeze(overlap(i).w_in__w_tc(end,:,:))); end
            w_in__w_tc_mu = squeeze(mean(w_in__w_tc)); w_in__w_tc_sem = squeeze(std(w_in__w_tc))/sqrt(N_nets);
            % w_in__w_ct
            for i=1:N_nets, w_in__w_ct(i,:,:) = abs(squeeze(overlap(i).w_in__w_ct(end,:,:))); end
            w_in__w_ct_mu = squeeze(mean(w_in__w_ct)); w_in__w_ct_sem = squeeze(std(w_in__w_ct))/sqrt(N_nets);
            % w_out__w_tc
            for i=1:N_nets, w_out__w_tc(i,:,:) = abs(squeeze(overlap(i).w_out__w_tc(end,:,:))); end
            w_out__w_tc_mu = squeeze(mean(w_out__w_tc)); w_out__w_tc_sem = squeeze(std(w_out__w_tc))/sqrt(N_nets);
            % w_out__w_ct
            for i=1:N_nets, w_out__w_ct(i,:,:) = abs(squeeze(overlap(i).w_out__w_ct(end,:,:))); end
            w_out__w_ct_mu = squeeze(mean(w_out__w_ct)); w_out__w_ct_sem = squeeze(std(w_out__w_ct))/sqrt(N_nets);
            % w_tc__w_ct
            for i=1:N_nets, w_tc__w_ct(i,:,:) = abs(squeeze(overlap(i).w_tc__w_ct(end,:,:))); end
            w_tc__w_ct_mu = squeeze(mean(w_tc__w_ct)); w_tc__w_ct_sem = squeeze(std(w_tc__w_ct))/sqrt(N_nets);
            
            % w_in__w_tc_mu
            subplot(2,4,1); hold on;
            p{1} = bar(w_in__w_tc_mu); ylim([0 1]);
            for k=1:numel(p{1}), p{1}(k).FaceColor = cmap(2+k,:); end
            set(gca,'TickDir','Out','Ticklength',[.03 .03],'Fontsize',14); box off;
            xlabel('Thalamic units','Fontsize',18); ylabel('Overlap with Input','Fontsize',18);
            title('w_{tc}','Fontsize',18); hline(0.1,'k');
            % w_in__w_ct_mu
            subplot(2,4,2); hold on;
            p{1} = bar(w_in__w_ct_mu); ylim([0 1]);
            for k=1:numel(p{1}), p{1}(k).FaceColor = cmap(2+k,:); end
            set(gca,'TickDir','Out','Ticklength',[.03 .03],'Fontsize',14); box off;
            title('w_{ct}','Fontsize',18); hline(0.1,'k');      
            % w_out__w_tc_mu
            subplot(2,4,5); hold on;
            p{2} = bar(w_out__w_tc_mu); ylim([0 1]);
            for k=1:numel(p{2}), p{2}(k).FaceColor = cmap(2+k,:); end
            set(gca,'TickDir','Out','Ticklength',[.03 .03],'Fontsize',14); box off;
            xlabel('Thalamic units','Fontsize',18); ylabel('Overlap with Output','Fontsize',18);
            title('w_{tc}','Fontsize',18); hline(0.1,'k');
            % w_out__w_ct_mu
            subplot(2,4,6); hold on;
            p{2} = bar(w_out__w_ct_mu); ylim([0 1]);
            for k=1:numel(p{2}), p{2}(k).FaceColor = cmap(2+k,:); end
            set(gca,'TickDir','Out','Ticklength',[.03 .03],'Fontsize',14); box off;
            title('w_{ct}','Fontsize',18); hline(0.1,'k');
            % recurrent
            subplot(2,4,[3 4 7 8]); hold on;
            p{3} = bar(w_tc__w_ct_mu); colormap(cmap); ylim([0 1]);
            for k=1:numel(p{3}), p{3}(k).FaceColor = cmap(2+k,:); end
            set(gca,'TickDir','Out','Ticklength',[.03 .03],'Fontsize',14); box off;
            xlabel('Thalamic units','Fontsize',18); ylabel('Overlap with w_{ct}','Fontsize',18);
            title('w_{tc}','Fontsize',18); hline(0.1,'k');
            
        case 'recurrent_strength'
            ntrls = learning_params(1).ntrls;
            overlap = [training.overlap];
            % w_tc__w_ct
            for i=1:N_nets, w_tc__w_ct(i,:,:,:) = abs(squeeze(overlap(i).w_tc__w_ct)); end
            w_tc__w_ct_mu = squeeze(mean(w_tc__w_ct)); w_tc__w_ct_sem = squeeze(std(w_tc__w_ct))/sqrt(N_nets);
            
            % w_tc__w_ct_mu
            epoch_len = size(w_tc__w_ct_mu,1);
            n_t = size(w_tc__w_ct_mu,3);
            for m=1:n_t
                for n=1:n_t
                    subplot(n_t,n_t,(m-1)*n_t + n); hold on;
                    p(m,n) = plot(epoch_len:epoch_len:ntrls,w_tc__w_ct_mu(:,m,n),...
                        'Linewidth',2,'Color',cmap(4,:));
                    set(gca,'TickDir','Out','Ticklength',[.03 .03],'Fontsize',14); box off;
                    hline(0.1,'k'); %ylim([0 0.5]);
                    if m==1 && n==1, xlabel('Trial','Fontsize',18); ylabel('Overlap','Fontsize',18); end
                end
            end
            sgtitle('Recurrent strength (w_{ct} - w_{tc} overlap)','Fontsize',18);
            
        case 'recurrent_matrix'
            ntrls = learning_params(1).ntrls;
            overlap = [training.overlap];
            % w_tc__w_ct
            for i=1:N_nets, w_tc__w_ct(i,:,:) = abs(squeeze(overlap(i).w_tc__w_ct(end,:,:))); end
            w_tc__w_ct_mu = squeeze(mean(w_tc__w_ct)); w_tc__w_ct_sem = squeeze(std(w_tc__w_ct))/sqrt(N_nets);
            
            % w_tc__w_ct_mu
            hold on;
            p = imagesc(w_tc__w_ct_mu,[0 0.5]); axis tight;
            colormap(gca, winter); colorbar;
            set(gca,'TickDir','Out','Ticklength',[.03 .03],'Fontsize',14); box off;
            xlabel('Neuron','Fontsize',18); ylabel('Neuron','Fontsize',18);
            sgtitle('Recurrent strength (w_{ct} - w_{tc} overlap)','Fontsize',18);
        
        case 'subspaceoverlap'
            subspaceoverlap = [training.subspaceoverlap];
            % w_in__w_tc
            for i=1:N_nets, w_in__w_tc(i,:) = abs(subspaceoverlap(i).w_in__w_tc); end
            w_in__w_tc_mu = mean(w_in__w_tc); w_in__w_tc_sem = std(w_in__w_tc)/sqrt(N_nets);
            % w_in__w_ct
            for i=1:N_nets, w_in__w_ct(i,:) = abs(subspaceoverlap(i).w_in__w_ct); end
            w_in__w_ct_mu = mean(w_in__w_ct); w_in__w_ct_sem = std(w_in__w_ct)/sqrt(N_nets);
            % w_out__w_tc
            for i=1:N_nets, w_out__w_tc(i,:) = abs(subspaceoverlap(i).w_out__w_tc); end
            w_out__w_tc_mu = mean(w_out__w_tc); w_out__w_tc_sem = std(w_out__w_tc)/sqrt(N_nets);
            % w_out__w_ct
            for i=1:N_nets, w_out__w_ct(i,:) = abs(subspaceoverlap(i).w_out__w_ct); end
            w_out__w_ct_mu = mean(w_out__w_ct); w_out__w_ct_sem = std(w_out__w_ct)/sqrt(N_nets);
            % w_tc__w_ct
            for i=1:N_nets, w_tc__w_ct(i,:) = abs(subspaceoverlap(i).w_tc__w_ct); end
            w_tc__w_ct_mu = mean(w_tc__w_ct); w_tc__w_ct_sem = std(w_tc__w_ct)/sqrt(N_nets);
            
            % plot subspace overlap
            ntrls = learning_params(1).ntrls;
            epochlength = numel(w_in__w_tc_mu);
            errorbar(epochlength:epochlength:ntrls, w_in__w_tc_mu, w_in__w_tc_sem);
            errorbar(epochlength:epochlength:ntrls, w_in__w_ct_mu, w_in__w_ct_sem);
            errorbar(epochlength:epochlength:ntrls, w_out__w_tc_mu, w_out__w_tc_sem);
            errorbar(epochlength:epochlength:ntrls, w_out__w_ct_mu, w_out__w_ct_sem);
            errorbar(epochlength:epochlength:ntrls, w_tc__w_ct_mu, w_tc__w_ct_sem);
    end
end