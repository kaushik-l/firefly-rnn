classdef rnn < handle
%{
    Thalamo-cortical recurrent neural network

    Parameters:
    ----------
    network_parameters {n_in, n_c, n_out : number of input, cortical, output units}
                       {w_in, w_cc, w_out : weights}
                       {g_cc: scaling of weight variance}
                       {tau_c : time constant of cortical, thalamic neurons, in units of timesteps}
    learning_parameters {train_cc, train_in, train_out: whether to train 
                        corticocortical, input, output weights}
                        {fb_type: 
                        type of feedback connection ('random', 'aligned')
                        {ntrls: number of trials to train}
                        }
    initial_cond {h0: initial state vector of the cortex}
    task_params: {x_in, y_out: input, output}
%}
    %%
    properties
        network_params
        learning_params
        initial_cond
        task_params
        training
    end
    %%
    methods
        %% class constructor
        function this = rnn(network_params, learning_params, initial_cond, task_params)      
            %% learning parameters
            this.learning_params.train_cc = learning_params.train_cc;
            this.learning_params.train_out = learning_params.train_out;
            this.learning_params.train_in = learning_params.train_in;
            this.learning_params.fb_type = learning_params.fb_type;
            this.learning_params.eta_out = learning_params.eta_out;
            this.learning_params.eta_cc = learning_params.eta_cc;
            this.learning_params.eta_in = learning_params.eta_in;
            this.learning_params.ntrls = learning_params.ntrls;
            this.learning_params.algorithm = learning_params.algorithm;
            this.learning_params.online_learning = learning_params.online_learning;
            
            %% network parameters
            this.network_params.n_c = network_params.n_c;
            this.network_params.n_in = network_params.n_in;
            this.network_params.n_out = network_params.n_out;
            this.network_params.tau_c = network_params.tau_c;
            this.network_params.g_cc = network_params.g_cc;
            this.network_params.w_in = 1*(2*rand(this.network_params.n_c,this.network_params.n_in) - 1)/sqrt(this.network_params.n_in);
            this.network_params.w_out = 2*(2*rand(this.network_params.n_out, this.network_params.n_c) - 1)/sqrt(this.network_params.n_c);
            this.network_params.w_cc = this.network_params.g_cc*randn(this.network_params.n_c,this.network_params.n_c)/sqrt(this.network_params.n_c);            
            if strcmp(this.learning_params.fb_type,'aligned')
                this.network_params.b = (this.network_params.w_out')*(sqrt(this.network_params.n_c)/sqrt(this.network_params.n_out));
            else
                this.network_params.b = randn(this.network_params.n_c, this.network_params.n_out)/sqrt(this.network_params.n_out);
            end
            
            %% intial condition
            this.initial_cond = initial_cond;
            
            %% task parameters
            this.task_params = task_params;
        end
        
        %% plot
        function plot(this,plottype)
            switch lower(plottype)
                case 'learningcurve'
                    p = plot(1:this.learning_params.ntrls,this.training.loss);
                    idx = mod(1:this.learning_params.ntrls,1e3)==0;
                    scatter(find(idx),this.training.loss(idx),'o','filled','MarkerFaceColor',p.Color);
                    set(gca,'TickDir','Out','Ticklength',[.03 .03],'Fontsize',14); box off;
                    xlabel('Trial','Fontsize',18); ylabel('MSE','Fontsize',18);
                case 'fbalign'
                    p = plot(1:this.learning_params.ntrls,this.training.overlap.b__w_out);
                    idx = mod(1:this.learning_params.ntrls,1e3)==0;
                    scatter(find(idx),this.training.overlap.b__w_out(idx),'o','filled','MarkerFaceColor',p.Color);
                    set(gca,'TickDir','Out','Ticklength',[.03 .03],'Fontsize',14); box off;
                    xlabel('Trial','Fontsize',18); ylabel('Feedback alignment','Fontsize',18);
                case 'input'
                    plot(this.task_params.x_in,'--k');
                    set(gca,'TickDir','Out','Ticklength',[.03 .03],'Fontsize',14); box off;
                    xlabel('Time','Fontsize',18); ylabel('Input','Fontsize',18);
                case 'output'
                    plot(this.task_params.y_out,'--k');
                    plot(this.training.activity.y_.post);
                    set(gca,'TickDir','Out','Ticklength',[.03 .03],'Fontsize',14); box off;
                    xlabel('Time','Fontsize',18); ylabel('Output','Fontsize',18);
                case 'pccortex'
                    npcs = size(this.training.pcs.score{end},2);
                    linecolors = copper(npcs);
                    for i=1:npcs
                        plot(this.training.pcs.score{end}(:,i),'Color',linecolors(i,:));
                    end
                    set(gca,'TickDir','Out','Ticklength',[.03 .03],'Fontsize',14); box off;
                    xlabel('Time','Fontsize',18); ylabel('PC score','Fontsize',18);
                case 'activitycortex'
                    h = this.training.activity.h_.post;
                    [~,idx] = max(h);
                    [~,idx] = sort(idx);
                    imagesc(h(:,idx)');
                    set(gca,'TickDir','Out','Ticklength',[.03 .03],'Fontsize',14); box off;
                    xlabel('Time','Fontsize',18); ylabel('Neuron','Fontsize',18);
                    axis tight;
                case 'activitythalamus'
                    r = this.training.activity.r_.post;
                    [~,idx] = max(r);
                    [~,idx] = sort(idx);
                    imagesc(r(:,idx)');
                    set(gca,'TickDir','Out','Ticklength',[.03 .03],'Fontsize',14); box off;
                    xlabel('Time','Fontsize',18); ylabel('Neuron','Fontsize',18);
                    axis tight;
                case 'pcoverlap'
                    n_in = this.network_params.n_in;
                    n_c = this.network_params.n_c;
                    n_t = this.network_params.n_t;
                    n_out = this.network_params.n_out;
                    overlap = this.training.overlap;
                    subplot(1,4,1); hold on;
                    plot(abs(overlap.pc__w_in(end,:)));
                    set(gca,'TickDir','Out','Ticklength',[.03 .03],'Fontsize',14); box off;
                    xlabel('Component','Fontsize',18); ylabel('w_{in}','Fontsize',18);
                    subplot(1,4,2); hold on;
                    for i=1:n_t, plot(squeeze(abs(overlap.pc__w_tc(end,:,i)))); end
                    set(gca,'TickDir','Out','Ticklength',[.03 .03],'Fontsize',14); box off;
                    xlabel('Component','Fontsize',18); ylabel('w_{tc}','Fontsize',18);
                    subplot(1,4,3); hold on;
                    for i=1:n_t, plot(squeeze(abs(overlap.pc__w_ct(end,i,:)))); end
                    set(gca,'TickDir','Out','Ticklength',[.03 .03],'Fontsize',14); box off;
                    xlabel('Component','Fontsize',18); ylabel('w_{ct}','Fontsize',18);
                    subplot(1,4,4); hold on;
                    plot(abs(overlap.pc__w_out(end,:)));
                    set(gca,'TickDir','Out','Ticklength',[.03 .03],'Fontsize',14); box off;
                    xlabel('Component','Fontsize',18); ylabel('w_{out}','Fontsize',18);
            end
        end
    end
end