% This function implements the plotting routine for performance (testing 
% accuracy and training loss) plots and can be used to compare different
% parameter settings
% 
% Note: plotting parameters (#architectures, #parameters, #epochs,
% batches_per_epoch) have to be specified in the function as well as the
% data to be visualized
% 
% 
% plot_loss_and_testing_accuracy_PSON1000()
%           
% output:   plot
%

function plot_loss_and_testing_accuracy_PSON1000()

number_of_architectures = 2;
number_of_parametersettings = 3; % needs to be consistent with the number of loaded settings

parameter_type = 'different_m'; % different_m or different_memory

epochs_plot = 10;
batches_per_epoch = 2; % needs to be a common divisor (1,2,5,10) of the individual batches_per_epoch if any

performance_tracking_all = nan(number_of_architectures,number_of_parametersettings,3,epochs_plot+1,batches_per_epoch);

if strcmp(parameter_type, 'different_m')
    
    % load shallow NN data
    load('CBOandPSO/NN/results/PSO/ShallowNN/PSOMNIST_N_1000_memory_1_lambda1_0div10_m_40div100_20epochs.mat')
    performance_tracking = performance_tracking(:,:,linspace(size(performance_tracking,3)/batches_per_epoch,size(performance_tracking,3),batches_per_epoch));
    performance_tracking_all(2,3,:,1:(epochs+1),:) = performance_tracking;
    load('CBOandPSO/NN/results/PSO/ShallowNN/PSOMNIST_N_1000_memory_1_lambda1_0div10_m_20div100_20epochs.mat')
    performance_tracking = performance_tracking(:,:,linspace(size(performance_tracking,3)/batches_per_epoch,size(performance_tracking,3),batches_per_epoch));
    performance_tracking_all(2,2,:,1:(epochs+1),:) = performance_tracking;
    load('CBOandPSO/NN/results/PSO/ShallowNN/PSOMNIST_N_1000_memory_1_lambda1_0div10_m_10div100_20epochs.mat')
    performance_tracking = performance_tracking(:,:,linspace(size(performance_tracking,3)/batches_per_epoch,size(performance_tracking,3),batches_per_epoch));
    performance_tracking_all(2,1,:,1:(epochs+1),:) = performance_tracking;
    
    % load CNN data
    load('CBOandPSO/NN/results/PSO/CNN/PSOMNIST_N_1000_memory_1_lambda1_0div10_m_4div10_5epochs.mat')
    performance_tracking = performance_tracking(:,:,linspace(size(performance_tracking,3)/batches_per_epoch,size(performance_tracking,3),batches_per_epoch));
    performance_tracking_all(1,3,:,1:(epochs+1),:) = performance_tracking;
    load('CBOandPSO/NN/results/PSO/CNN/PSOMNIST_N_1000_memory_1_lambda1_0div10_m_2div10_5epochs.mat')
    performance_tracking = performance_tracking(:,:,linspace(size(performance_tracking,3)/batches_per_epoch,size(performance_tracking,3),batches_per_epoch));
    performance_tracking_all(1,2,:,1:(epochs+1),:) = performance_tracking;
    load('CBOandPSO/NN/results/PSO/CNN/PSOMNIST_N_1000_memory_1_lambda1_0div10_m_1div10_5epochs.mat')
    performance_tracking = performance_tracking(:,:,linspace(size(performance_tracking,3)/batches_per_epoch,size(performance_tracking,3),batches_per_epoch));
    performance_tracking_all(1,1,:,1:(epochs+1),:) = performance_tracking;
    
elseif strcmp(parameter_type, 'different_memory')
    
    % load shallow NN data
    load('CBOandPSO/NN/results/PSO/ShallowNN/PSOMNIST_N_1000_memory_1_lambda1_4div10_m_20div100_20epochs.mat')
    performance_tracking = performance_tracking(:,:,linspace(size(performance_tracking,3)/batches_per_epoch,size(performance_tracking,3),batches_per_epoch));
    performance_tracking_all(2,3,:,1:(epochs+1),:) = performance_tracking;
    load('CBOandPSO/NN/results/PSO/ShallowNN/PSOMNIST_N_1000_memory_1_lambda1_0div10_m_20div100_20epochs.mat')
    performance_tracking = performance_tracking(:,:,linspace(size(performance_tracking,3)/batches_per_epoch,size(performance_tracking,3),batches_per_epoch));
    performance_tracking_all(2,2,:,1:(epochs+1),:) = performance_tracking;
    load('CBOandPSO/NN/results/PSO/ShallowNN/PSOMNIST_N_1000_memory_0_lambda1_0div10_m_20div100_20epochs.mat')
    performance_tracking = performance_tracking(:,:,linspace(size(performance_tracking,3)/batches_per_epoch,size(performance_tracking,3),batches_per_epoch));
    performance_tracking_all(2,1,:,1:(epochs+1),:) = performance_tracking;
    
    % load CNN data
    load('CBOandPSO/NN/results/PSO/CNN/PSOMNIST_N_1000_memory_1_lambda1_4div10_m_2div10_5epochs.mat')
    performance_tracking = performance_tracking(:,:,linspace(size(performance_tracking,3)/batches_per_epoch,size(performance_tracking,3),batches_per_epoch));
    performance_tracking_all(1,3,:,1:(epochs+1),:) = performance_tracking;
    load('CBOandPSO/NN/results/PSO/CNN/PSOMNIST_N_1000_memory_1_lambda1_0div10_m_2div10_5epochs.mat')
    performance_tracking = performance_tracking(:,:,linspace(size(performance_tracking,3)/batches_per_epoch,size(performance_tracking,3),batches_per_epoch));
    performance_tracking_all(1,2,:,1:(epochs+1),:) = performance_tracking;
    %load('CBOandPSO/NN/results/PSO/CNN/PSOMNIST_N_1000_memory_0_lambda1_0div10_m_2div10_5epochs.mat')
    %performance_tracking = performance_tracking(:,:,linspace(size(performance_tracking,3)/batches_per_epoch,size(performance_tracking,3),batches_per_epoch));
    performance_tracking_all(1,1,:,1:(epochs+1),:) = zeros(3,epochs+1,batches_per_epoch);
    
end

%performance_tracking_all = performance_tracking_all(1:number_of_architectures,1:number_of_parametersettings,1:3,1:(epochs_plot+1),1:10);
performance_tracking_all(performance_tracking_all==0) = nan;


%% plotting
set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');

figure('Position', [1200 800 600 350]);
co = set_color();

loss_color = co(1,:);
accuracy_color = co(2,:);

for k = 1:number_of_architectures
    
    for l = 1:number_of_parametersettings
    
        performance_tracking = reshape(performance_tracking_all(k,l,:,:,:), [3,size(performance_tracking_all(k,l,:,:,:),4),batches_per_epoch]);

        [~, epochs_, batch_size_E_batches] = size(performance_tracking); epochs_ = epochs_-1;

        performance_tracking_k = [performance_tracking(:,1,end), reshape(permute(performance_tracking(:,2:end,:),[1 3 2]), [3, epochs_*batch_size_E_batches])];

        epoch_batch_discretization = 0:1/batch_size_E_batches:epochs_;

        if k==1
            line_style = '-';
        elseif k==2
            line_style = '--';
        elseif k==3
            line_style = '-.';
        else
            line_style = ':';
        end
        
        if l==1
            line_opacity = 0.4;
        elseif l==2
            line_opacity = 0.7;
        elseif l==3
            line_opacity = 1;
        else
            error('Change line_opacities.')
        end

        yyaxis right
        % plot testing accuracy 
        kk((k-1)*3+1) = plot(epoch_batch_discretization, performance_tracking_k(2,:), 'LineWidth', 2, 'LineStyle', line_style, 'Color', [accuracy_color, line_opacity], 'Marker', 'none'); hold on
        %plot(epoch_discretization, performance_tracking_epoch(1,:), 'LineStyle', '-', 'LineWidth', 2, 'Color', co(1,:)); hold on

        xlabel('number of epochs','Interpreter','latex','FontSize',15)
        ylabel('testing accuracy','Interpreter','latex','FontSize',15)
        ylim([0.6 1])
        yticks(0:0.04:1)
        if epochs_plot<=20
            xlim([0 epochs_plot])
            xticks(0:1:epochs_plot)
        elseif epochs_plot<=50
            xlim([0 epochs_plot])
            xticks(0:5:epochs_plot)
        end
        ax = gca;
        ax.FontSize = 13;
        ax.YColor = accuracy_color;
        box on

        yyaxis left
        % plot training loss
        kk((k-1)*3+2) = plot(epoch_batch_discretization, performance_tracking_k(3,:), 'LineWidth', 2, 'LineStyle', line_style, 'Color', [loss_color, line_opacity], 'Marker', 'none'); hold on
        %plot(epoch_discretization, performance_tracking_epoch(2,:), 'LineStyle', '-', 'LineWidth', 2, 'Color', co(2,:)); hold on

        ylabel('crossentropy risk~$\mathcal{E}$','Interpreter','latex','FontSize',15)
        ylim([0 0.2])
        yticks(0:0.02:1)
        ax = gca;
        ax.YColor = loss_color;


        % plot for legend
        %(k-1)*number_of_architectures+3
        kk((k-1)*3+3) = plot(epoch_batch_discretization, 100*ones(size(epoch_batch_discretization)), 'LineWidth', 2, 'LineStyle', line_style, 'Color', 'black', 'Marker', 'none'); hold on
    
    end
end
grid on
ax = gca;
ax.GridColor = [0.2, 0.2, 0.2];

legend(kk([3 6]), 'Convolutional neural network','Shallow neural network','Location','southeast','Interpreter','latex','FontSize',15)

end


