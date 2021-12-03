%% Settings for Easy Handling and Notes

% % parameter for comparison
pt_diagram_x = 'm';
pt_diagram_y = 'sigma2';

% number of particles
N = 100;

% memory
memory = 1; % 0 or 1
% lambda1, sigma1, kappa and beta have no effect for memory=0.


% lambda1 (drift towards in-time best parameter)
lambda1 = 0.4;


%%
if strcmp([pt_diagram_x,pt_diagram_y], 'msigma2')
    filename = ['CBOandPSO/EnergyBasedPSOAnalysis/results/PhaseTransitionDiagrams_msigma2/', 'memory', num2str(memory), 'phasediagram_', pt_diagram_x, pt_diagram_y, '_lambda1_', num2str(lambda1*10), 'div10_N_', num2str(N)];
else
    filename = ['CBOandPSO/EnergyBasedPSOAnalysis/results/', 'memory', num2str(memory), 'phasediagram_', pt_diagram_x, pt_diagram_y, '_lambda1_', num2str(lambda1*10), 'div10_N_', num2str(N)];
    error('TBD')
end
load(filename)


imagesc(flipud(pt_diagram))

c = colorbar;
set(c,'TickLabelInterpreter','latex')
caxis([0 1])

set(groot,'defaultAxesTickLabelInterpreter','latex');  
if strcmp(pt_diagram_x, 'm')
    pt_diagram_x_label = 'm';
    x_ticks = 1:3:length(pt_diagram_x_values);
    x_ticks_labels = log10(pt_diagram_x_values);
    x_ticks_labels = x_ticks_labels(1:3:end);
    x_ticks_labels_cell = cell(1, length(x_ticks_labels));
    for s=1:length(x_ticks_labels)
        x_ticks_labels_cell{s} = ['$10^{', num2str(x_ticks_labels(s)),'}$'];
    end
elseif strcmp(pt_diagram_x, '')
    pt_diagram_x_label = '';
end
xlabel(['$', pt_diagram_x_label, '$'],'Interpreter','latex','FontSize',15)
xticks(x_ticks)
xticklabels(x_ticks_labels_cell)


if strcmp(pt_diagram_y, 'sigma2')
    if memory==1
        pt_diagram_y_label = '\sigma_2';
    elseif memory==0
        pt_diagram_y_label = '\sigma';
    end
    y_ticks = 1:2:length(pt_diagram_y_values);
    y_ticks_labels = flipud(pt_diagram_y_values')';
    y_ticks_labels = y_ticks_labels(1:2:end);
elseif strcmp(pt_diagram_y, '')
    pt_diagram_y_label = '';
end
ylabel(['$', pt_diagram_y_label, '$'],'Interpreter','latex','FontSize',15)
yticks(y_ticks)
yticklabels(y_ticks_labels)