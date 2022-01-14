% PSO phase transition diagrams for different parameters
%
% This script produces phase diagrams with respect to two selected
% parameters of PSO.
%

%%
clear; clc; close all;

co = set_color();

numberofworkers = 4; %25;


%% Energy Function E

% % dimension of the ambient space
d = 20;

% % energy function E
% (E is a function mapping columnwise from R^{d\times N} to R)
objectivefunction = 'Rastrigin';
[E, parametersE, ~, ~] = objective_function(objectivefunction, d, 'PSO');

% global minimizer
vstar = zeros(d,1); %fminbnd(E,xrange_plot(1),xrange_plot(2));


%% Settings for Easy Handling and Notes

% % parameter for comparison (with values)
% (this overwrites the one from below)
pt_diagram_x = 'dt';
pt_diagram_y = 'kappa';
pt_diagram_x_values = 10.^[-4:1/2:-2];
pt_diagram_y_values = 10.^[0:0.5:6];


%%% interesting phase diagrams
% % m and sigma2
% pt_diagram_x = 'm';
% pt_diagram_y = 'sigma2';
% pt_diagram_x_values = 10.^[-4:1/6:-0.5];
% pt_diagram_y_values = 1:0.5:12;
% % ... and ...
% pt_diagram_x = '';
% pt_diagram_y = '';
% pt_diagram_x_values = ;
% pt_diagram_y_values = ;


number_runs = 8;


%% Parameters of PSO Algorithm

% time horizon
T = 20;

% discrete time size
dt = 0.01;

% number of particles
N = 100;

% inertia weight
m = 0.01;

%
kappa = 1/dt;

% friction coefficient
gamma = 1-m;

% memory
memory = 1; % 0 or 1
% lambda1, sigma1, kappa and beta have no effect for memory=0.


% lambda1 (drift towards in-time best parameter)
lambda1 = 0;
% lambda2 (drift towards global and in-time best parameter)
lambda2 = 1;

% type of diffusion for noise term 1
anisotropic1 = 1;
% sigma (parameter of noise term 1)
sigma1 = lambda1*8;
% type of diffusion for noise term 2
anisotropic2 = 1;
% sigma (parameter of noise term 2)
sigma2 = 8;

% alpha (weight in Gibbs measure for global and in-time best position computation)
alpha = 100;
% beta (regularization parameter for sigmoid)
beta = 10^15; % 'inf' or -1 for using Heaviside function instead of S_beta


%% Initialization
X0mean = 2*ones(d,1);
X0std = 4;
V0mean = zeros(d,1);
V0std = 0;

parametersPSO = containers.Map({'T', 'dt', 'N', 'm', 'kappa', 'gamma', 'memory', 'lambda1', 'lambda2', 'anisotropic1', 'sigma1', 'anisotropic2', 'sigma2', 'alpha', 'beta'},...
                               {  T,   dt,   N,   m,   kappa,   gamma,   memory,   lambda1,   lambda2,   anisotropic1,   sigma1,   anisotropic2,   sigma2,   alpha,   beta});
parametersInitialization = containers.Map({'X0mean', 'X0std', 'V0mean', 'V0std'},...
                                          {  X0mean,   X0std,   V0mean,   V0std});


%% Phase Transition Diagram table
pt_diagram = zeros(number_runs, length(pt_diagram_y_values), length(pt_diagram_x_values));


%% PSO Algorithm

for i = 1:length(pt_diagram_x_values)
    
    % setting parameter of interest
    parametersPSO(pt_diagram_x) = pt_diagram_x_values(i);
    if strcmp(pt_diagram_y, 'lambda1')
        parametersPSO('sigma1') = parametersPSO('lambda1')*parametersPSO('sigma2');
    end
    
    for j = 1:length(pt_diagram_y_values)
        
        disp(['iteration (', num2str(i), '/', num2str(length(pt_diagram_x_values)), ', ' num2str(j), '/', num2str(length(pt_diagram_y_values)), ')'])
        
        % setting parameter of interest
        parametersPSO(pt_diagram_y) = pt_diagram_y_values(j);
        if strcmp(pt_diagram_y, 'sigma2')
            parametersPSO('sigma1') = parametersPSO('lambda1')*parametersPSO('sigma2');
        end
        
        parfor (r = 1:number_runs, numberofworkers)
            
            %initialization
            X0 = X0mean+X0std*randn(d,parametersPSO('N'));
            V0 = V0mean+V0std*randn(d,parametersPSO('N'));
            X = X0;
            V = V0;
            
            % PSO
            [xstar_app] = PSO(E, parametersPSO, X0, V0);

            % count successful runs
            if E(xstar_app)<0.8
                pt_diagram(r, j, i) = pt_diagram(r, j, i)+1;
            end

        end
        
    end
    
end

pt_diagram = reshape(mean(pt_diagram,1), [length(pt_diagram_y_values), length(pt_diagram_x_values)]);


%%

filename = ['CBOandPSO/EnergyBasedPSOAnalysis/results/', 'memory', num2str(memory), 'phasediagram_', pt_diagram_x, pt_diagram_y, '_lambda1_', num2str(lambda1*10), 'div10_N_', num2str(N)];

save(filename, 'pt_diagram', 'd', 'objectivefunction', 'pt_diagram_x', 'pt_diagram_y', 'pt_diagram_x_values', 'pt_diagram_y_values', 'number_runs', 'T', 'dt', 'N', 'm', 'kappa', 'gamma', 'memory', 'lambda1', 'lambda2', 'anisotropic1', 'sigma1', 'anisotropic2', 'sigma2', 'alpha', 'beta', 'X0mean', 'X0std', 'V0mean', 'V0std')


%% Plotting

% imagesc(flipud(pt_diagram))
% colorbar
% xlabel(['$', pt_diagram_x, '$'],'interpreter','latex')
% xticks(1:length(pt_diagram_x_values))
% xticklabels(pt_diagram_x_values)
% ylabel(['$', pt_diagram_y, '$'],'interpreter','latex')
% yticks(1:length(pt_diagram_y_values))
% yticklabels(flipud(pt_diagram_y_values')')
