% Comparison between isotropic and anisotropic PSO
%
% This script compares the decay behavior of our functional V for different
% parameters of the dynamics with the theoretically expected rates.
%

%%
clear; clc; close all;

co = set_color();
co = co([1,2,3,5,6],:);


%% Settings for Easy Handling and Notes
% 
% use pre-set PSO setting (overrides manually chosen parameters)
pre_setparameters = 0;

% save plot
pdfexport = 0;

% 
H_type = 'star'; % variance or star

% plot settings
semilogy_plot = 0; % show decays in semilogy plot
normalized = 1; % normalize energy functional V

% % parameter for comparison (with values)
% (this overwrites the one from below)
parameter_of_interest = 'm';
parameter_values_of_interest = [0.4,0.2,0.1,0.01,0.001];
% 'm';            [1,0.5,0.1,0.01];
% 'kappa';        [];
% 'gamma';        [0.1,0.15,0.2,0.4,0.8]; in combination with m = 0.1 or 0.01
% 'lambda1';      [];
% 'lambda2';      [];
% 'sigma1';       [];
% 'sigma2';       [0.02,0.08,0.2,0.4];
% 'alpha';        [1,2,4,10,100];
% 'beta';         [1,2,4,10,100];


%% Energy Function E

% % dimension of the ambient space
d = 20;

% % energy function E
% (E is a function mapping columnwise from R^{d\times N} to R)
objectivefunction = 'Rastrigin';
[E, parametersE, ~, ~] = objective_function(objectivefunction, d, 'PSO');

% range of x (and x and y for plotting)
xrange_plot = parametersE(:,1)';
yrange_plot = parametersE(:,2)';
xrange = 100*xrange_plot;


% global minimizer
xstar = zeros(d,1); %fminbnd(E,xrange_plot(1),xrange_plot(2));


%% Parameters of PSO Algorithm

% time horizon
T = 10;

% discrete time size
dt = 0.01;

% number of particles
N = 1000; %320000;

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
sigma2 = 0.2;

% alpha (weight in Gibbs measure for global and in-time best position computation)
alpha = 1000;
% beta (regularization parameter for sigmoid)
beta = -1; % 'inf' or -1 for using Heaviside function instead of S_beta

 
%% Initialization
X0mean = 2*ones(d,1);
X0std = 4;
V0mean = zeros(d,1);
V0std = 0;
X0 = X0mean+X0std*randn(d,N);
Y0 = X0;
V0 = V0mean+V0std*randn(d,N);

parametersInitialization = containers.Map({'X0mean', 'X0std', 'V0mean', 'V0std'},...
                                          {  X0mean,   X0std,   V0mean,   V0std});


%% Error Metrics

% % Functional 
Hfunctional = NaN(length(parameter_values_of_interest),1+T/dt);
Hvarfunctional = NaN(length(parameter_values_of_interest),1+T/dt);

for i = 1:length(parameter_values_of_interest)
    
    parametersPSO = containers.Map({'T', 'dt', 'N', 'm', 'kappa', 'gamma', 'memory', 'lambda1', 'lambda2', 'anisotropic1', 'sigma1', 'anisotropic2', 'sigma2', 'alpha', 'beta'},...
                                   {  T,   dt,   N,   m,   kappa,   gamma,   memory,   lambda1,   lambda2,   anisotropic1,   sigma1,   anisotropic2,   sigma2,   alpha,   beta});
    
    % setting parameter of interest
    parametersPSO(parameter_of_interest) = parameter_values_of_interest(i);
    
    X = X0;
    Y = Y0;
    V = V0;
    
    % normalization of error metrics
    m = parametersPSO('m');
    gamma = 1-m;
    if normalized
        normal_Hfunctional = (gamma/m)^2*sum(vecnorm(X-xstar,2,1).^2)/N + sum(vecnorm(V,2,1).^2)/N;% + (gamma/m)*sum(sum((X-xstar).*V,1),2)/N;
        Expectation0 = sum(X0,2)/N;
        normal_Hvarfunctional = (gamma/m)^2*sum(vecnorm(X-Expectation0,2,1).^2)/N + sum(vecnorm(V,2,1).^2)/N;% + (gamma/m)*sum(sum((X-Expectation0).*V,1),2)/N;
    end
    
    % % Initialization of error metrics
    Hfunctional(i,1) = normal_Hfunctional;
    Hvarfunctional(i,1) = normal_Hvarfunctional;
    if normalized
        Hfunctional(i,1) = Hfunctional(i,1)/normal_Hfunctional;
        Hvarfunctional(i,1) = Hvarfunctional(i,1)/normal_Hvarfunctional;
    end
    

    % PSO Algorithm 
    for k = 1:T/dt
        
        % % PSO iteration
        % compute global and in-time best position y_alpha
        y_alpha = compute_yalpha(E, alpha, Y);

        % position updates of one iteration of PSO
        [X, Y, V] = PSO_update(E, parametersPSO, y_alpha, X, Y, V);
        
        % % Computation of Error Metrics
        % Energy Functional Hstar
        Hfunctional(i,k+1) = (gamma/m)^2*sum(vecnorm(X-xstar,2,1).^2)/N + sum(vecnorm(V,2,1).^2)/N;% + (gamma/m)*sum(sum((X-xstar).*V,1),2)/N;
        % Energy Functional Hvar
        Expectation = sum(X,2)/N;
        Hvarfunctional(i,k+1) = (gamma/m)^2*sum(vecnorm(X-Expectation,2,1).^2)/N + sum(vecnorm(V,2,1).^2)/N;% + (gamma/m)*sum(sum((X-Expectation).*V,1),2)/N;
        
        if normalized
            Hfunctional(i,k+1) = Hfunctional(i,k+1)/normal_Hfunctional;
            Hvarfunctional(i,k+1) = Hvarfunctional(i,k+1)/normal_Hvarfunctional;
        end

    end

end

%% Plotting of Error Metrics

set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');

if strcmp(parameter_of_interest,'m')
    parameter_of_interest_string = '$m=\,$';
elseif strcmp(parameter_of_interest,'kappa')
    parameter_of_interest_string = '$\kappa=\,$';
elseif strcmp(parameter_of_interest,'gamma')
    parameter_of_interest_string = '$\gamma=\,$';
elseif strcmp(parameter_of_interest,'lambda1')
    parameter_of_interest_string = '$\lambda_1=\,$';
elseif strcmp(parameter_of_interest,'lambda2')
    parameter_of_interest_string = '$\lambda_2=\,$';
elseif strcmp(parameter_of_interest,'sigma1')
    parameter_of_interest_string = '$\sigma_1=\,$';
elseif strcmp(parameter_of_interest,'sigma2')
    parameter_of_interest_string = '$\sigma_2=\,$';
elseif strcmp(parameter_of_interest,'alpha')
    parameter_of_interest_string = '$\alpha=\,$';
elseif strcmp(parameter_of_interest,'beta')
    parameter_of_interest_string = '$\beta=\,$';
else
    error('parameter_of_interest not known')
end
    
f = figure('Position', [1700 800 600 400]);


for i = 1:length(parameter_values_of_interest)
    if ~normalized
        label_Hfunctional = ['$\mathcal{H}(\widehat\rho^N_t)$, ', parameter_of_interest_string,num2str(parameter_values_of_interest(i))];
    else
        label_Hfunctional = ['$\mathcal{H}(\widehat\rho^N_t)/\mathcal{H}(\rho_0)$, ', parameter_of_interest_string ,num2str(parameter_values_of_interest(i))];
    end
    
    if strcmp(H_type, 'variance')
        if ~semilogy_plot
            errormetric_plot = plot(0:dt:T,Hvarfunctional(i,:), "color", co(i,:), 'LineWidth', 2, 'LineStyle', '-','DisplayName',label_Hfunctional);
        else
            errormetric_plot = semilogy(0:dt:T,Hvarfunctional(i,:), "color", co(i,:), 'LineWidth', 2, 'LineStyle', '-','DisplayName',label_Hfunctional);
        end
    elseif strcmp(H_type, 'star')
        if ~semilogy_plot
            errormetric_plot = plot(0:dt:T,Hfunctional(i,:), "color", co(i,:), 'LineWidth', 2, 'LineStyle', '-','DisplayName',label_Hfunctional);
        else
            errormetric_plot = semilogy(0:dt:T,Hfunctional(i,:), "color", co(i,:), 'LineWidth', 2, 'LineStyle', '-','DisplayName',label_Hfunctional);
        end
    else
        error('H_type error. H_type not know.')
    end
    hold on
end




%%%%%
hold on
rate_plot = plot(0:dt:T,exp(-(2*lambda2-d*0.8^2)*[0:dt:T]), "color", 0.4*[1,1,1], 'LineWidth', 2, 'LineStyle', ':','DisplayName','CBO');
hold on
%%%%



xlim([0,T])
xticks([0:0.5:T])
if ~semilogy_plot
    % normal plot
    ylim([0,1])
    yticks([0 0.25 0.5 0.75 1])
else
    % semilogy plot
    %ylim([5*10^-3,1])
end

% % rate of decay reference line (from theory)
% for i = 1:length(parameter_values_of_interest)
%     label_rate = ['$\exp\!\big(\!-(2\lambda-d\sigma^2)t\big),\,d=\,$',regexprep(num2str(d),'\s+',', ')];
%     if ~semilogy_plot
%         rate_plot = plot(0:dt:T,exp(-(2*lambda-d(i)*sigma^2)*[0:dt:T]), "color", 0.4*[1,1,1], 'LineWidth', 2, 'LineStyle', ':','DisplayName',label_rate);
%     else
%         rate_plot = semilogy(0:dt:T,exp(-(2*lambda-d(i)*sigma^2)*[0:dt:T]), "color", 0.4*[1,1,1], 'LineWidth', 2, 'LineStyle', ':','DisplayName',label_rate);
%     end
%     if i>1
%         rate_plot.Annotation.LegendInformation.IconDisplayStyle = 'off';
%     end
% end
    
ax = gca;
ax.FontSize = 13;

xlabel('$t$','Interpreter','latex','FontSize',15)
if ~semilogy_plot
    legend('Interpreter','latex','FontSize',15,'Location','northeast')
else
    legend('Interpreter','latex','FontSize',15,'Location','southwest')
end


%% Save Image
if pdfexport
    print(f,['images_videos/VforAn_isotropicforVariousdim_',objectivefunction],'-dpdf');

    % save parameters
    save(['images_videos/VforAn_isotropicforVariousdim_',objectivefunction,'_param'], 'objectivefunction', 'E', 'xstar', 'd', 'T', 'dt', 'N', 'alpha', 'lambda', 'gamma', 'learning_rate', 'sigma', 'V0mean_radial', 'V0mean_type', 'V0std')
end

