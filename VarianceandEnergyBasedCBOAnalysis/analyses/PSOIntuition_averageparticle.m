% PSO intuition
%
% This script illustrates the intuition behind our novel analysis approach
% for PSO.
% Such plot is used in Figure ... in "On the Global Convergence of 
% Particle-Swarm Optimization Methods"
%

%%
clear; clc; close all;

co = set_color();

%% Settings for Easy Handling and Notes
% 
% decice if time steps require pressing some arbitrary key
manual_steps = 0;
% plotting empirical expectation of the particles
show_expectation = 0;
% plot position X or local best position Y
plot_XorY = 'Y';
% use pre-set CBO setting (overrides manually chosen parameters)
pre_setparameters = 0;

% 3d plot
spatial_plot = 0;

% save plot
pdfexport = 0;


%% Energy Function E

% % dimension of the ambient space
d = 2;

% % energy function E
% (E is a function mapping columnwise from R^{d\times N} to R)
% lopsided W-shaped function in 1d
objectivefunction = 'Rastrigin';
[E, parametersE, parametersPSO, parametersInitialization] = objective_function(objectivefunction, d, 'PSO');

% range of x (and x and y for plotting)
xrange_plot = parametersE(:,1)';
yrange_plot = parametersE(:,2)';
zrange_plot = parametersE(:,3)';
xrange = 100*xrange_plot;
yrange = 100*yrange_plot;

% global minimizer
xstar = zeros(d,1); %fminbnd(E,xrange_plot(1),xrange_plot(2));


%% Parameters of PSO Algorithm

% time horizon
T = 20;

% discrete time size
dt = 0.1;

% number of particles
N = 3200; % N = 320000; N = 32000; N = 3200;

% inertia weight
m = 0.6;

%
kappa = 1;

% friction coefficient
gamma = 1-m;

% memory
memory = 0; % 0 or 1
% lambda1, sigma1, kappa and beta have no effect for memory=0.

% lambda1 (drift towards in-time best parameter)
lambda1 = 0;
% lambda2 (drift towards global and in-time best parameter)
lambda2 = 1;

% type of diffusion for noise term 1
anisotropic1 = 1;
% sigma (parameter of noise term 1)
sigma1 = 0;
% type of diffusion for noise term 2
anisotropic2 = 1;
% sigma (parameter of noise term 2)
sigma2 = 0.1;

% alpha (weight in Gibbs measure for global and in-time best position computation)
alpha = 10^15;
% beta (regularization parameter for sigmoid)
beta = 10^15; % 'inf' or -1 for using Heaviside function instead of S_beta



%% Initialization
X0mean = [4;4];
X0std = 10;
V0mean = [2;0];
V0std = 0;


%% Use pre-set setting
if pre_setparameters==1
    T = parametersCBO('T');
    dt = parametersCBO('dt');
    N = parametersCBO('N');
    lambda = parametersCBO('lambda');
    gamma = parametersCBO('gamma');
    learning_rate = parametersCBO('learning_rate');
    anisotropic = parametersCBO('anisotropic');
    sigma = parametersCBO('sigma');
    alpha = parametersCBO('alpha');
    V0mean = parametersInitialization('V0mean');
    V0std = parametersInitialization('V0std');
else
    parametersPSO = containers.Map({'T', 'dt', 'N', 'm', 'kappa', 'gamma', 'memory', 'lambda1', 'lambda2', 'anisotropic1', 'sigma1', 'anisotropic2', 'sigma2', 'alpha', 'beta'},...
                                   {  T,   dt,   N,   m,   kappa,   gamma,   memory,   lambda1,   lambda2,   anisotropic1,   sigma1,   anisotropic2,   sigma2,   alpha,   beta});
    parametersInitialization = containers.Map({'X0mean', 'X0std', 'V0mean', 'V0std'},...
                                              {  X0mean,   X0std,   V0mean,   V0std});
end


%% Exemplary Particle
Xex = [[-2;4]]; %[[-2;4],[4.5;1.5],[-1.5;-1.5]];
[~, NUM_EX] = size(Xex);
N = N+NUM_EX;


%% Data Generation

NUM_RUNS = 100;
Expectation_trajectories = zeros(NUM_RUNS,d,1+T/dt);
Xex_trajectories = zeros(NUM_RUNS,d,NUM_EX,1+T/dt);

for r = 1:NUM_RUNS
    
    [Xex_trajectories(r,:,1:NUM_EX,:), Expectation_trajectories(r,:,:), Yex_trajectories(r,:,1:NUM_EX,:), YExpectation_trajectories(r,:,:)] = PSO_trajectories(E,parametersPSO,X0mean,X0std,V0mean,V0std,Xex);
    
end


%% Plotting

set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');

% % color setting
opacity_sampletrajectories = 0.4;
grayish_averagetrajectory = 0.8;


% % plot setting
f = figure('Position', [1200 800 600 500]);
%title('Consensus Based Optimization','Interpreter','latex','FontSize',16)  

% % plotting energy function E
[X,Y] = meshgrid(xrange_plot(1):.002:xrange_plot(2),yrange_plot(1):.002:yrange_plot(2));
XY = [X(:)';Y(:)'];
Z = E(XY);
Z = reshape(Z,size(X));

Eplot = surf(X,Y,Z,'FaceAlpha',0.25); % 0.5 und 0.25
Eplot.EdgeColor = 'None';
hold on

contour(X,Y,Z,20);
hold on

if spatial_plot
    view(-25,12.5)
else
    view(2)
end

xlim(xrange_plot)
ylim(yrange_plot)
if spatial_plot
    zlim([zrange_plot(1),zrange_plot(2)+0.01])
else
    zlim([zrange_plot(1),zrange_plot(2)+10])
end

xticks([-2.5 0 2.5 5])
yticks([-2.5 0 2.5 5])

% way of plotting of all points
if spatial_plot
    F = @(x) E(x);
else
    F = @(x) zrange_plot(2)*ones(size(sum(x.*x))); 
end
%F = @(x) 0*zeros(size(sum(x.*x)));

% % plot global minimizer of energy function E
xstarplot = plot3(xstar(1), xstar(2), F(xstar), '*', 'MarkerSize', 10, 'LineWidth', 1.8, "color", co(5,:));
hold on



if strcmp(plot_XorY,'X')
    XYex_trajectories = Xex_trajectories;
    XYExpectation_trajectories = Expectation_trajectories;
elseif strcmp(plot_XorY,'Y')
    XYex_trajectories = Yex_trajectories;
    XYExpectation_trajectories = YExpectation_trajectories;
else
    error('plot_XorY not known') 
end


for r = 1:NUM_RUNS
    
    if show_expectation
        Expectation_trajectory = reshape(XYExpectation_trajectories(r,:,:), [d,1+T/dt]);
        Expectation_trajectory_plot = plot3(Expectation_trajectory(1,:),Expectation_trajectory(2,:), F(Expectation_trajectory), '-', 'Linewidth', 2, "color", [co(2,:),opacity_sampletrajectories]);
        hold on
    end
    
    for e = 1:NUM_EX
        XYex_trajectory = reshape(XYex_trajectories(r,:,e,:), [d,1+T/dt]);
        XYex_trajectory_plot = plot3(XYex_trajectory(1,:),XYex_trajectory(2,:), F(XYex_trajectory), '-', 'Linewidth', 2, "color", [co(3,:),opacity_sampletrajectories]);
        hold on
    end
    
end

if show_expectation
    Expectation_mean_trajectory = reshape(mean(XYExpectation_trajectories,1), [d,1+T/dt]);
    Expectation_mean_trajectory_plot = plot3(Expectation_mean_trajectory(1,:),Expectation_mean_trajectory(2,:), F(Expectation_mean_trajectory), '-', 'Linewidth', 2.5, "color", grayish_averagetrajectory*co(2,:));
    hold on
end

for e = 1:NUM_EX
    XYex_mean_trajectory = reshape(mean(XYex_trajectories(:,:,e,:),1), [d,1+T/dt]);
    XYex_mean_trajectory_plot = plot3(XYex_mean_trajectory(1,:),XYex_mean_trajectory(2,:), F(XYex_mean_trajectory), '-', 'Linewidth', 2.5, "color", grayish_averagetrajectory*co(3,:));
end

% add initial positions
Xex_0 = plot3(Xex(1,:), Xex(2,:), F(Xex)+0.01, '.', 'MarkerSize', 25, 'LineWidth', 1.8, "color", grayish_averagetrajectory*co(3,:));
hold on

% % replot global minimizer of energy function E
xstarplot = plot3(xstar(1), xstar(2), F(xstar)+0.01, '*', 'MarkerSize', 10, 'LineWidth', 1.8, "color", co(5,:), 'MarkerEdgeColor', co(5,:), 'MarkerFaceColor', co(5,:));
hold on


if strcmp(plot_XorY,'X')
    if show_expectation
        legend([xstarplot, Xex_0, XYex_trajectory_plot, XYex_mean_trajectory_plot, Expectation_trajectory_plot, Expectation_mean_trajectory_plot], ...
            'Global minimizer $x^*$', ...
            'Initial positions of fixed particles',...
            'Sample trajectories for each fixed particle', ...
            'Average trajectory for each fixed particle', ...
            'Sample trajectories of the average particle $\textbf{E}\overline{X}$', ...
            'Average trajectory of the average particle','Location','northeast','Interpreter','latex','FontSize',15)
    else
        legend([xstarplot, Xex_0, XYex_trajectory_plot, XYex_mean_trajectory_plot], ...
            'Global minimizer $x^*$', ...
            'Initial positions of fixed particles',...
            'Sample trajectories for each fixed particle', ...
            'Mean trajectory for each fixed particle', 'Interpreter','latex','FontSize',15)
    end
else
    if show_expectation
        legend([xstarplot, Xex_0, XYex_trajectory_plot, XYex_mean_trajectory_plot, Expectation_trajectory_plot, Expectation_mean_trajectory_plot], ...
            'Global minimizer $x^*$', ...
            'Initial positions of fixed particles',...
            'Sample trajectories of \textbf{local best} for each fixed particle', ...
            'Average trajectory of \textbf{local best} for each fixed particle', ...
            'Sample trajectories of the average particle $\textbf{E}\overline{Y}$', ...
            'Average trajectory of the average particle','Location','northeast','Interpreter','latex','FontSize',15)
    else
        legend([xstarplot, Xex_0, XYex_trajectory_plot, XYex_mean_trajectory_plot], ...
            'Global minimizer $x^*$', ...
            'Initial positions of fixed particles',...
            'Sample trajectories of \textbf{local best} for each fixed particle', ...
            'Mean trajectory of \textbf{local best} for each fixed particle', 'Interpreter','latex','FontSize',15)
    end
end

ax = gca;
ax.FontSize = 14;


%% Save Image
if pdfexport
    disp('Needs to be saved manually to obtain high resolution.')
    disp('(File -> Export Setup -> Rendering -> Resolution: 2400dpi; Star for x* needs to be added manually.)')
    %print(f,['images_videos/CBOIntuition_',objectivefunction],'-dpdf');

    if anisotropic1
        filename = ['PSOIntuition_',objectivefunction,'_anisotropicN',num2str(N),'sigma',num2str(100*sigma),'div100'];
    else
        filename = ['PSOIntuition_',objectivefunction,'_isotropicN',num2str(N),'sigma',num2str(100*sigma),'div100'];
    end
    save(['images_videos/',filename,'_param'], 'objectivefunction', 'E', 'anisotropic', 'vstar', 'd', 'T', 'dt', 'N', 'alpha', 'lambda', 'gamma', 'learning_rate', 'sigma', 'V0mean', 'V0std', 'Vex', 'NUM_RUNS')

    disp('Filename when saved in higher resolution:')
    disp(filename)
    saveas(f,['images_videos/',filename,'.jpg']);
end


%% slightly modified CBO Function
function [Xex_trajectory, Expectation_trajectory, Yex_trajectory, YExpectation_trajectory] = PSO_trajectories(E,parametersPSO,X0mean,X0std,V0mean,V0std,Xex)

% get parameters
[d,~] = size(Xex);
T = parametersPSO('T');
dt = parametersPSO('dt');
N = parametersPSO('N');
alpha = parametersPSO('alpha');

% storage for trajectories
s_Vex = size(Xex);
NUM_EX = s_Vex(2);

Expectation_trajectory = zeros(d,1+T/dt);
Xex_trajectory = zeros(d,NUM_EX,1+T/dt);
YExpectation_trajectory = zeros(d,1+T/dt);
Yex_trajectory = zeros(d,NUM_EX,1+T/dt);

%initialization
X0 = X0mean+X0std*randn(d,N-NUM_EX);
V0 = V0mean+V0std*randn(d,N);
X = [Xex,X0];
V = V0;
Y = X;


Expectation_trajectory(:,1) = sum(X,2)/N;
Xex_trajectory(:,1:NUM_EX,1) = X(:,1:NUM_EX);
YExpectation_trajectory(:,1) = sum(X,2)/N;
Yex_trajectory(:,1:NUM_EX,1) = Y(:,1:NUM_EX);

% PSO
for k = 1:T/dt
    
    % % PSO iteration
    % compute global and in-time best position y_alpha
    y_alpha = compute_yalpha(E, alpha, Y);
    
    % position updates of one iteration of PSO
    [X, Y, V] = PSO_update(E, parametersPSO, y_alpha, X, Y, V);
    
    Expectation = sum(X,2)/N;
    Expectation_trajectory(:,k+1) = Expectation;
    Xex_trajectory(:,1:NUM_EX,k+1) = X(:,1:NUM_EX);
    
    YExpectation = sum(Y,2)/N;
    YExpectation_trajectory(:,k+1) = YExpectation;
    Yex_trajectory(:,1:NUM_EX,k+1) = Y(:,1:NUM_EX);
    
end

end

