% PSO numerical example
%
% This script tests PSO numerically and outputs the approximation to the
% global minimizer.
%

%%
clear; clc; close all;

co = set_color();


%% Energy Function E

% % dimension of the ambient space
d = 20;

% % energy function E
% (E is a function mapping columnwise from R^{d\times N} to R)
objectivefunction = 'Rastrigin';
[E, parametersE, ~, ~] = objective_function(objectivefunction, d, 'PSO');

% global minimizer
vstar = zeros(d,1); %fminbnd(E,xrange_plot(1),xrange_plot(2));


%% Parameters of PSO Algorithm

% time horizon
T = 100;

% discrete time size
dt = 0.01;

% number of particles
N = 100;

% inertia weight
m = 10^(-2.5);

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
sigma1 = lambda1*4;
% type of diffusion for noise term 2
anisotropic2 = 1;
% sigma (parameter of noise term 2)
sigma2 = 11;

% alpha (weight in Gibbs measure for global and in-time best position computation)
alpha = 100;
% beta (regularization parameter for sigmoid)
beta = -1; % 'inf' or -1 for using Heaviside function instead of S_beta


%% Initialization
X0mean = 2*ones(d,1);
X0std = 4;
V0mean = zeros(d,1);
V0std = 0;

parametersPSO = containers.Map({'T', 'dt', 'N', 'm', 'kappa', 'gamma', 'memory', 'lambda1', 'lambda2', 'anisotropic1', 'sigma1', 'anisotropic2', 'sigma2', 'alpha', 'beta'},...
                               {  T,   dt,   N,   m,   kappa,   gamma,   memory,   lambda1,   lambda2,   anisotropic1,   sigma1,   anisotropic2,   sigma2,   alpha,   beta});
parametersInitialization = containers.Map({'X0mean', 'X0std', 'V0mean', 'V0std'},...
                                          {  X0mean,   X0std,   V0mean,   V0std});


%% PSO Algorithm
%initialization
X0 = X0mean+X0std*randn(d,N);
V0 = V0mean+V0std*randn(d,N);
X = X0;
V = V0;

% PSO
[xstar_app] = PSO(E, parametersPSO, X0, V0);

fmtvstar     = ['global minimizer (numerically): [', repmat('%g, ', 1, numel(vstar)-1), '%g]\n'];
fprintf(fmtvstar, vstar)
fprintf('          with objective value: %f\n', E(vstar))

fmtvstar_app = ['final approximated minimizer  : [', repmat('%g, ', 1, numel(xstar_app)-1), '%g]\n'];
fprintf(fmtvstar_app, xstar_app)
fprintf('          with objective value: %f\n', E(xstar_app))
if E(xstar_app)<0.8
    fprintf('************** PSO   successful **************\n')
else
    fprintf('************** PSO UNsuccessful **************\n')
end

