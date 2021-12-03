% MNIST Classification with PSO for HPC
%
% This function 
%

function MNISTClassificationPSO_forHPC(numberofworkers)


%% MNIST Dataset

image_size = [28, 28]; % the standard size of MNIST is [28, 28]

[train_data, train_label, test_data, test_label] = load_MNIST(image_size);


%% Neural Network Architecture

% type of the neural network
NNtype = 'CNN'; % fully_connected or CNN

% architecture of the neural network
if strcmp(NNtype, 'fully_connected')
    architecture = ['d'];
    neurons =      [ 10];
    %architecture = ['d', 'd', 'd', 'd'];
    %neurons =      [ 20,  10,  10,  10];
elseif strcmp(NNtype, 'CNN')
    architecture = ['c', 'p', 'c', 'p', 'd'];
    neurons =      [  4,   0,   3,   0,  10 ; % #filters (for 'c') and #weights (for 'd')
                      5,   2    5,   2,   0]; % size of kernel for 'c' and 'p'
else
    error('NN architecture type not known.')
end

% % parameter dimension aka dimension of the optimization space
[d, ~, ~] = number_of_weightsbiases(NNtype, architecture, neurons, image_size);

NN_architecture = containers.Map({'NNtype', 'architecture', 'neurons', 'd'},...
                                 {  NNtype,   architecture,   neurons,   d});


%% Parameters of PSO Algorithm

% number of epochs
epochs = 100;

% discrete time size
dt = 0.1;

% number of particles
N = 100;
% particle reduction strategy (for N)
particle_reduction = 0;


% % PSO parameters
% parameter cooling strategy (for sigma and alpha)
parameter_cooling = 1;


%
kappa = 1/dt;


% lambda2 (drift towards global and in-time best parameter)
lambda2 = 1;

% type of diffusion for noise term 1
anisotropic1 = 1;
% type of diffusion for noise term 2
anisotropic2 = 1;


% beta (regularization parameter for sigmoid)
beta = -1; % 'inf' or -1 for using Heaviside function instead of S_beta






% mini batch size in the number of particles
batch_size_N = 100; % ensure that batch_size_E divides N

% batch size used for the evaluation of the objective
batch_size_E = 60; % ensure that batch_size_N divides size(train_data,2)

% full or partial update
full_or_partial_XYV_update = 'full';

parametersbatch = containers.Map({'batch_size_N', 'batch_size_E', 'full_or_partial_XYV_update'},...
                                 {  batch_size_N,   batch_size_E,   full_or_partial_XYV_update});

                             

% % parameters different on each worker

% N: number of particles
% m: inertia weight
% kappa: 
% lambda1: drift towards in-time best parameter
% lambda2: drift towards global and in-time best parameter
% anisotropic1: type of diffusion for noise term 1
% sigma1 (parameter of noise term 1)
% anisotropic2: type of diffusion for noise term 2
% sigma2: parameter of noise term 2
% alpha: weight in Gibbs measure for global and in-time best position computation
% beta: regularization parameter for sigmoid ('inf' or -1 for using Heaviside function instead of S_beta)

%               m, memory, lambda1,    sigma2, alpha
parameters = [0.1,      1,       0, sqrt(0.4),    50;
              0.2,      1,       0, sqrt(0.4),    50;
              0.4,      1,       0, sqrt(0.4),    50;
              0.2,      1,     0.4, sqrt(0.4),    50];


%% Initialization

%initialization
X0mean = zeros(d,1);
X0std = 1;
V0mean = zeros(d,1);
V0std = 0;

parametersInitialization = containers.Map({'X0mean', 'X0std', 'V0mean', 'V0std'},...
                                          {  X0mean,   X0std,   V0mean,   V0std});


%% PSO or PSO Algorithm

parfor (worker = 1:size(parameters,1), numberofworkers)
    
    disp(['Worker ', num2str(worker), ' started.'])
    

    % % PSO parameters
    
    % inertia weight
    m = parameters(worker, 1);
    

    % friction coefficient
    gamma = 1-m;

    % memory
    memory = parameters(worker, 2); % 0 or 1
    % lambda1, sigma1, kappa and beta have no effect for memory=0.


    % lambda1 (drift towards in-time best parameter)
    lambda1 = parameters(worker, 3);

    % sigma (parameter of noise term 1)
    sigma1 = lambda1*parameters(worker, 4);
    % sigma (parameter of noise term 2)
    sigma2 = parameters(worker, 4);

    % alpha (weight in Gibbs measure for global and in-time best position computation)
    alpha = parameters(worker, 5);



    parametersPSO = containers.Map({'epochs', 'dt', 'N', 'm', 'kappa', 'gamma', 'alpha', 'beta', 'memory', 'lambda1', 'lambda2', 'anisotropic1', 'sigma1', 'anisotropic2', 'sigma2', 'particle_reduction', 'parameter_cooling'},...
                                   {  epochs,   dt,   N,   m,   kappa,   gamma,   alpha,   beta,   memory,   lambda1,   lambda2,   anisotropic1,   sigma1,   anisotropic2,   sigma2,   particle_reduction,   parameter_cooling});

    % PSO
    PSOmachinelearning(parametersPSO, parametersbatch, parametersInitialization, train_data, train_label, test_data, test_label, image_size, NN_architecture, worker);

    disp(['Worker ', num2str(worker), ' finished.'])
end

end



%% Parameter Junks

% Runs on Cluster:
% fully_connected, shallow
% parameters = [0.05,     1,       0, sqrt(0.4),    50;
%               0.1,      1,       0, sqrt(0.4),    50;
%               0.2,      1,       0, sqrt(0.4),    50;
%               0.4,      1,     0.4, sqrt(0.4),    50];
% parameters = [0.4,      1,       0, sqrt(0.4),    50;
%               0.2,      1,     0.2, sqrt(0.4),    50;
%               0.2,      0,       0, sqrt(0.4),    50;
%               0.2,      1,     0.4, sqrt(0.4),    50];
% fully_connected, shallow II
% parameters = [0.1,      1,       0, sqrt(0.4),    50;
%               0.2,      1,       0, sqrt(0.4),    50;
%               0.4,      1,       0, sqrt(0.4),    50;
%               0.2,      1,     0.4, sqrt(0.4),    50];
% parameters = [0.1,      0,       0, sqrt(0.4),    50;
%               0.2,      0,       0, sqrt(0.4),    50;
%               0.4,      0,       0, sqrt(0.4),    50];

% CNN
% parameters = [0.1,      1,       0, sqrt(0.4),    50;
%               0.2,      1,       0, sqrt(0.4),    50;
%               0.4,      1,       0, sqrt(0.4),    50;
%               0.2,      1,     0.4, sqrt(0.4),    50];
% parameters = [0.1,      0,       0, sqrt(0.4),    50;
%               0.2,      0,       0, sqrt(0.4),    50;
%               0.4,      0,       0, sqrt(0.4),    50];




















