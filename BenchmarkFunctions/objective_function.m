% Objective-function function
%
% This function returns the objective function E as an anonymous function
% together with a set of necessary parameters (parametersE) for plotting. 
% Additionally, the function returns two further sets of parameters which
% are suitable parameters for CBO or PSO (parametersOptimizer) as well as a
% suitable initialization (parametersInitialization).
%
% The function E maps columnwise from R^{d\times N} to R, i.e., for a 
% matrixX in R^{d\times N} the function is applied to every column and 
% returns a row vector in R^N (matrix in R^{1\times N})
% 
% 
% [E, parametersE, parametersOptimizer, parametersInitialization] = objective_function(name, d, CBOorPSO)
% 
% input:    name                = name of objective function E
%                                 (e.g., Rastrigin, Ackley, Wshaped)
%           d                   = ambient dimension of the optimization problem
%           CBOorPSO            = type of optimizer (CBO or PSO)
%           
% output:   E                   = objective function E (as anonymous function)
%           parametersE         = necessary parameters for plotting of E
%                               = [xrange_plot,yrange_plot,zrange_plot]
%                 - *range_plot = range of *coordinate as column vector
%     if Optimizer = CBO
%           parametersOptimizer = suitable parameters for CBO
%                               = [T, dt, N, lambda, gamma, learning_rate, sigma, alpha]
%               - T             = time horizon
%               - dt            = time step size
%               - N             = number of particles
%               - lambda        = consensus drift parameter
%               - gamma         = gradient drift parameter
%               - l._r.         = learning rate associated with gradient drift
%               - anisotropic   = type of diffusion/noise (standard: isotropic)
%               - sigma         = exploration/noise parameter
%               - alpha         = weight/temperature parameter alpha
%           parametersInitialization = suitable parameters for initialization
%                               = [V0mean, V0std]
%               - V0mean        = mean of initial distribution
%               - V0std         = standard deviation of initial distribution
%     if Optimizer = PSO
%           parametersOptimizer = suitable parameters for PSO
%                               = [T, dt, N, m, kappa, gamma, lambda1, lambda2, anisotropic1, sigma1, anisotropic2, sigma2, alpha, beta]
%               - T             = time horizon
%               - dt            = time step size
%               - N             = number of particles
%               - m             = inertia weight
%               - kappa         = 
%               - gamma         = friction coefficient
%               - lambda1       = drift towards in-time best parameter
%               - lambda2       = drift towards global and in-time best parameter
%               - anisotropic1  = noise 1 type
%               - sigma1        = noise parameter 1
%               - anisotropic2  = noise 2 type
%               - sigma2        = noise parameter 2
%               - alpha         = weight/temperature parameter
%               - beta          = smooting parameter for local best
%           parametersInitialization = suitable parameters for initialization
%                               = [X0mean, X0std, V0mean, V0std]
%               - X0mean        = mean of initial particle distribution
%               - X0std         = standard deviation of initial particle distribution
%               - V0mean        = mean of initial velocity distribution
%               - V0std         = standard deviation of initial velocity distribution
%

function [E, parametersE, parametersOptimizer, parametersInitialization] = objective_function(name, d, CBOorPSO)

% standard domain parameters
xrange_plot = [-5;10];
yrange_plot = xrange_plot;
zrange_plot = [0;100];

if nargin == 3
    if strcmp(CBOorPSO, 'CBO')
        % standard CBO parameters
        if d == 1
            parametersOptimizer = containers.Map({'T', 'dt', 'N', 'alpha', 'lambda', 'gamma', 'learning_rate', 'anisotropic', 'sigma'},...
                                                 {  4, 0.04, 100,   10^15,        1,       0,            0.01,             0,     0.1});
            parametersInitialization = containers.Map({'V0mean', 'V0std'},...
                                                      {       3,       2});
        elseif d == 2
            parametersOptimizer = containers.Map({'T', 'dt', 'N', 'alpha', 'lambda', 'gamma', 'learning_rate', 'anisotropic', 'sigma'},...
                                                 {  4, 0.02, 100,   10^15,        1,       0,            0.01,             0,     0.1});
            parametersInitialization = containers.Map({'V0mean', 'V0std'},...
                                                      {   [4;6],      8});
        else
            parametersOptimizer = [];
            parametersInitialization = [];
        end
    elseif strcmp(CBOorPSO, 'PSO')
        if d == 1
            parametersOptimizer = containers.Map({'T', 'dt', 'N', 'm', 'kappa', 'gamma', 'lambda1', 'lambda2', 'anisotropic1', 'sigma1', 'anisotropic2', 'sigma2', 'alpha', 'beta'},...
                                                 {  4, 0.04, 100, 0.5,     0.5,     0.5,         1,         1,              0,     0.01,              0,     0.01,   10^15,  10^15});
            parametersInitialization = containers.Map({'X0mean', 'X0std', 'V0mean', 'V0std'},...
                                                      {       3,       2,        0,       0});
        elseif d == 2
            parametersOptimizer = containers.Map({'T', 'dt', 'N', 'm', 'kappa', 'gamma', 'lambda1', 'lambda2', 'anisotropic1', 'sigma1', 'anisotropic2', 'sigma2', 'alpha', 'beta'},...
                                                 {  4, 0.02, 100, 0.5,     0.5,     0.5,         1,         1,              0,     0.01,              0,     0.01,   10^15,  10^15});
            parametersInitialization = containers.Map({'X0mean', 'X0std', 'V0mean', 'V0std'},...
                                                      {   [4;6],       8,    [0;0],       0});
        else
            parametersOptimizer = [];
            parametersInitialization = [];
        end
    else
        error('Optimization method error. Method not known.')
    end
elseif nargin==2
    parametersOptimizer = [];
    parametersInitialization = [];
else
    error('Input error. Wrong number of inputs.')
end
    
if strcmp(name,'Ackley')
    E = @(v) -20*exp(-0.2*sqrt(1/d*sum(v.*v,1)))-exp(1/d*sum(cos(2*pi*v),1))+exp(1)+20;
    xrange_plot = [-5;10];
    parametersE = [xrange_plot*ones(1,d), [0;20]];
    parametersOptimizer = parametersOptimizer;
    parametersInitialization = parametersInitialization;
elseif strcmp(name,'Rastrigin')
    E = @(v) sum(v.*v,1) + 2.5*sum(1-cos(2*pi*v),1);
    xrange_plot = [-2.5;5];
    parametersE = [xrange_plot*ones(1,d), (1*[0;25]+(d-1)*[0;25])]; %(1*[0;25]+(d-1)*[0;30]) for flat 2d plot
    parametersOptimizer = parametersOptimizer;
    parametersInitialization = parametersInitialization;
elseif strcmp(name,'RastriginNonSeparable2')
    E = @(v) sum(v.*v,1) + 2.5*sum(1-cos(2*pi*v),1) + abs(sum((flipud(v)-v).*v,1));
    xrange_plot = [-2.5;5];
    parametersE = [xrange_plot*ones(1,d), 1.8*(1*[0;25]+(d-1)*[0;25])]; %(1*[0;25]+(d-1)*[0;30]) for flat 2d plot
    parametersOptimizer = parametersOptimizer;
    parametersInitialization = parametersInitialization;
elseif strcmp(name,'Quadratic')
    E = @(v) sum(v.*v,1);
    xrange_plot = [-6;8];
    parametersE = [xrange_plot*ones(1,d), d^2*[0;50]];
    parametersOptimizer = parametersOptimizer;
    parametersInitialization = parametersInitialization;
elseif strcmp(name,'Himmelblau')
    error('Not implemented.')
    if d~=2
        error('Dimension Error. The Himmelblau function is only available in 2d.')
    else
        E = @(v) (v(1,:).^2+v(2,:)-11).^2+(v(1,:)+v(2,:).^2-7).^2;
        parametersE = [xrange_plot*ones(1,d), [0;100]];
        parametersOptimizer = parametersOptimizer;
        parametersInitialization = parametersInitialization;
    end
elseif strcmp(name,'Wshaped')
    if d>2
        error('Dimension Error. The Wshaped function is only available in 1d or 2d.')
    elseif d==1
        E = @(v) v.^2.*((v-4).^2+1);
        parametersE = [[-2;6],[0;60]];
        parametersOptimizer = parametersOptimizer;
    elseif d==2
        E = @(v) 0.1*(v(1,:).^2.*((v(1,:)-4).^2+1)+10*(v(2,:)).^2);
        parametersE = [[-2;6],[-4;4],[0;30]];
        parametersOptimizer = parametersOptimizer;
        if strcmp(CBOorPSO, 'CBO')
            parametersInitialization = containers.Map({'V0mean', 'V0std'},...
                                                      {   [4;2],       4});
        elseif strcmp(CBOorPSO, 'PSO')
            parametersInitialization = containers.Map({'X0mean', 'X0std', 'V0mean', 'V0std'},...
                                                      {   [4;2],       4,    [0;0],       1});
        end

    end
elseif strcmp(name,'Wshaped2')
    if d>2
        error('Dimension Error. The Wshaped2 function is only available in 1d or 2d.')
    elseif d==1
        [E, parametersE, parametersOptimizer, parametersInitialization] = objective_function('Wshaped',d);
    elseif d==2
        E = @(v) 0.1*(v(1,:).^2.*((v(1,:)-4).^2+1)+v(2,:).^2.*((v(2,:)-4).^2+1));
        parametersE = [[-2;6],[-2;6],[0;30]];
        parametersOptimizer = parametersOptimizer;
        if strcmp(CBOorPSO, 'CBO')
            parametersInitialization = containers.Map({'V0mean', 'V0std'},...
                                                      {   [3;3],       4});
        elseif strcmp(CBOorPSO, 'PSO')
            parametersInitialization = containers.Map({'X0mean', 'X0std', 'V0mean', 'V0std'},...
                                                      {   [3;3],       4,    [0;0],       1});
        end
    end
elseif strcmp(name,'Rosenbrock')
    if d>2
        error('Dimension Error. The Rosenbrock function is only available in 2d.')
    elseif d==2
        a = 1;
        E = @(v) 0.002*(v(1,:).^2 + 50*((v(2,:)+a^2)-(-v(1,:)+a).^2).^2);
        parametersE = [[-2.5;5],[-2.5;5],[0;40]];
        parametersOptimizer = parametersOptimizer;
        if strcmp(CBOorPSO, 'CBO')
            parametersInitialization = containers.Map({'V0mean', 'V0std'},...
                                                      {   [3;3],       4});
        elseif strcmp(CBOorPSO, 'PSO')
            parametersInitialization = containers.Map({'X0mean', 'X0std', 'V0mean', 'V0std'},...
                                                      {   [3;3],       4,    [0;0],       1});
        end
    end
elseif strcmp(name,'GrandCanyon')
    if d>2
        error('Dimension Error. The GrandCanyon function is only available in 2d.')
    elseif d==2
        E = @(v) 40*(1-exp(-0.1*(-v(2,:)-v(1,:).*(v(1,:)-4)).^2)) + 1*v(1,:).*v(1,:) + 0.5*v(2,:).*v(2,:) + 2.5*sum(1-cos(0.75*pi*v),1);
        xrange_plot = [-2.5;5];
        parametersE = [xrange_plot*ones(1,d), 1.4*(1*[0;25]+(d-1)*[0;25])]; %(1*[0;25]+(d-1)*[0;30]) for flat 2d plot
        parametersOptimizer = parametersOptimizer;
        parametersInitialization = parametersInitialization;
    end
else
    disp('This objective function is not implemented yet.')

end
