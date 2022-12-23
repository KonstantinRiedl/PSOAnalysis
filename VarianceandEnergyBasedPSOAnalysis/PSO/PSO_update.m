% Position updates of one iteration of consensus based optimization (PSO)
%
% This function performs the position updates of one iteration of PSO.
% 
% 
% [X, Y, V, objective_function_Y] = PSO_update(E, grad_E, parametersPSO, y_alpha, X, Y, V, objective_function_Y)
% 
% input:    E                    = objective function E (as anonymous function)
%           grad_E               = gradient of objective function E (as anonymous function) <- currently not used
%           parametersPSO        = suitable parameters for PSO
%                                = [T, dt, N, m, kappa, gamma, lambda1, lambda2, anisotropic1, sigma1, anisotropic2, sigma2, alpha, beta]
%               - T              = time horizon
%               - dt             = time step size
%               - N              = number of particles
%               - m              = inertia weight
%               - kappa          = scaling parameter (usually 1/dt)
%               - gamma          = friction coefficient
%               - lambda1        = drift towards in-time best parameter
%               - lambda2        = drift towards global and in-time best parameter
%               - anisotropic1   = noise 1 type
%               - sigma1         = noise parameter 1
%               - anisotropic2   = noise 2 type
%               - sigma2         = noise parameter 2
%               - alpha          = weight/temperature parameter
%               - beta           = regularization parameter for sigmoid
%           y_alpha              = global and in-time best position y_alpha
%           X                    = former positions of the particles
%           Y                    = former best positions of the particles
%           V                    = former velocities of the particles
%           objective_function_Y = former objective values of best positions
%           
% output:   X                    = positions of the particles afterwards
%           Y                    = best positions of the particles afterwards
%           V                    = velocities of the particles afterwards
%           objective_function_Y = objective values of best positions afterwards
%

function [X, Y, V, objective_function_Y] = PSO_update(E, grad_E, parametersPSO, y_alpha, X, Y, V, objective_function_Y)

% get parameters
d = size(X,1);
dt = parametersPSO('dt');
m = parametersPSO('m');
kappa = parametersPSO('kappa');
gamma = parametersPSO('gamma');
memory = parametersPSO('memory');
lambda1 = parametersPSO('lambda1');
lambda2 = parametersPSO('lambda2');
anisotropic1 = parametersPSO('anisotropic1');
sigma1 = parametersPSO('sigma1');
anisotropic2 = parametersPSO('anisotropic2');
sigma2 = parametersPSO('sigma2');
beta = parametersPSO('beta');


% % particle iteration step (according to SDE system)

% % implicit Euler-Maruyama discretization
% % update of the velocity
% friction
V = (m/(m+gamma*dt))*V;
% drift towards the in-time best particle position
%  (individually for each particle, i.e., locally)
if memory
    V = V + (lambda1/(m+gamma*dt))*(Y-X)*dt;
end
% drift towards the global and in-time best position y_alpha
V = V + (lambda2/(m+gamma*dt))*(y_alpha*ones(1,size(V,2))-X)*dt;
% noise terms
if memory
    dB1 = randn(d,size(V,2));
    if anisotropic1
        V = V + (sigma1/(m+gamma*dt))*abs(Y-X)*sqrt(dt).*dB1;
    else
        V = V + (sigma1/(m+gamma*dt))*vecnorm(Y-X,2,1)*sqrt(dt).*dB1;
    end
end
dB2 = randn(d,size(V,2));
if anisotropic2
    V = V + (sigma2/(m+gamma*dt))*abs(y_alpha*ones(1,size(V,2))-X)*sqrt(dt).*dB2;
else
    V = V + (sigma2/(m+gamma*dt))*vecnorm(y_alpha*ones(1,size(V,2))-X,2,1)*sqrt(dt).*dB2;
end

% % update of the particle positions
X = X + V*dt;

if memory
    % % update of the in-time best particle positions (for each particle)
    if strcmp(beta, 'inf') || beta==-1
        if nargin == 8
            objective_function_X = E(X);
            Y = Y + (X-Y).*double((objective_function_Y-objective_function_X)>0);
            objective_function_Y = min(objective_function_Y, objective_function_X);
        else
            Y = Y + (X-Y).*double(E(Y)-E(X)>0);
            objective_function_Y = nan;
        end
    else
        if nargin == 8
            objective_function_X = E(X);
            Y = Y + kappa*(X-Y).*S_beta(E, beta, X, Y, objective_function_X, objective_function_Y)*dt;
            objective_function_Y = E(Y);
        else
            Y = Y + kappa*(X-Y).*S_beta(E, beta, X, Y)*dt;
            objective_function_Y = nan;
        end
    end
else
    Y = X;
    if nargin == 7
        objective_function_Y = E(X);
    else
        objective_function_Y = nan;
    end
end


clear E parametersPSO y_alpha
clear d dt N m kappa gamma lambda1 lambda2 anisotropic1 sigma1 anisotropic2 sigma2 beta
clear dB1 dB2


% % alternative explicit Euler-Maruyama discretization
% % % update of the particle positions
% X = X + V*dt;
% 
% % % update of the in-time best particle positions (for each particle)
% Y = Y + kappa*(X-Y).*S_beta(E, beta, X, Y)*dt;
% 
% % % update of the velocity
% % friction
% V = V - gamma/m*V*dt;
% % drift towards the in-time best particle position
% %  (individually for each particle, i.e., locally)
% V = V + lambda1/m*(Y-X)*dt;
% % drift towards the global and in-time best position y_alpha
% V = V + lambda2/m*(y_alpha*ones(1,N)-X)*dt;
% % noise terms
% if anisotropic1
%     V = V + (sigma1/(m+gamma*dt))*abs(Y-X)*sqrt(dt).*dB1;
% else
%     V = V + (sigma1/(m+gamma*dt))*vecnorm(Y-X,2,1)*sqrt(dt).*dB1;
% end
% if anisotropic2
%     V = V + (sigma2/(m+gamma*dt))*abs(y_alpha*ones(1,N)-X)*sqrt(dt).*dB2;
% else
%     V = V + (sigma2/(m+gamma*dt))*vecnorm(y_alpha*ones(1,N)-X,2,1)*sqrt(dt).*dB2;
% end

end
