% Particle swarm optimization (PSO)
%
% This function performs PSO.
% 
% 
% [xstar_approx] = PSO(E, parametersPSO, X0, V0)
% 
% input:    E                  = objective function E (as anonymous function)
%           grad_E             = gradient of objective function E (as anonymous function) <- currently not used
%           parametersPSO      = suitable parameters for PSO
%                              = [T, dt, N, m, kappa, gamma, lambda1, lambda2, sigma1, anisotropic, sigma2, alpha, beta]
%               - T            = time horizon
%               - dt           = time step size
%               - N            = number of particles
%               - m            = inertia weight
%               - kappa        = scaling parameter (usually 1/dt)
%               - gamma        = friction coefficient
%               - lambda1      = drift towards in-time best parameter
%               - lambda2      = drift towards global and in-time best parameter
%               - anisotropic1 = noise 1 type
%               - sigma1       = noise parameter 1
%               - anisotropic2 = noise 2 type
%               - sigma2       = noise parameter 2
%               - alpha        = weight/temperature parameter
%               - beta         = regularization parameter for sigmoid
%           X0                 = initial positions of the particles
%           V0                 = initial velocities of the particles
%           
% output:   ystar_approx       = approximation to ystar
%

function [ystar_approx] = PSO(E, grad_E, parametersPSO, X0, V0)

% get parameters
T = parametersPSO('T');
dt = parametersPSO('dt');
%anisotropic1 = parametersPSO('anisotropic1');
%anisotropic2 = parametersPSO('anisotropic2');
alpha = parametersPSO('alpha');

% initialization
X = X0;
Y = X;
V = V0;

% % PSO
% if anisotropic1 && anisotropic2
% 	disp('PSO with ANisotropic diffusion used.')
% elseif anisotropic1 && ~anisotropic2
%     disp('PSO with ANisotropic diffusion only locally used.')
% elseif ~anisotropic1 && ~anisotropic2
%     disp('PSO with ANisotropic diffusion only globally used.')
% else
% 	disp('PSO with isotropic diffusion used.')
% end
for k = 1:T/dt
    
    % % PSO iteration
    % compute global and in-time best position y_alpha
    y_alpha = compute_yalpha(E, alpha, Y);
    
    % position updates of one iteration of PSO
    [X, Y, V] = PSO_update(E, grad_E, parametersPSO, y_alpha, X, Y, V);
    
end

y_alpha = compute_yalpha(E, alpha, Y);
ystar_approx = y_alpha;

clear E parametersPSO X0 V0
clear T dt anisotropic1 anisotropic2 alpha
clear X Y V y_alpha

end
