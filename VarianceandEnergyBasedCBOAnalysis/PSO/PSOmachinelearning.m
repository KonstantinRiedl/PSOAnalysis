function [ystar_approx, performance_tracking] = PSOmachinelearning(parametersPSO, parametersbatch, parametersInitialization, train_data, train_label, test_data, test_label, image_size, NN_architecture, worker)

% get parameters
epochs = parametersPSO('epochs');
batch_size_N = parametersbatch('batch_size_N');
batch_size_E = parametersbatch('batch_size_E');
full_or_partial_XYV_update = parametersbatch('full_or_partial_XYV_update');

NNtype = NN_architecture('NNtype');
architecture = NN_architecture('architecture');
neurons = NN_architecture('neurons');
d = NN_architecture('d');


% initialization
X = parametersInitialization('X0mean') + parametersInitialization('X0std')*randn(d,parametersPSO('N'));
Y = X;
V = parametersInitialization('V0mean') + parametersInitialization('V0std')*randn(d,parametersPSO('N'));
y_alpha_on_batch_old = zeros(d,1);


% % definition of the risk (objective function E)
% predicted and true label
predicted_label = @(x, data) NN(x, data, image_size, NNtype, architecture, neurons);
true_label_1hot = @(x, label) reshape(repmat((label==(0:9)'),[1, size(x,2)]), [neurons(1,end), size(label,2), size(x,2)]);
% categorical crossentropy loss function
categ_CE_loss = @(x, data, label) -1/neurons(1,end)*reshape(sum(true_label_1hot(x, label).*log(predicted_label(x, data)), 1), [size(data, 2), size(x, 2)]);
% risk (objective function E)
E = @(x, data, label) 1/size(data, 2)*sum(categ_CE_loss(x, data, label),1);


% performance tracking during training
training_batches_per_epoch = size(train_data,2)/batch_size_E;
recording_sample_size = 10000;
recording_frequency = 500; % ensure that recording_frequency divides training_batches_per_epoch
performance_tracking = NaN(3, epochs+1, training_batches_per_epoch/recording_frequency);


% compute, display and save initial performance
rand_indices_train = randsample(size(train_data,2),recording_sample_size)';
rand_indices_test = randsample(size(test_data,2),recording_sample_size)';
alg_state = containers.Map({'epoch', 'epochs', 'batch', 'training_batches_per_epoch'},...
                           {      0,   epochs,       0,                            0});
if nargin==10
    [train_accuracy, test_accuracy, objective_value] = comp_performance(E, Y, train_data(:,rand_indices_train), test_data(:,rand_indices_test), image_size, train_label(:,rand_indices_train), test_label(:,rand_indices_test), NNtype, architecture, neurons, alg_state, 1, worker);
else
    [train_accuracy, test_accuracy, objective_value] = comp_performance(E, Y, train_data(:,rand_indices_train), test_data(:,rand_indices_test), image_size, train_label(:,rand_indices_train), test_label(:,rand_indices_test), NNtype, architecture, neurons, alg_state, 1);
end
performance_tracking(1,1,end) = train_accuracy;
performance_tracking(2,1,end) = test_accuracy;
performance_tracking(3,1,end) = objective_value;


% % PSO
for epoch = 1:epochs
    
    % employ optional particle reduction strategy
    particle_reduction = parametersPSO('particle_reduction');
    if particle_reduction
        varianceY_before = norm(var(Y,0,2));
    end
    
    for batch = 1:training_batches_per_epoch
        
        % % definition of objective function E on current training batch
        % indices of current training batch
        indices_t_b = (batch-1)*batch_size_E+(1:batch_size_E);
        batch_data = train_data(:,indices_t_b);
        batch_label = train_label(indices_t_b);
        % objective function E on current training batch
        E_train_batch = @(x) E(x, batch_data, batch_label);
        objective_function_Y = E_train_batch(Y);
        
        
        % % update of particles' positions 
        particle_batches = parametersPSO('N')/batch_size_N;
        permutation = randperm(parametersPSO('N'));
        X = X(:,permutation);
        Y = Y(:,permutation);
        objective_function_Y = objective_function_Y(:,permutation);
        V = V(:,permutation);
        for particle_batch = 1:particle_batches
            
            % indices of current particle batch
            indices_p_b = (particle_batch-1)*batch_size_N+(1:batch_size_N);
            X_particle_batch = X(:,indices_p_b);
            Y_particle_batch = Y(:,indices_p_b);
            objective_function_Y_batch = objective_function_Y(:,indices_p_b);
            V_particle_batch = V(:,indices_p_b);
            
            % % CBO iteration
            % compute current consensus point yalpha
            yalpha_on_batch = compute_yalpha(E_train_batch, parametersPSO('alpha'), Y_particle_batch, objective_function_Y_batch);

            % position updates of one iteration of PSO
            if strcmp(full_or_partial_XYV_update, 'partial')
                [X_particle_batch, Y_particle_batch, V_particle_batch, objective_function_Y_batch] = PSO_update(E_train_batch, parametersPSO, yalpha_on_batch, X_particle_batch, Y_particle_batch, V_particle_batch, objective_function_Y_batch);
                X(:,indices_p_b) = X_particle_batch;
                Y(:,indices_p_b) = Y_particle_batch;
                objective_function_Y(:,indices_p_b) = objective_function_Y_batch;
                V(:,indices_p_b) = V_particle_batch;
                clear permutation
            elseif strcmp(full_or_partial_XYV_update, 'full')
                [X, Y, V, objective_function_Y] = PSO_update(E_train_batch, parametersPSO, yalpha_on_batch, X, Y, V, objective_function_Y); %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            else
                error('full_or_partial_V_update type not known.')
            end
            
            % random Brownian motion if yalpha_on_batch is not changing
            if norm(y_alpha_on_batch_old-yalpha_on_batch, 'inf') < 10^-5
                dB = randn(d,parametersPSO('N'));
                sigma = max(parametersPSO('sigma1'), parametersPSO('sigma2'));
                dt = parametersPSO('dt');
                %X = X + sigma*sqrt(dt).*dB;
                V = V + sigma*sqrt(dt).*dB;
                clear dB sigma dt
            end
            y_alpha_on_batch_old = yalpha_on_batch;
            clear yalpha_on_batch
            
        end
        
        % compute, display and save current performance
        if mod(batch,recording_frequency)==0
            rand_indices_train = randsample(size(train_data,2),recording_sample_size)';
            rand_indices_test = randsample(size(test_data,2),recording_sample_size)';
            alg_state = containers.Map({'epoch', 'epochs', 'batch', 'training_batches_per_epoch'},...
                                       {  epoch,   epochs,   batch,  training_batches_per_epoch});
            if nargin==10
                [train_accuracy, test_accuracy, objective_value] = comp_performance(E, Y, train_data(:,rand_indices_train), test_data(:,rand_indices_test), image_size, train_label(:,rand_indices_train), test_label(:,rand_indices_test), NNtype, architecture, neurons, alg_state, 1, worker);
            else
                [train_accuracy, test_accuracy, objective_value] = comp_performance(E, Y, train_data(:,rand_indices_train), test_data(:,rand_indices_test), image_size, train_label(:,rand_indices_train), test_label(:,rand_indices_test), NNtype, architecture, neurons, alg_state, 1);
            end
            performance_tracking(1,epoch+1,batch/recording_frequency) = train_accuracy;
            performance_tracking(2,epoch+1,batch/recording_frequency) = test_accuracy;
            performance_tracking(3,epoch+1,batch/recording_frequency) = objective_value;
            
            clear E_train rand_indices_train rand_indices_test
        end
        
    end
    
    % employ optional particle reduction strategy (PRS)
    speedup_PRS = 0.2;
    if particle_reduction
        varianceY = norm(var(Y,0,2));
        reduction_factor = min(max(1-speedup_PRS,(speedup_PRS*varianceY+(1-speedup_PRS)*varianceY_before)/varianceY_before),1);
        if reduction_factor<1
            parametersPSO('N') = ceil(parametersPSO('N')/batch_size_N*reduction_factor)*batch_size_N;
            X = X(:,randsample(size(X,2), parametersPSO('N'))');
            Y = Y(:,randsample(size(Y,2), parametersPSO('N'))');
            V = V(:,randsample(size(V,2), parametersPSO('N'))');
        end
    end
    
    % employ optional and cooling strategy (CS) 
    parameter_cooling = parametersPSO('parameter_cooling');
    if parameter_cooling
        parametersPSO('sigma1') = parametersPSO('sigma1')*log2(epoch+1)/log2(epoch+2);
        parametersPSO('sigma2') = parametersPSO('sigma2')*log2(epoch+1)/log2(epoch+2);
        parametersPSO('alpha') = parametersPSO('alpha')*2;
    end
    
    
    % saving results and parameters
    if strcmp(NNtype, 'fully_connected') && length(architecture)==1
        NNtype_save = 'ShallowNN';
    elseif strcmp(NNtype, 'fully_connected') && length(architecture)>1
        NNtype_save = 'DeepNN';
    elseif strcmp(NNtype, 'CNN')
        NNtype_save = 'CNN';
    else
        error('NNtype does not exist')
    end
    if nargin==10
        filename = ['CBOandPSO/NN/results/PSO/', NNtype_save, '/', 'PSOMNIST_worker', num2str(worker), '_N_', num2str(parametersPSO('N')), '_memory_', num2str(parametersPSO('memory')), '_lambda1_', num2str(parametersPSO('lambda1')*10), 'div10', '_m_', num2str(parametersPSO('m')*10), 'div10_', num2str(epochs), 'epochs', '_preliminary'];
    else
        filename = ['CBOandPSO/NN/results/PSO/', NNtype_save, '/', 'PSOMNIST', '_N_', num2str(parametersPSO('N')), '_memory_', num2str(parametersPSO('memory')), '_lambda1_', num2str(parametersPSO('lambda1')*10), 'div10', '_m_', num2str(parametersPSO('m')*10), 'div10_', num2str(epochs), 'epochs', '_preliminary'];
    end
    parsave_PSO(filename, y_alpha_on_batch_old, performance_tracking, image_size, NN_architecture, NNtype, architecture, neurons, d, epochs, parametersPSO('dt'), parametersPSO('N'), parametersPSO('m'), parametersPSO('kappa'), parametersPSO('gamma'), parametersPSO('alpha'), parametersPSO('beta'), parametersPSO('memory'), parametersPSO('lambda1'), parametersPSO('lambda2'), parametersPSO('anisotropic1'), parametersPSO('sigma1'), parametersPSO('anisotropic2'), parametersPSO('sigma2'), particle_reduction, parameter_cooling, batch_size_N, batch_size_E, full_or_partial_XYV_update, parametersInitialization('X0mean'), parametersInitialization('X0std'), parametersInitialization('V0mean'), parametersInitialization('V0std'))
 
end

ystar_approx = nan;
%E_train = @(x) E(x, train_data, train_label);
%ystar_approx = compute_valpha(E_train, parametersPSO('alpha'), Y);

% saving final results and parameters
if strcmp(NNtype, 'fully_connected') && length(architecture)==1
    NNtype_save = 'ShallowNN';
elseif strcmp(NNtype, 'fully_connected') && length(architecture)>1
    NNtype_save = 'DeepNN';
elseif strcmp(NNtype, 'CNN')
    NNtype_save = 'CNN';
else
    error('NNtype does not exist')
end
if nargin==10
    filename = ['CBOandPSO/NN/results/PSO/', NNtype_save, '/', 'PSOMNIST_worker', num2str(worker), '_N_', num2str(parametersPSO('N')), '_memory_', num2str(parametersPSO('memory')), '_lambda1_', num2str(parametersPSO('lambda1')*10), 'div10', '_m_', num2str(parametersPSO('m')*10), 'div10_', num2str(epochs), 'epochs', '_final'];
else
    filename = ['CBOandPSO/NN/results/PSO/', NNtype_save, '/', 'PSOMNIST', '_N_', num2str(parametersPSO('N')), '_memory_', num2str(parametersPSO('memory')), '_lambda1_', num2str(parametersPSO('lambda1')*10), 'div10', '_m_', num2str(parametersPSO('m')*10), 'div10_', num2str(epochs), 'epochs', '_final'];
end
parsave_PSO(filename, ystar_approx, performance_tracking, image_size, NN_architecture, NNtype, architecture, neurons, d, epochs, parametersPSO('dt'), parametersPSO('N'), parametersPSO('m'), parametersPSO('kappa'), parametersPSO('gamma'), parametersPSO('alpha'), parametersPSO('beta'), parametersPSO('memory'), parametersPSO('lambda1'), parametersPSO('lambda2'), parametersPSO('anisotropic1'), parametersPSO('sigma1'), parametersPSO('anisotropic2'), parametersPSO('sigma2'), particle_reduction, parameter_cooling, batch_size_N, batch_size_E, full_or_partial_XYV_update, parametersInitialization('X0mean'), parametersInitialization('X0std'), parametersInitialization('V0mean'), parametersInitialization('V0std'))

end