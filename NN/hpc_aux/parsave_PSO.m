function parsave_PSO(filename, ystar_app, performance_tracking, image_size, NN_architecture, NNtype, architecture, neurons, d, epochs, dt, N, m, kappa, gamma, alpha, beta, memory, lambda1, lambda2, anisotropic1, sigma1, anisotropic2, sigma2, particle_reduction, parameter_cooling, batch_size_N, batch_size_E, full_or_partial_XYV_update, X0mean, X0std, V0mean, V0std)

    save(filename, 'ystar_app', 'performance_tracking', 'image_size', 'NN_architecture', 'NNtype', 'architecture', 'neurons', 'd', 'epochs', 'dt', 'N', 'm', 'kappa', 'gamma', 'alpha', 'beta', 'memory', 'lambda1', 'lambda2', 'anisotropic1', 'sigma1', 'anisotropic2', 'sigma2', 'particle_reduction', 'parameter_cooling', 'batch_size_N', 'batch_size_E', 'full_or_partial_XYV_update', 'X0mean', 'X0std', 'V0mean', 'V0std')

end

