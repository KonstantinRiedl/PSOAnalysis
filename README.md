# PSOAnalysis
Numerical analysis of Particle Swarm Optimization

PSO is a multi-agent metaheuristic derivative-free optimization method capable of globally minimizing nonconvex and nonsmooth functions in high dimensions. It was initially introduced by Kennedy and Eberhart in the 1990s.

Version 1.2

Date 31.05.2023

------

## R e f e r e n c e

### On the Global Convergence of Particle Swarm Optimization Methods

https://link.springer.com/article/10.1007/s00245-023-09983-3

by

- Hui &nbsp; H u a n g &nbsp; (University of Graz), 
- Jinniao &nbsp; Q i u &nbsp; (University of Calgary),
- Konstantin &nbsp; R i e d l &nbsp; (Technical University of Munich & Munich Center for Machine Learning)

------

## D e s c r i p t i o n

MATLAB implementation, which illustrates PSO and the influence of its parameters, and tests the method on a competitive high-dimensional and well understood benchmark problem in the machine learning literature.

For the reader's convenience we describe the folder structure in what follows:

BenchmarkFunctions
* objective_function.m: objective function generator
* ObjectiveFunctionPlot1/2d.m: plotting routine for objective function

VarianceandEnergyBasedCBOAnalysis
* analyses: convergence and parameter analyses of PSO
    * PSONumericalExample.m: testing script
    * PSOParameters_PhaseTransition.m: Phase transition diagrams for parameter analysis
* results: folder to save numerical results
    * plot_phasetransitiondiagram: plotting routine for phase diagrams
    * PhaseTransitionDiagrams_msigma2: folder for phase transition diagrams for m and sigma2
* PSO: code of PSO optimizer
    * compute_yalpha.m: computation of consensus point
    * S_beta.m: function to compare current with in-time best position
    * PSO_update: one PSO step
    * PSO.m: PSO optimizer
    * PSOmachinelearning.m: PSO optimizer for machine learning applications
* visualizations: visualization of the CBO dynamics
    * PSODynamicsIllustration.m: Illustration of the PSO dynamics
    * PSOIllustrative.m: Illustration of the PSO at work

NN: machine learning experiments with PSO as optimization method for training
* architecture
    * NN.m: forward pass of NN
    * eval_accuracy.m: evaluate training or test accuracy
    * comp_performance.m: compute and display loss and training or test accuracy
* data: data and function to load data
* Scripts_for_PSO
    * MNISTClassificationPSO.m: script training the NN for MNIST with PSO
* results/PSO: folder to save numerical results
    * plot_training_testing_accuracy: plotting routine for performance plots
    * CNN: folder for CNN results
    * ShallowNN: folder for shallow NN results
    * plot_loss_and_testing_accuracy_PSON100/1000: plotting routine for performance plots

------

## C i t a  t i o n

```bibtex
@article{huang2023global,
  title={On the global convergence of particle swarm optimization methods},
  author={Huang, Hui and Qiu, Jinniao and Riedl, Konstantin},
  journal={Applied Mathematics \& Optimization},
  volume={88},
  number={2},
  pages={30},
  year={2023},
  publisher={Springer}
}
```
