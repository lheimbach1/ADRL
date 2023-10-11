# ADRL

## Folder Structure
```
ADRL
├── train (Directory where the training files are)
│   ├── train.py
├── test (Directory where the testing files are)
│   ├── test.py
├── environment (Directory where the environment files are)
│   ├── environment.py
├── benchmarks (Directory where the binomial and binomial smoothing filters are tested for AD)
│   ├── binomial_filters.py
├── models (Directory where the trained policy networks are stored)
│   ├── policy_network.pth
├── plotting (Directory where the plotting files for the plots in the paper are stored)
│   ├── plot_actions.py
│   ├── plot_cumulative_rewards.py
│   ├── plot_energy_spectrum_filters.py
│   ├── plot_energy_spectrum_initial.py
│   ├── plot_rewards.py
│   ├── plot_training_reward.py
│   ├── plot_transfer_function.py
├── figures (Directory where the figures are stored)
├── arrays (Directory where the arrays from the simulations are stored)
```
## Commands for Training and Testing on Local Computer
### Training
```
cd train/
python3 train.py
```
### Testing
```
cd test/
python3 test.py
```

## Commands for Training RL Model on Euler
### Loading Modules in Euler
```
env2lmod
module load gcc/8.2.0 python/3.10.4 hdf5/1.10.1 eth_proxy
```
### Training RL Model with 20 CPU's (running the simulations in parallel) and 1 GPU (training the policy network)
```
cd train/
sbatch --time=04:00:00 --gpus=1 --ntasks=20 --mem-per-cpu=1024 --wrap="python3 train.py"
```