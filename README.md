# FluidFrame: A reinforcement learning agent that navigates a cellular flow
This repository contains Python code for training a reinforcement learning agent to navigate a cellular flow. This accompanies the following publication:
FluidFrame is a research framework for training and evaluating reinforcement learning agents navigating a cellular flow. It originated as companion code for the following publication:
_Shruti Mishra, Michael Chang, Vamsi Spandan, and Shmuel M. Rubinstein, A perspective on fluid mechanical environments for challenges in reinforcement learning (2025). In Finding the Frame workshop at the Reinforcement Learning Conference._

The framework is being developed beyond the original paper, with ongoing work on continuous environments.

This implementation derives from the work of Colabrese et al. (2017) with the following differences:
- The swimmer advances in the environment using a forward Euler integration scheme, versus the Runge-Kutta method in Colabrese et al. (2017).
- The environment transitions are discretised using fixed timesteps, versus a new state being specified to the agent when the observation changes in Colabrese et al. (2017).

![swimmer-results](https://sm7610.github.io/assets/work/swimmer-results.png)

Figure 1: Plots of trained and naïve swimmers navigating through a cellular flow. The training is done using `main.py`, and the evaluation is done using `eval.py`.

## Installation guide
With [dedalus](http://dedalus-project.org) as a dependency for some environments in this repository, the installation guide uses `conda`. An example of the installation instructions is below:

```
$ conda create -n fluidframe_env python=3.13.7  # creates a virtual environment for the installation
$ conda activate fluidframe_env  # activates the virtual environment
$ conda install -c conda-forge dedalus  # installs dedalus
$ pip install tqdm
```

## Usage
The installation has been tested on a MacOS environment. The code can be verified and run as follows:

### Testing
```
$ pytest tests/.
```

### Example
The code generates the following example output:
```
➜  fluidframe git:(main) ✗ OMP_NUM_THREADS=1 python main.py --use-dedalus-environment
Using dedalus to specify flow variables ...
  0%|                                                                     | 0/5000 [00:00<?, ?it/s]
The policy is [1 0 3 1 0 3 1 1 3 0 0 0].
Episode 0 return: 	 22.267112542257625
.
.
.
Last episode return: 	 448.4741962995584
Policy: 	 [2 2 1 1 1 1 1 1 1 0 0 1].
```

The swimmer speed and alignment timescale can be configured via the command line interface, e.g. 
```
$ python main.py --swimmer-speed 0.3 --alignment-timescale 1.0
```

To save checkpoints into a folder `checkpoints`, the command `mkdir checkpoints` can be used from the root folder of the repository.

## Bibliography
Burns, K.J., Vasil, G.M., Oishi, J.S., Lecoanet, D. and Brown, B.P., 2020. Dedalus: A flexible framework for numerical simulations with spectral methods. Physical Review Research, 2(2), p.023068.

Colabrese, S., Gustavsson, K., Celani, A. and Biferale, L., 2017. Flow navigation by smart microswimmers via reinforcement learning. Physical Review Letters, 118(15), p.158004.

Mishra, S., Chang, M., Arza, V.S. and Rubinstein, S., 2025. A perspective on fluid mechanical environments for challenges in reinforcement learning. In Finding the Frame Workshop at the Reinforcement Learning Conference.

### Additional acknowledgements

[Maxwell Svetlik](https://github.com/maxsvetlik)