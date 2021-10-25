# Ranking Policy Decisions

Code associated with paper [Ranking Policy Decisions](https://arxiv.org/abs/2008.13607).

Given a policy _p_, a default policy _d_, and some condition, this code will allow you to score states 
according to how important it is to follow _p_ over _d_
when trying to satisfy the condition. 
If we choose _d_ to be some simple default action,
we can understand in which states _p_ is actually useful over doing something obvious.

## Setting Up (Python) Environment

Using [Anaconda](https://www.anaconda.com/) to make a new
environment:

```
conda env create -f env.yml
conda activate polrank
```

## Setting Up (RL) Environments

#### Carpole
For [CartPole](https://gym.openai.com/envs/CartPole-v0/), that's it!

#### Minigrid
For [Minigrid](https://github.com/maximecb/gym-minigrid):

```
pip3 install gym-minigrid
pip3 install torch-ac
```

#### Atari Games
For [Atari Games](https://gym.openai.com/envs/#atari), if you want to use the pre-trained [Atari-Zoo](https://github.com/uber-research/atari-model-zoo) agents (recommended),
you will need to set up their package as well.

```
git clone https://github.com/uber-research/atari-model-zoo.git
cd atari-model-zoo
python3 setup.py install
```

<!-- ```
pip3 install atari-py
pip3 install gym[atari]
```

or, on windows:

```
pip3 install -f https://github.com/Kojoley/atari-py/releases atari_py
pip3 install gym[atari]
``` -->


## Basic Commands

To run experiments from paper, use quick_start.py. Run the experiment with:
```
python3 quick_start.py [ENV_NAME]
```

To show the possible games, do
```
python3 quick_start.py -h
```

Running an experiment will download any models needed to run the experiments. Then, this will run a counting phase, in which the test suite is built, a scoring phase in which all the states are scored, and an interpolating phase, in which pruned policies are made and tested. Results will be stored in the ```results``` folder.

## Beyond quick_start.py

To run the code with all the customization that is available, use:
```
python3 polrank
```

You may want to look at the commands in ```quick_start.py``` to get started.

## Credit
Credit for each environment and policy-training method is supplied in the README for each environment, in ```polrank/environments/```
