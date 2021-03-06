# Ranking Policy Decisions

Code associated with paper [Ranking Policy Decisions](https://arxiv.org/abs/2008.13607).
Website with results and examples available [here](https://sites.google.com/view/rankingpolicydecisions/home).

Given a policy _p_, a default policy _d_, and some condition, this code will allow you to score states 
according to how important it is to follow _p_ over _d_. If we choose _d_ to be some simple default action,
we can understand in which states _p_ is actually useful over doing something obvious.

## Setting Up (Python) Environment

Using [Anaconda](https://www.anaconda.com/) to make a new
environment:

```
conda env create -f environment.yml
conda activate polrank
```

Using pip (python >=3.8 required):

```
pip3 install -r requirements.txt
```

## Setting Up (RL) Environments

For all of them, start with:

```
pip3 install gym
```
#### Carpole
For [CartPole](https://gym.openai.com/envs/CartPole-v0/), that's it!

#### Minigrid
For [Minigrid](https://github.com/maximecb/gym-minigrid):

```
pip3 install gym-minigrid
pip3 install torch-ac
```

#### Atari Games
For [Atari Games](https://gym.openai.com/envs/#atari):

```
pip3 install atari-py
pip3 install gym[atari]
```

or, on windows:

```
pip3 install -f https://github.com/Kojoley/atari-py/releases atari_py
pip3 install gym[atari]
```


## Basic Commands

To run experiments from paper, use q.py. Refer to the following table for the environment ID:

|                   Environment |  ID |
|:------------------------------|:--:|
| MiniGrid                      |  0 |
| CartPole                      |  1 |
| Atlantis                      |  2 |
| Boxing                        |  3 |
| Breakout                      |  4 |
| Breakout (custom abstraction) |  5 |
| Chopper Command               |  6 |
| Kung Fu Master                |  7 |
| Pong                          |  8 |
| Seaquest                      |  9 |
| Space Invaders                | 10 |

Run the experiment with:
```
python3 q.py [ID]
```
First, this will download any models needed to run the experiments. Then, this will run a counting phase, in which the test suite is built, a scoring phase in which all the states are scored, and an interpolating phase, in which pruned policies are made and tested. Results will be stored in the ```results``` folder.

## Credit
Credit for each environment and policy-training method is supplied in the README for each environment, in ```polexp/environments/```
