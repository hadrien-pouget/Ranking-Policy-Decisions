# -*- coding: utf-8 -*-
from __future__ import division
import os
import argparse
import numpy as np
import torch
import requests
from torch import optim

from .model import DQN
from utils.download_weights import check_and_dwnld

def parse_args():
  parser = argparse.ArgumentParser(description='Rainbow')
  parser.add_argument('--id', type=str, default='default', help='Experiment ID')
  parser.add_argument('--seed', type=int, default=123, help='Random seed')
  parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
  parser.add_argument('--game', type=str, default='space_invaders', help='ATARI game')
  parser.add_argument('--T-max', type=int, default=int(50e6), metavar='STEPS', help='Number of training steps (4x number of frames)')
  parser.add_argument('--max-episode-length', type=int, default=int(108e3), metavar='LENGTH', help='Max episode length in game frames (0 to disable)')
  parser.add_argument('--history-length', type=int, default=4, metavar='T', help='Number of consecutive states processed')
  parser.add_argument('--architecture', type=str, default='canonical', choices=['canonical', 'data-efficient'], metavar='ARCH', help='Network architecture')
  parser.add_argument('--hidden-size', type=int, default=512, metavar='SIZE', help='Network hidden size')
  parser.add_argument('--noisy-std', type=float, default=0.1, metavar='σ', help='Initial standard deviation of noisy linear layers')
  parser.add_argument('--atoms', type=int, default=51, metavar='C', help='Discretised size of value distribution')
  parser.add_argument('--V-min', type=float, default=-10, metavar='V', help='Minimum of value distribution support')
  parser.add_argument('--V-max', type=float, default=10, metavar='V', help='Maximum of value distribution support')
  parser.add_argument('--model', type=str, metavar='PARAMS', help='Pretrained model (state dict)')
  parser.add_argument('--memory-capacity', type=int, default=int(1e6), metavar='CAPACITY', help='Experience replay memory capacity')
  parser.add_argument('--replay-frequency', type=int, default=4, metavar='k', help='Frequency of sampling from memory')
  parser.add_argument('--priority-exponent', type=float, default=0.5, metavar='ω', help='Prioritised experience replay exponent (originally denoted α)')
  parser.add_argument('--priority-weight', type=float, default=0.4, metavar='β', help='Initial prioritised experience replay importance sampling weight')
  parser.add_argument('--multi-step', type=int, default=3, metavar='n', help='Number of steps for multi-step return')
  parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor')
  parser.add_argument('--target-update', type=int, default=int(8e3), metavar='τ', help='Number of steps after which to update target network')
  parser.add_argument('--reward-clip', type=int, default=1, metavar='VALUE', help='Reward clipping (0 to disable)')
  parser.add_argument('--learning-rate', type=float, default=0.0000625, metavar='η', help='Learning rate')
  parser.add_argument('--adam-eps', type=float, default=1.5e-4, metavar='ε', help='Adam epsilon')
  parser.add_argument('--batch-size', type=int, default=32, metavar='SIZE', help='Batch size')
  parser.add_argument('--learn-start', type=int, default=int(20e3), metavar='STEPS', help='Number of steps before starting training')
  parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
  parser.add_argument('--evaluation-interval', type=int, default=100000, metavar='STEPS', help='Number of training steps between evaluations')
  parser.add_argument('--evaluation-episodes', type=int, default=10, metavar='N', help='Number of evaluation episodes to average over')
  # TODO: Note that DeepMind's evaluation method is running the latest agent for 500K frames ever every 1M steps
  parser.add_argument('--evaluation-size', type=int, default=500, metavar='N', help='Number of transitions to use for validating Q')
  parser.add_argument('--render', action='store_true', help='Display screen (testing only)')
  parser.add_argument('--enable-cudnn', action='store_true', help='Enable cuDNN (faster but nondeterministic)')
  parser.add_argument('--checkpoint-interval', default=0, help='How often to checkpoint the model, defaults to 0 (never checkpoint)')
  parser.add_argument('--memory', help='Path to save/load the memory from')
  parser.add_argument('--disable-bzip-memory', action='store_true', help='Don\'t zip the memory file. Not recommended (zipping is a bit slower and much, much smaller)')
  args = parser.parse_args([])
  return args

download_links = {
  "space_invaders": "https://www.dropbox.com/s/c0to2kha9qnq73b/space_invaders.pth?dl=1",
  "seaquest": "https://www.dropbox.com/s/wjmmyiectfpdwyi/seaquest.pth?dl=1",
  "pong": "https://www.dropbox.com/s/ql3ir5rr5qpzrzt/pong.pth?dl=1",
  "kung_fu_master": "https://www.dropbox.com/s/ho710ew5idl23c3/kung_fu_master.pth?dl=1",
  "chopper_command": "https://www.dropbox.com/s/rrlegfibn54gzuy/chopper_command.pth?dl=1",
  "breakout": "https://www.dropbox.com/s/fnxil6lcqfutens/breakout.pth?dl=1",
  "boxing": "https://www.dropbox.com/s/mi3c9bajbfd0zxw/boxing.pth?dl=1",
  "atlantis": "https://www.dropbox.com/s/o5kb7q2m2w4n4kw/atlantis.pth?dl=1",
}

def get_agent(name, env, device):
  args = parse_args()
  args.model = os.path.join('polrank', 'environments', 'gym_atari', name + '.pth')

  # Download model if we can
  # if not os.path.isfile(args.model):
  #   if name in download_links:
  #     print("Downloading classifier checkpoint...")
  #     r = requests.get(download_links[name])
  #     with open(args.model, 'wb') as f:
  #       f.write(r.content)
  #     print("Download complete")
  #   else:
  #     print("No model checkpoint found for this game, and none can be downloaded. \
  #       Include checkpoint 'polrank/environments/gym_atari/'game_name'.pth")
  #     exit()
  check_and_dwnld(args.model, download_links.get(name, None))

  args.device = device
  # args.architecture = 'data-efficient'
  # args.hidden_size = 256
  agent = Agent(args, env)
  agent.eval()
  return agent

class Agent():
  def __init__(self, args, env):
    self.action_space = env.action_space()
    self.atoms = args.atoms
    self.Vmin = args.V_min
    self.Vmax = args.V_max
    self.support = torch.linspace(args.V_min, args.V_max, self.atoms).to(device=args.device)  # Support (range) of z
    self.delta_z = (args.V_max - args.V_min) / (self.atoms - 1)
    self.batch_size = args.batch_size
    self.n = args.multi_step
    self.discount = args.discount
    self.device = args.device

    self.online_net = DQN(args, self.action_space).to(device=args.device)
    if args.model:  # Load pretrained model if provided
      if os.path.isfile(args.model):
        state_dict = torch.load(args.model, map_location='cpu')  # Always load tensors onto CPU by default, will shift to GPU if necessary
        if 'conv1.weight' in state_dict.keys():
          for old_key, new_key in (('conv1.weight', 'convs.0.weight'), ('conv1.bias', 'convs.0.bias'), ('conv2.weight', 'convs.2.weight'), ('conv2.bias', 'convs.2.bias'), ('conv3.weight', 'convs.4.weight'), ('conv3.bias', 'convs.4.bias')):
            state_dict[new_key] = state_dict[old_key]  # Re-map state dict for old pretrained models
            del state_dict[old_key]  # Delete old keys for strict load_state_dict
        self.online_net.load_state_dict(state_dict)
        print("Loading pretrained model: " + args.model)
      else:  # Raise error if incorrect model path provided
        raise FileNotFoundError(args.model)

    self.online_net.train()

    self.target_net = DQN(args, self.action_space).to(device=args.device)
    self.update_target_net()
    self.target_net.train()
    for param in self.target_net.parameters():
      param.requires_grad = False

    self.optimiser = optim.Adam(self.online_net.parameters(), lr=args.learning_rate, eps=args.adam_eps)

  # Resets noisy weights in all linear layers (of online net only)
  def reset_noise(self):
    self.online_net.reset_noise()

  # Acts based on single state (no batch)
  def act(self, state):
    with torch.no_grad():
      return (self.online_net(state.unsqueeze(0)) * self.support).sum(2).argmax(1).item()

  # Acts with an ε-greedy policy (used for evaluation only)
  def act_e_greedy(self, state, epsilon=0.01):  # High ε can reduce evaluation scores drastically
    return np.random.randint(0, self.action_space) if np.random.random() < epsilon else self.act(state)

  def learn(self, mem):
    # Sample transitions
    idxs, states, actions, returns, next_states, nonterminals, weights = mem.sample(self.batch_size)

    # Calculate current state probabilities (online network noise already sampled)
    log_ps = self.online_net(states, log=True)  # Log probabilities log p(s_t, ·; θonline)
    log_ps_a = log_ps[range(self.batch_size), actions]  # log p(s_t, a_t; θonline)

    with torch.no_grad():
      # Calculate nth next state probabilities
      pns = self.online_net(next_states)  # Probabilities p(s_t+n, ·; θonline)
      dns = self.support.expand_as(pns) * pns  # Distribution d_t+n = (z, p(s_t+n, ·; θonline))
      argmax_indices_ns = dns.sum(2).argmax(1)  # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
      self.target_net.reset_noise()  # Sample new target net noise
      pns = self.target_net(next_states)  # Probabilities p(s_t+n, ·; θtarget)
      pns_a = pns[range(self.batch_size), argmax_indices_ns]  # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)

      # Compute Tz (Bellman operator T applied to z)
      Tz = returns.unsqueeze(1) + nonterminals * (self.discount ** self.n) * self.support.unsqueeze(0)  # Tz = R^n + (γ^n)z (accounting for terminal states)
      Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)  # Clamp between supported values
      # Compute L2 projection of Tz onto fixed support z
      b = (Tz - self.Vmin) / self.delta_z  # b = (Tz - Vmin) / Δz
      l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
      # Fix disappearing probability mass when l = b = u (b is int)
      l[(u > 0) * (l == u)] -= 1
      u[(l < (self.atoms - 1)) * (l == u)] += 1

      # Distribute probability of Tz
      m = states.new_zeros(self.batch_size, self.atoms)
      offset = torch.linspace(0, ((self.batch_size - 1) * self.atoms), self.batch_size).unsqueeze(1).expand(self.batch_size, self.atoms).to(actions)
      m.view(-1).index_add_(0, (l + offset).view(-1), (pns_a * (u.float() - b)).view(-1))  # m_l = m_l + p(s_t+n, a*)(u - b)
      m.view(-1).index_add_(0, (u + offset).view(-1), (pns_a * (b - l.float())).view(-1))  # m_u = m_u + p(s_t+n, a*)(b - l)

    loss = -torch.sum(m * log_ps_a, 1)  # Cross-entropy loss (minimises DKL(m||p(s_t, a_t)))
    self.online_net.zero_grad()
    (weights * loss).mean().backward()  # Backpropagate importance-weighted minibatch loss
    self.optimiser.step()

    mem.update_priorities(idxs, loss.detach().cpu().numpy())  # Update priorities of sampled transitions

  def update_target_net(self):
    self.target_net.load_state_dict(self.online_net.state_dict())

  # Save model parameters on current device (don't move model between devices)
  def save(self, path, name='model.pth'):
    torch.save(self.online_net.state_dict(), os.path.join(path, name))

  # Evaluates Q-value based on single state (no batch)
  def evaluate_q(self, state):
    with torch.no_grad():
      return (self.online_net(state.unsqueeze(0)) * self.support).sum(2).max(1)[0].item()

  def train(self):
    self.online_net.train()

  def eval(self):
    self.online_net.eval()
