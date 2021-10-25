import numpy as np
import cv2
import gym
from PIL import Image, ImageDraw, ImageFont

from elements.envs import AbstractEnv
from environments.uber_gym.dopamine_preprocessing import AtariPreprocessing as DopamineAtariPreprocessing
from environments.uber_gym.preprocessing import FrameStack, ScaledFloatFrame, FireResetEnv, MaxStepsEnv

def get_env(name, seed, **kwargs):
    name += "NoFrameskip-v4"
    env = GymEnv(name, seed, kwargs['device'], kwargs['abst_type'], kwargs['vis_type'], kwargs['max_steps'])
    return env

class GymEnv(AbstractEnv):
    def __init__(self, name, seed, device, abst_type, vis_type, max_steps):
        self.name = name
        self.abst_type = abst_type
        self.vis_type = vis_type
        self.max_steps = max_steps if max_steps > 0 else 100000

        dual_state = True
        env = gym.make(name)
        if hasattr(env,'unwrapped'):
            env = env.unwrapped
        env = DopamineAtariPreprocessing(env, dual_state)
        env = FrameStack(env, 4, dual_state)
        env = ScaledFloatFrame(env, dual_state, scale=1.0/255.0)
        env = FireResetEnv(env)
        env = MaxStepsEnv(env, self.max_steps)
        self.env = env
        super().__init__(do_nothing=0, actions=list(range(self.env.action_space.n)))

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()

    def abst(self, state, ret_str=True):
        return gym_abs_func(self.abst_type)(state, self.name, ret_str=ret_str)

    def get_RGB(self, env, state, action, mut, score):
        return gym_get_frame_proc(self.vis_type)(env, state, action, mut, score)

    def action_space(self):
        return self.env.action_space.n
    
    def observation_space(self):
        return self.env.observation_space.shape

    def close(self):
        self.env.close()

### Utils ###
def gym_find(rgb_state, color, off_top, off_bot, off_side):
    """ Return the items's top-left coordinate or None if there is none.
    Coordinate in terms of array index, need to swap for cv2.
    Only uses R of RGB """
    area = rgb_state[off_top:-off_bot, off_side:-off_side, 0]
    try:
        loc = next(zip(*np.where(area == color[0])))
    except StopIteration:
        return (-1, -1)
    else:
        return (loc[0] + off_top, loc[1] + off_side)

### General Abstraction ###
def get_RL_view(state):
    """ Env returns tuple (stacked states, game RGB)
    The former is fed to the network. """
    return state[0].copy() if len(state) == 2 else state.copy()

def gym_get_RGB(state):
    return state[1].copy() if len(state) == 2 else state.copy()

def RL_abs(state, name, ret_str=True):
    """ Just isolate state and maybe turn into string """
    state = get_RL_view(state)
    state = state.cpu().numpy()
    if ret_str:
        return np.array2string(state, precision=0, threshold=10000)
    else:
        return state

def basic_abs(state, name, ret_str=True):
    """ Generic gray-scale, down-sample, and change pixel intensity precision.
    Similar to go-explore abstraction."""
    state = gym_get_RGB(state)

    size = (18, 14)
    max_pix = 8
    gray = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    round_pix = ((small / 255.0) * max_pix).astype(np.uint8)
    return str(round_pix) if ret_str else round_pix

### Breakout Abstractions ####
ball_color = [200, 72, 72]
paddle_color = [200, 72, 72]

def brkt_get_ball(rgb_state):
    off_top, off_bot, off_side = 93, 21, 8
    return gym_find(rgb_state, ball_color, off_top, off_bot, off_side)

def brkt_get_paddle(rgb_state):
    off_top, off_bot, off_side = 189, 17, 8
    return gym_find(rgb_state, paddle_color, off_top, off_bot, off_side)[1]

def brkt_abs(state, name, ret_str=True):
    """ basic_abs for breakout but make ball bigger so that it doesn't disappear """
    state = gym_get_RGB(state)
    loc = brkt_get_ball(state)
    if loc != (-1, -1):
        loc = (loc[1], loc[0])
        cv2.rectangle(state, tuple(loc), (loc[0] + 10, loc[1] + 10), ball_color, -1)

    abs_state = basic_abs(state, name, ret_str=ret_str)
    return abs_state

def brkt_poss_abs(state, name, ret_str=True):
    """ State is ball and paddle location with some precision """
    state = gym_get_RGB(state)
    ball_loc = brkt_get_ball(state)
    pad_loc = brkt_get_paddle(state)

    ball_loc = None if ball_loc is None else (round(ball_loc[0], -1), round(ball_loc[1], -1))
    pad_loc = None if pad_loc is None else round(pad_loc, -1)

    if ret_str:
        return "Ball: " + str(ball_loc) + " Paddle: " + str(pad_loc)
    else:
        return pad_loc, ball_loc

### Pong abstraction ###
pong_ball_color = [236, 236, 236]
pong_paddle_color = [92, 186, 92]

def pong_get_ball(rgb_state):
    off_top, off_bot, off_side = 34, 16, 16
    return gym_find(rgb_state, pong_ball_color, off_top, off_bot, off_side)

def pong_get_paddle(rgb_state):
    off_top, off_bot, off_side = 34, 16, 16
    return gym_find(rgb_state, pong_paddle_color, off_top, off_bot, off_side)[0]

def pong_poss_abs(state, name, ret_str=True):
    """ State is ball and player's paddle location with some precision """
    state = gym_get_RGB(state)
    ball_loc = pong_get_ball(state)
    pad_loc = pong_get_paddle(state)

    ball_loc = None if ball_loc is None else (round(ball_loc[0], -1), round(ball_loc[1], -1))
    pad_loc = None if pad_loc is None else round(pad_loc, -1)

    if ret_str:
        return "Ball: " + str(ball_loc) + " Paddle: " + str(pad_loc)
    else:
        return pad_loc, ball_loc

### Visualisation ###
def gym_just_RGB(env, state, action, mut, score):
    return gym_get_RGB(state)

def gym_get_RGB_abs(env, state, action, mut, score):
    # Some gym abstractions don't return frames, and so this won't work
    frame = env.abst(state, ret_str=False)
    mx = np.max(frame)
    mn = np.min(frame)
    frame = frame - mn
    frame = frame * (255/mx)
    frame = frame.astype('uint8')
    return frame

def gym_get_RGB_mut(env, state, action, mut, score):
    frame = gym_just_RGB(env, state, action, mut, score)
    frame = Image.fromarray(frame)
    draw = ImageDraw.Draw(frame)
    col = 'rgb(255,0,0)' if not mut else 'rgb(0,0,255)'
    draw.rectangle([(0,0), (10,10)], fill=col)
    return frame

### Selectors ###
def gym_abs_func(n=-1):
    funcs = [basic_abs, RL_abs, brkt_abs, brkt_poss_abs, pong_poss_abs]
    return funcs[n+1]

def gym_get_frame_proc(n=-1):
    funcs = [gym_just_RGB, gym_get_RGB_abs, gym_get_RGB_mut]
    return funcs[n+1]
