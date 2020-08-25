import gym
from PIL import Image, ImageDraw, ImageFont

from elements.envs import AbstractEnv

def get_env(name, seed, **kwargs):
    env = CartpoleEnv(kwargs['abst_type'], kwargs['vis_type'])
    return env

class CartpoleEnv(AbstractEnv):
    def __init__(self, abst_type, vis_type):
        self.abst_type = abst_type
        self.vis_type = vis_type

        self.env = gym.make('CartPole-v0')
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        super().__init__(do_nothing=None, actions=list(range(self.action_space.n)))

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()

    def abst(self, state):
        return cartpole_abs_func(self.abst_type)(state)

    def get_RGB(self, env, state, action, mut, score):
        return cartpole_get_frame_proc(self.vis_type)(env, state, action, mut, score)

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        self.env.close()

### Abstraction ###
def cartpole_abs_func(n=-1):
    """
            Num	Observation                 Min         Max
        0	Cart Position             -4.8            4.8
        1	Cart Velocity             -Inf            Inf
        2	Pole Angle                 -24 deg        24 deg
        3	Pole Velocity At Tip      -Inf            Inf
    """
    if n == 0:
        return lambda state: str([
            round(state[0]),
            round(state[1], 1),
            round(state[2], 2),
            round(state[3], 1)])
    elif n == 1:
        return lambda state: str(round(state[2], 2))
    else:
        return lambda state: str([
            abs(round(state[0])),
            abs(round(state[1], 1)),
            abs(round(state[2]/4, 2)),
            abs(round(state[3], 1))])

### Visualisation ###
def cartpole_get_RGB(env, state, action, mut):
    frame = env.render(mode='rgb_array')
    frame = Image.fromarray(frame)
    return frame

# Old visualisation code
# def cartpole_get_RGB_mut(env, state, action, mut):
#     frame = cartpole_get_RGB(env, state, action, mut)
#     draw = ImageDraw.Draw(frame)
#     txt = 'Mut' if mut else 'Normal'
#     draw.text((200, 200), txt, fill='rgb(0,0,0)')
#     if action is not None:
#         act = '<--' if action == 0 else '-->'
#     else:
#         act = ''
#     draw.text((200, 220), act, fill='rgb(0,0,0)')
#     return frame

def arrow_coords(x, y, l, w, drct):
    if drct == "left":
        x = x + (0.5*l)
    else:
        x = x - (0.5*l)
    coords = [(x, y+(0.3*w))]
    coords.append((x, y+(0.7*w)))
    if drct == "left":
        l = -l
    coords.append((x+(0.7*l), y+(0.7*w)))
    coords.append((x+(0.7*l), y+w))
    coords.append((x+l, y+(0.5*w)))
    coords.append((x+(0.7*l), y))
    coords.append((x+(0.7*l), y+(0.3*w)))
    return coords

def cartpole_get_RGB_mut(env, state, action, mut, score, do_scale=False, crop=True):
    frame = cartpole_get_RGB(env, state, action, mut)
    draw = ImageDraw.Draw(frame)
    drct = "right" if action == 1 else "left"
    col = 'rgb(255,0,0)' if not mut else 'rgb(0,0,255)'
    if score != "Not scored" and action is not None:
        if do_scale:
            score = float(score)
            l, w = 100*score, 40*score
        else:
            l, w = 40, 15
        draw.polygon(arrow_coords(215, 220, l, w, drct), fill=col)
    frame = frame.crop((150, 75, 450, 350)) if crop else frame
    return frame

font = ImageFont.truetype("arial.ttf", 16)
def cartpole_get_RGB_scores(env, state, action, mut, score):
    frame = cartpole_get_RGB_mut(env, state, action, mut, score, True, False)
    draw = ImageDraw.Draw(frame)
    if isinstance(score, float):
        draw.text((200, 200), "{0:.2f}".format(score), fill='rgb(0,0,0)', font=font)
    else:
        draw.text((200, 200), str(score), fill='rgb(0,0,0)', font=font)
    frame = frame.crop((150, 75, 450, 350))
    return frame

def cartpole_get_frame_proc(n=-1):
    funcs = [cartpole_get_RGB, cartpole_get_RGB_mut, cartpole_get_RGB_scores]
    return funcs[n+1]
