import random

import tensorflow as tf
from atari_zoo import MakeAtariModel

from elements.policies import AbstractPol

def get_pol(name, env, device, **kwargs):
    name += "NoFrameskip-v4"

    m = MakeAtariModel('rainbow', name, 1)()
    m.load_graphdef()
    nA = env.action_space()
    obs_shape = list(env.observation_space())
    return RainbowAgent(m, nA, obs_shape, name)

class RainbowAgent(AbstractPol):
    def __init__(self, pol, nA, obs_shape, name):
        super().__init__(pol)
        self.nA = nA
        config = tf.ConfigProto(
                device_count = {'GPU': 1}
            )
        config.gpu_options.allow_growth=True
        self.sess = tf.Session(config=config)

        self.X_t = tf.placeholder(tf.float32, [None] + obs_shape)
        T = import_model(self.pol, self.X_t, self.X_t, scope=name)
        self.action_sample = self.pol.get_action(T)

    def __call__(self, states, actions, rews):
        if random.random() < 0.01:
            act = random.randint(0,self.nA-1)
        else:
            train_dict = {self.X_t:states[-1][0][None]}
            results = self.sess.run([self.action_sample], feed_dict=train_dict)
            act = results[0]
        return act

def import_model(model, t_image, t_image_raw, scope="import"):
    """ Taken from lucid.optvis.render and modified to 
    use a custom scope. This is useful if you want to 
    load a new model after having already loaded one.
    """
    model.import_graph(t_image, scope=scope, forget_xy_shape=True)

    def T(layer):
        if layer == "input": return t_image_raw
        if layer == "labels": return model.labels
        return t_image.graph.get_tensor_by_name("{}/{}:0".format(scope, layer))

    return T
