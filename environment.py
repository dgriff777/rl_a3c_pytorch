from __future__ import division
import gym
import numpy as np
from gym.spaces.box import Box
from skimage.color import rgb2gray
from cv2 import resize
#from skimage.transform import resize
#from scipy.misc import imresize as resize


def atari_env(env_id, env_conf):
    env = gym.make(env_id)
    env = AtariRescale(env, env_conf)
    env = NormalizedEnv(env)
    return env


def _process_frame(frame, conf):
    frame = frame[conf["crop1"]:conf["crop2"] + 160, :160]
    frame = resize(rgb2gray(frame), (80, conf["dimension2"]))
    frame = resize(frame, (80, 80))
    frame = np.float32(frame)
    frame = np.reshape(frame, [1, 80, 80])
    return frame


class AtariRescale(gym.ObservationWrapper):
    def __init__(self, env, env_conf):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = Box(0.0, 1.0, [1, 80, 80])
        self.conf = env_conf

    def _observation(self, observation):
        return _process_frame(observation, self.conf)


class NormalizedEnv(gym.ObservationWrapper):
    def __init__(self, env=None):
        gym.ObservationWrapper.__init__(self, env)
        self.state_mean = 0
        self.state_std = 0
        self.alpha = 0.9999
        self.num_steps = 0

    def _observation(self, observation):
        self.num_steps += 1
        self.state_mean = self.state_mean * self.alpha + \
            observation.mean() * (1 - self.alpha)
        self.state_std = self.state_std * self.alpha + \
            observation.std() * (1 - self.alpha)

        unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
        unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))

        return (observation - unbiased_mean) / (unbiased_std + 1e-8)
