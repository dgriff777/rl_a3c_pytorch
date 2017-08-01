from __future__ import division
import gym
import numpy as np
from gym.spaces.box import Box
from universe import vectorized
from universe.wrappers import Unvectorize, Vectorize
from skimage.color import rgb2gray
from scipy.misc import imresize
#import cv2


def atari_env(env_id, env_conf):
    env = gym.make(env_id)
    if len(env.observation_space.shape) > 1:
        env = Vectorize(env)
        env = AtariRescale(env, env_conf)
        env = NormalizedEnv(env)
        env = Unvectorize(env)
    return env


def _process_frame(frame, conf):
    frame = frame[conf["crop1"]:conf["crop2"] + 160, :160]
    #frame = cv2.resize(rgb2gray(frame), (80, conf["dimension2"]))
    #frame = cv2.resize(frame, (80, 80))
    frame = imresize(rgb2gray(frame), (80, conf["dimension2"]))
    frame = imresize(frame, (80, 80))
    frame = np.reshape(frame, [1, 80, 80])
    return frame


class AtariRescale(vectorized.ObservationWrapper):
    def __init__(self, env, env_conf):
        super(AtariRescale, self).__init__(env)
        self.observation_space = Box(0.0, 1.0, [1, 80, 80])
        self.conf = env_conf

    def _observation(self, observation_n):
        return [
            _process_frame(observation, self.conf)
            for observation in observation_n
        ]


class NormalizedEnv(vectorized.ObservationWrapper):
    def __init__(self, env=None):
        super(NormalizedEnv, self).__init__(env)
        self.state_mean = 0
        self.state_std = 0
        self.alpha = 0.9999
        self.num_steps = 0

    def _observation(self, observation_n):
        for observation in observation_n:
            self.num_steps += 1
            self.state_mean = self.state_mean * self.alpha + \
                observation.mean() * (1 - self.alpha)
            self.state_std = self.state_std * self.alpha + \
                observation.std() * (1 - self.alpha)

        unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
        unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))

        return [(observation - unbiased_mean) / (unbiased_std + 1e-8)
                for observation in observation_n]
