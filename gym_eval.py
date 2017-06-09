from __future__ import division
import math
import argparse
import os
import sys
import torch
import torch.nn.functional as F
from envs import atari_env, read_config, setup_logger
from model import A3Clstm, normalized_columns_initializer
from torch.autograd import Variable
from torchvision import datasets, transforms
import time
import gym
import logging


parser = argparse.ArgumentParser(description='A3C_EVAL')
parser.add_argument('--env-name', default='Pong-v0', metavar='ENV',
                    help='environment to train on (default: Pong-v0)')
parser.add_argument('--env-config', default='config.json', metavar='EC',
                    help='environment to crop and resize info (default: config.json)')
parser.add_argument('--num-episodes', type=int, default=100, metavar='NE',
                    help='how many episodes in evaluation (default: 100)')
parser.add_argument('--load-model-dir', default='trained_models/', metavar='LMD',
                    help='folder to load trained models from')
parser.add_argument('--log-dir', default='logs/', metavar='LG',
                    help='folder to save logs')
parser.add_argument('--render', default=False, metavar='R',
                    help='Watch game as it being played')
parser.add_argument('--render-freq', type=int, default=1, metavar='RF',
                    help='Frequency to watch rendered game play')
parser.add_argument('--max-episode-length', type=int, default=100000, metavar='M',
                    help='maximum length of an episode (default: 100000)')
args = parser.parse_args()

setup_json = read_config(args.env_config)
env_conf = setup_json["Default"]
for i in setup_json.keys():
    if i in args.env_name:
        env_conf = setup_json[i]
torch.set_default_tensor_type('torch.FloatTensor')

saved_state = torch.load('{0}{1}.dat'.format(args.load_model_dir, args.env_name),
                         map_location=lambda storage, loc: storage)

done = True

log = {}
setup_logger('{}_mon_log'.format(args.env_name),
             r'{0}{1}_mon_log'.format(args.log_dir, args.env_name))
log['{}_mon_log'.format(args.env_name)] = logging.getLogger(
    '{}_mon_log'.format(args.env_name))


env = atari_env("{}".format(args.env_name), env_conf)
model = A3Clstm(env.observation_space.shape[0], env.action_space)
model.eval()

env = gym.wrappers.Monitor(env, "{}_monitor".format(args.env_name), force=True)
num_tests = 0
reward_total_sum = 0
for i_episode in range(args.num_episodes):
    state = env.reset()
    episode_length = 0
    reward_sum = 0
    while True:
        if args.render:
            if i_episode % args.render_freq == 0:
                env.render()
        if done:
            model.load_state_dict(saved_state)
            cx = Variable(torch.zeros(1, 512), volatile=True)
            hx = Variable(torch.zeros(1, 512), volatile=True)
        else:
            cx = Variable(cx.data, volatile=True)
            hx = Variable(hx.data, volatile=True)

        state = torch.from_numpy(state).float()
        value, logit, (hx, cx) = model(
            (Variable(state.unsqueeze(0), volatile=True), (hx, cx)))
        prob = F.softmax(logit)
        action = prob.max(1)[1].data.numpy()
        state, reward, done, _ = env.step(action[0, 0])
        episode_length += 1
        reward_sum += reward
        done = done or episode_length >= args.max_episode_length
        if done:
            num_tests += 1
            reward_total_sum += reward_sum
            reward_mean = reward_total_sum / num_tests
            log['{}_mon_log'.format(args.env_name)].info(
                "reward sum: {0}, reward mean: {1:.4f}".format(reward_sum, reward_mean))

            break
