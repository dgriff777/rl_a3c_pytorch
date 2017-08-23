from __future__ import division
import os
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import torch
from environment import atari_env
from utils import read_config, setup_logger
from model import A3Clstm
from player_util import Agent
from torch.autograd import Variable
import gym
import logging
from gym.configuration import undo_logger_setup

undo_logger_setup()

parser = argparse.ArgumentParser(description='A3C_EVAL')
parser.add_argument(
    '--env',
    default='Pong-v0',
    metavar='ENV',
    help='environment to train on (default: Pong-v0)')
parser.add_argument(
    '--env-config',
    default='config.json',
    metavar='EC',
    help='environment to crop and resize info (default: config.json)')
parser.add_argument(
    '--num-episodes',
    type=int,
    default=100,
    metavar='NE',
    help='how many episodes in evaluation (default: 100)')
parser.add_argument(
    '--load-model-dir',
    default='trained_models/',
    metavar='LMD',
    help='folder to load trained models from')
parser.add_argument(
    '--log-dir', default='logs/', metavar='LG', help='folder to save logs')
parser.add_argument(
    '--render',
    default=False,
    metavar='R',
    help='Watch game as it being played')
parser.add_argument(
    '--render-freq',
    type=int,
    default=1,
    metavar='RF',
    help='Frequency to watch rendered game play')
parser.add_argument(
    '--max-episode-length',
    type=int,
    default=100000,
    metavar='M',
    help='maximum length of an episode (default: 100000)')
args = parser.parse_args()

setup_json = read_config(args.env_config)
env_conf = setup_json["Default"]
for i in setup_json.keys():
    if i in args.env:
        env_conf = setup_json[i]
torch.set_default_tensor_type('torch.FloatTensor')

saved_state = torch.load(
    '{0}{1}.dat'.format(args.load_model_dir, args.env),
    map_location=lambda storage, loc: storage)

log = {}
setup_logger('{}_mon_log'.format(args.env), r'{0}{1}_mon_log'.format(
    args.log_dir, args.env))
log['{}_mon_log'.format(args.env)] = logging.getLogger(
    '{}_mon_log'.format(args.env))

d_args = vars(args)
for k in d_args.keys():
    log['{}_mon_log'.format(args.env)].info('{0}: {1}'.format(k, d_args[k]))

env = atari_env("{}".format(args.env), env_conf)
num_tests = 0
reward_total_sum = 0
player = Agent(None, env, args, None)
player.model = A3Clstm(player.env.observation_space.shape[0],
                       player.env.action_space)
player.env = gym.wrappers.Monitor(
    player.env, "{}_monitor".format(args.env), force=True)
player.model.eval()

for i_episode in range(args.num_episodes):
    player.state = player.env.reset()
    player.state = torch.from_numpy(player.state).float()
    player.eps_len = 0
    reward_sum = 0
    while True:
        if args.render:
            if i_episode % args.render_freq == 0:
                player.env.render()

        if player.done:
            player.model.load_state_dict(saved_state)

        player.action_test()
        reward_sum += player.reward

        if player.done:
            player.current_life = 0
            num_tests += 1
            reward_total_sum += reward_sum
            reward_mean = reward_total_sum / num_tests
            log['{}_mon_log'.format(args.env)].info(
                "reward sum: {0}, reward mean: {1:.4f}".format(
                    reward_sum, reward_mean))

            break
