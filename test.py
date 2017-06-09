from __future__ import division
import math
import os
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim
from envs import atari_env, setup_logger
from model import A3Clstm
from torch.autograd import Variable
from torchvision import datasets, transforms
import time
import logging


def test(rank, args, shared_model, env_conf):
    log = {}
    setup_logger('{}_log'.format(args.env_name),
                 r'{0}{1}_log'.format(args.log_dir, args.env_name))
    log['{}_log'.format(args.env_name)] = logging.getLogger(
        '{}_log'.format(args.env_name))
    d_args = vars(args)
    for k in d_args.keys():
        log['{}_log'.format(args.env_name)].info(
            '{0}: {1}'.format(k, d_args[k]))

    torch.manual_seed(args.seed)
    env = atari_env(args.env_name, env_conf)
    model = A3Clstm(env.observation_space.shape[0], env.action_space)
    model.eval()

    state = env.reset()
    state = torch.from_numpy(state).float()
    reward_sum = 0
    done = True
    start_time = time.time()
    episode_length = 0
    num_tests = 0
    reward_total_sum = 0
    while True:
        episode_length += 1
        # Sync with the shared model
        if done:
            model.load_state_dict(shared_model.state_dict())
            cx = Variable(torch.zeros(1, 512), volatile=True)
            hx = Variable(torch.zeros(1, 512), volatile=True)
        else:
            cx = Variable(cx.data, volatile=True)
            hx = Variable(hx.data, volatile=True)

        value, logit, (hx, cx) = model((Variable(
            state.unsqueeze(0), volatile=True), (hx, cx)))
        prob = F.softmax(logit)
        action = prob.max(1)[1].data.numpy()
        state, reward, done, _ = env.step(action[0, 0])
        done = done or episode_length >= args.max_episode_length
        reward_sum += reward

        if done:
            num_tests += 1
            reward_total_sum += reward_sum
            reward_mean = reward_total_sum / num_tests
            log['{}_log'.format(args.env_name)].info(
                "Time {0}, episode reward {1}, episode length {2}, reward mean {3:.4f}".
                format(
                    time.strftime("%Hh %Mm %Ss",
                                  time.gmtime(time.time() - start_time)),
                    reward_sum, episode_length, reward_mean))
            if reward_sum > args.save_score_level:
                model.load_state_dict(shared_model.state_dict())
                state_to_save = model.state_dict()
                torch.save(state_to_save, '{0}{1}.dat'.format(
                    args.save_model_dir, args.env_name))

            reward_sum = 0
            episode_length = 0
            state = env.reset()
            time.sleep(60)

        state = torch.from_numpy(state).float()
