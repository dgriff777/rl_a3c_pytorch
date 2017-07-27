from __future__ import division
import torch
from environment import atari_env
from utils import setup_logger
from model import A3Clstm
from player_util import Agent, player_act, player_start
from torch.autograd import Variable
import time
import logging


def test(args, shared_model, env_conf):
    log = {}
    setup_logger('{}_log'.format(args.env), r'{0}{1}_log'.format(
        args.log_dir, args.env))
    log['{}_log'.format(args.env)] = logging.getLogger(
        '{}_log'.format(args.env))
    d_args = vars(args)
    for k in d_args.keys():
        log['{}_log'.format(args.env)].info('{0}: {1}'.format(k, d_args[k]))

    torch.manual_seed(args.seed)
    env = atari_env(args.env, env_conf)
    model = A3Clstm(env.observation_space.shape[0], env.action_space)


    state = env.reset()
    reward_sum = 0
    start_time = time.time()
    num_tests = 0
    reward_total_sum = 0
    player = Agent(model, env, args, state)
    player.state = torch.from_numpy(state).float()
    player.model.eval()
    while True:

        if player.done:
            player.model.load_state_dict(shared_model.state_dict())
            player.cx = Variable(torch.zeros(1, 512), volatile=True)
            player.hx = Variable(torch.zeros(1, 512), volatile=True)
            player = player_start(player, train=False)
        else:
            player.cx = Variable(player.cx.data, volatile=True)
            player.hx = Variable(player.hx.data, volatile=True)

        player, reward = player_act(player, train=False)
        reward_sum += reward

        if not player.done:
            if player.current_life > player.info['ale.lives']:
                player.flag = True
                player.current_life = player.info['ale.lives']
            else:
                player.current_life = player.info['ale.lives']
                player.flag = False

        if player.flag:
            player = player_start(player, train=False)

        if player.done:
            num_tests += 1
            player.current_life = 0
            player.flag = False
            reward_total_sum += reward_sum
            reward_mean = reward_total_sum / num_tests
            log['{}_log'.format(args.env)].info(
                "Time {0}, episode reward {1}, episode length {2}, reward mean {3:.4f}".
                format(
                    time.strftime("%Hh %Mm %Ss",
                                  time.gmtime(time.time() - start_time)),
                    reward_sum, player.eps_len, reward_mean))

            if reward_sum > args.save_score_level:
                player.model.load_state_dict(shared_model.state_dict())
                state_to_save = player.model.state_dict()
                torch.save(state_to_save, '{0}{1}.dat'.format(
                    args.save_model_dir, args.env))

            reward_sum = 0
            player.eps_len = 0
            state = player.env.reset()
            time.sleep(60)
            player.state = torch.from_numpy(state).float()
