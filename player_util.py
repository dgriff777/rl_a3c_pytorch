from __future__ import division
import torch
import torch.nn.functional as F
from torch.autograd import Variable


class Agent(object):
    def __init__(self, model, env, args, state):
        self.model = model
        self.env = env
        self.current_life = 0
        self.state = state
        self.hx = None
        self.cx = None
        self.eps_len = 0
        self.args = args
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.done = True
        self.flag = False
        self.info = None
        self.starter = False or args.env[:8] == 'Breakout'


def player_act(player, train):
    if train:
        value, logit, (player.hx, player.cx) = player.model(
            (Variable(player.state.unsqueeze(0)), (player.hx, player.cx)))
    else:
        value, logit, (player.hx, player.cx) = player.model((Variable(
            player.state.unsqueeze(0), volatile=True), (player.hx, player.cx)))
        prob = F.softmax(logit)
        action = prob.max(1)[1].data.numpy()
        state, reward, player.done, player.info = player.env.step(action[0])
        player.state = torch.from_numpy(state).float()
        player.eps_len += 1
        player.done = player.done or player.eps_len >= player.args.max_episode_length
        return player, reward
    prob = F.softmax(logit)
    log_prob = F.log_softmax(logit)
    entropy = -(log_prob * prob).sum(1)
    player.entropies.append(entropy)
    action = prob.multinomial().data
    log_prob = log_prob.gather(1, Variable(action))
    state, reward, player.done, player.info = player.env.step(action.numpy())
    player.state = torch.from_numpy(state).float()
    player.eps_len += 1
    player.done = player.done or player.eps_len >= player.args.max_episode_length
    reward = max(min(reward, 1), -1)
    player.values.append(value)
    player.log_probs.append(log_prob)
    player.rewards.append(reward)
    return player


def player_start(player, train):
    for i in range(3):
	player.flag = False
        if train:
            value, logit, (player.hx, player.cx) = player.model(
                (Variable(player.state.unsqueeze(0)), (player.hx, player.cx)))
        else:
            value, logit, (player.hx, player.cx) = player.model((Variable(
                player.state.unsqueeze(0), volatile=True), (player.hx,
                                                            player.cx)))
        prob = F.softmax(logit)
        log_prob = F.log_softmax(logit)
        entropy = -(log_prob * prob).sum(1)
        player.entropies.append(entropy)
        action = prob.multinomial().data
        log_prob = log_prob.gather(1, Variable(action))
        state, reward, player.done, player.info = player.env.step(1)
        player.state = torch.from_numpy(state).float()
        player.eps_len += 1
        player.done = player.done or player.eps_len >= player.args.max_episode_length
        if train:
            reward = max(min(reward, 1), -1)
            player.values.append(value)
            player.log_probs.append(log_prob)
            player.rewards.append(reward)
        if player.done:
            return player
    return player
