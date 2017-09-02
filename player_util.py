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
        self.info = None
        self.reward = 0

    def action_train(self):
        if self.done:
            self.cx = Variable(torch.zeros(1, 512))
            self.hx = Variable(torch.zeros(1, 512))
        else:
            self.cx = Variable(self.cx.data)
            self.hx = Variable(self.hx.data)
        value, logit, (self.hx, self.cx) = self.model((Variable(self.state.unsqueeze(0)), (self.hx, self.cx)))
        prob = F.softmax(logit)
        log_prob = F.log_softmax(logit)
        entropy = -(log_prob * prob).sum(1)
        self.entropies.append(entropy)
        action = prob.multinomial().data
        log_prob = log_prob.gather(1, Variable(action))
        state, self.reward, self.done, self.info = self.env.step(action.numpy())
        self.state = torch.from_numpy(state).float()
        self.eps_len += 1
        self.done = self.done or self.eps_len >= self.args.max_episode_length
        self.reward = max(min(self.reward, 1), -1)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.rewards.append(self.reward)
        return self

    def action_test(self):
        if self.done:
            self.cx = Variable(torch.zeros(1, 512), volatile=True)
            self.hx = Variable(torch.zeros(1, 512), volatile=True)
        else:
            self.cx = Variable(self.cx.data, volatile=True)
            self.hx = Variable(self.hx.data, volatile=True)
        value, logit, (self.hx, self.cx) = self.model((Variable(self.state.unsqueeze(0), volatile=True), (self.hx, self.cx)))
        prob = F.softmax(logit)
        action = prob.max(1)[1].data.numpy()
        state, self.reward, self.done, self.info = self.env.step(action[0])
        self.state = torch.from_numpy(state).float()
        self.eps_len += 1
        self.done = self.done or self.eps_len >= self.args.max_episode_length
        return self

    def check_state(self):
        if self.current_life > self.info['ale.lives']:
            self.done = True
        self.current_life = self.info['ale.lives']
        return self

    def clear_actions(self):
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        return self
