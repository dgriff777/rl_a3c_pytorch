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
        self.gpu_id = -1
        self.max_length = False


    def action_train(self):
        if self.done:
            if self.gpu_id >= 0:
                with torch.cuda.device(self.gpu_id):
                    self.cx = Variable(torch.zeros(1, 512).cuda())
                    self.hx = Variable(torch.zeros(1, 512).cuda())
            else:
                self.cx = Variable(torch.zeros(1, 512))
                self.hx = Variable(torch.zeros(1, 512))
        else:
            self.cx = Variable(self.cx.data)
            self.hx = Variable(self.hx.data)
        value, logit, (self.hx, self.cx) = self.model(
            (Variable(self.state.unsqueeze(0)), (self.hx, self.cx)))
        prob = F.softmax(logit, dim=1)
        log_prob = F.log_softmax(logit, dim=1)
        entropy = -(log_prob * prob).sum(1)
        self.entropies.append(entropy)
        action = prob.multinomial().data
        log_prob = log_prob.gather(1, Variable(action))
        state, self.reward, self.done, self.info = self.env.step(
            action.cpu().numpy())
        self.state = torch.from_numpy(state).float()
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.state = self.state.cuda()
        self.eps_len += 1
        if self.eps_len >= self.args.max_episode_length:  #ugly hack need to clean this up
           if not self.done:
               self.max_length = True
               self.done = True
           else:
               self.max_length = False
        else:
           self.max_length = False
        self.reward = max(min(self.reward, 1), -1)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.rewards.append(self.reward)
        return self

    def action_test(self):
        if self.done:
            if self.gpu_id >= 0:
                with torch.cuda.device(self.gpu_id):
                    self.cx = Variable(torch.zeros(
                        1, 512).cuda(), volatile=True)
                    self.hx = Variable(torch.zeros(
                        1, 512).cuda(), volatile=True)
            else:
                self.cx = Variable(torch.zeros(1, 512), volatile=True)
                self.hx = Variable(torch.zeros(1, 512), volatile=True)
        else:
            self.cx = Variable(self.cx.data, volatile=True)
            self.hx = Variable(self.hx.data, volatile=True)
        value, logit, (self.hx, self.cx) = self.model(
            (Variable(self.state.unsqueeze(0), volatile=True), (self.hx, self.cx)))
        prob = F.softmax(logit, dim=1)
        action = prob.max(1)[1].data.cpu().numpy()
        state, self.reward, self.done, self.info = self.env.step(action[0])
        self.state = torch.from_numpy(state).float()
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.state = self.state.cuda()
        self.eps_len += 1
        if self.eps_len >= self.args.max_episode_length:  #ugly hack need to clean this up
           if not self.done:
               self.max_length = True
               self.done = True 
           else:
               self.max_length = False
        else:
           self.max_length = False
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

