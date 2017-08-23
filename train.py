from __future__ import division
import torch
import torch.optim as optim
from environment import atari_env
from utils import ensure_shared_grads
from model import A3Clstm
from player_util import Agent
from torch.autograd import Variable


def train(rank, args, shared_model, optimizer, env_conf):

    torch.manual_seed(args.seed + rank)
    env = atari_env(args.env, env_conf)
    if optimizer is None:
        if args.optimizer == 'RMSprop':
            optimizer = optim.RMSprop(shared_model.parameters(), lr=args.lr)
        if args.optimizer == 'Adam':
            optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)

    env.seed(args.seed + rank)
    player = Agent(None, env, args, None)
    player.model = A3Clstm(
        player.env.observation_space.shape[0], player.env.action_space)
    player.state = player.env.reset()
    player.state = torch.from_numpy(player.state).float()
    player.model.train()

    while True:
        player.model.load_state_dict(shared_model.state_dict())
        for step in range(args.num_steps):
            player.action_train()
            if args.count_lives:
                player.check_state()
            if player.done:
                break

        if player.done:
            player.eps_len = 0
            player.current_life = 0
            state = player.env.reset()
            player.state = torch.from_numpy(state).float()

        R = torch.zeros(1, 1)
        if not player.done:
            value, _, _ = player.model(
                (Variable(player.state.unsqueeze(0)), (player.hx, player.cx)))
            R = value.data

        player.values.append(Variable(R))
        policy_loss = 0
        value_loss = 0
        R = Variable(R)
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(player.rewards))):
            R = args.gamma * R + player.rewards[i]
            advantage = R - player.values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimataion
            delta_t = player.rewards[i] + args.gamma * \
                player.values[i + 1].data - player.values[i].data
            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = policy_loss - \
                player.log_probs[i] * \
                Variable(gae) - 0.01 * player.entropies[i]

        optimizer.zero_grad()
        (policy_loss + 0.5 * value_loss).backward()
        torch.nn.utils.clip_grad_norm(player.model.parameters(), 40)
        ensure_shared_grads(player.model, shared_model)
        optimizer.step()
        player.clear_actions()
