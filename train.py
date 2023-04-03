from __future__ import division
import os
os.environ["OMP_NUM_THREADS"] = "1"
from setproctitle import setproctitle as ptitle
import torch
import torch.optim as optim
from environment import atari_env
from utils import ensure_shared_grads
from model import A3Clstm
from player_util import Agent
from torch.autograd import Variable
import time

def train(rank, args, shared_model, optimizer, env_conf):
    ptitle(f"Train Agent: {rank}")
    gpu_id = args.gpu_ids[rank % len(args.gpu_ids)]
    torch.manual_seed(args.seed + rank)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed + rank)
    hidden_size = args.hidden_size
    env = atari_env(args.env, env_conf, args)
    if optimizer is None:
        if args.optimizer == 'RMSprop':
            optimizer = optim.RMSprop(shared_model.parameters(), lr=args.lr)
        if args.optimizer == 'Adam':
            optimizer = optim.Adam(
                shared_model.parameters(), lr=args.lr, amsgrad=args.amsgrad)
    env.seed(args.seed + rank)
    player = Agent(None, env, args, None)
    player.gpu_id = gpu_id
    player.model = A3Clstm(player.env.observation_space.shape[0], player.env.action_space, args)

    player.state = player.env.reset()
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            player.state = torch.from_numpy(player.state).float().cuda()
            player.model = player.model.cuda()
    else:
        player.state = torch.from_numpy(player.state).float()
    player.model.train()
    if len(args.distributed_step_size) > 0:
        num_steps = args.distributed_step_size[rank%len(args.distributed_step_size)]
    else:
        num_steps = args.num_steps
    try:
        while 1:
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.model.load_state_dict(shared_model.state_dict())
            else:
                player.model.load_state_dict(shared_model.state_dict())
            if player.done:
                if gpu_id >= 0:
                    with torch.cuda.device(gpu_id):
                        player.cx = torch.zeros(1, hidden_size).cuda()
                        player.hx = torch.zeros(1, hidden_size).cuda()
                else:
                    player.cx = torch.zeros(1, hidden_size)
                    player.hx = torch.zeros(1, hidden_size)
            else:
                player.cx = player.cx.data
                player.hx = player.hx.data
            for step in range(num_steps):
                player.action_train()

                if player.done:
                    break

            if player.done:
                player.eps_len = 0
                state = player.env.reset()
                if gpu_id >= 0:
                    with torch.cuda.device(gpu_id):
                        player.state = torch.from_numpy(state).float().cuda()
                else:
                    player.state = torch.from_numpy(state).float()

            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    R = torch.zeros(1, 1).cuda()
                    gae = torch.zeros(1, 1).cuda()
            else:
                R = torch.zeros(1, 1)
                gae = torch.zeros(1, 1)
            if not player.done:
                state = player.state
                value, _, _, _ = player.model(
                    state.unsqueeze(0), player.hx, player.cx
                )
                R = value.detach()
            player.values.append(R)
            policy_loss = 0
            value_loss = 0
            for i in reversed(range(len(player.rewards))):
                R = args.gamma * R + player.rewards[i]
                advantage = R - player.values[i]
                value_loss = value_loss + 0.5 * advantage.pow(2)

                # Generalized Advantage Estimataion
                delta_t = (
                    player.rewards[i]
                    + args.gamma * player.values[i + 1].data
                    - player.values[i].data
                )

                gae = gae * args.gamma * args.tau + delta_t
                policy_loss = (
                    policy_loss
                    - (player.log_probs[i] * gae)
                    - (args.entropy_coef * player.entropies[i])
                )

            player.model.zero_grad()
            (policy_loss + 0.5 * value_loss).backward()
            ensure_shared_grads(player.model, shared_model, gpu=gpu_id >= 0)
            optimizer.step()
            player.clear_actions()
    except KeyboardInterrupt:
        time.sleep(0.01)
        print("KeyboardInterrupt exception is caught")
    finally:
        print(f"train agent {rank} process finished")
