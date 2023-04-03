from __future__ import division
import os
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import torch
from environment import atari_env
from utils import read_config, setup_logger
from model import A3Clstm
from player_util import Agent
import gym
import logging
import time

gym.logger.set_level(40)

parser = argparse.ArgumentParser(description="A3C_EVAL")
parser.add_argument(
    "-ev",
    "--env",
    default="PongNoFrameskip-v4",
    help="environment to train on (default: PongNoFrameskip-v4)",
)
parser.add_argument(
    "-evc", "--env-config",
    default="config.json",
    help="environment to crop and resize info (default: config.json)")
parser.add_argument(
    "-ne",
    "--num-episodes",
    type=int,
    default=100,
    help="how many episodes in evaluation (default: 100)",
)
parser.add_argument(
    "-lmd",
    "--load-model-dir",
    default="trained_models/",
    help="folder to load trained models from",
)
parser.add_argument("-lgd", "--log-dir", default="logs/", help="folder to save logs")
parser.add_argument(
    "-r", "--render", action="store_true", help="Watch game as it being played"
)
parser.add_argument(
    "-rf",
    "--render-freq",
    type=int,
    default=1,
    help="Frequency to watch rendered game play",
)
parser.add_argument(
    "-mel",
    "--max-episode-length",
    type=int,
    default=10000,
    help="maximum length of an episode (default: 100000)",
)
parser.add_argument(
    "-nge",
    "--new-gym-eval",
    action="store_true",
    help="Create a gym evaluation for upload",
)
parser.add_argument(
    "-s", "--seed", type=int, default=1, help="random seed (default: 1)"
)
parser.add_argument(
    "-gid",
    "--gpu-id",
    type=int,
    default=-1,
    help="GPU to use [-1 CPU only] (default: -1)",
)
parser.add_argument(
    "-hs",
    "--hidden-size",
    type=int,
    default=512,
    help="LSTM Cell number of features in the hidden state h",
)
parser.add_argument(
    "-sk", "--skip-rate",
    type=int,
    default=4,
    help="frame skip rate (default: 4)")
args = parser.parse_args()

setup_json = read_config(args.env_config)
env_conf = setup_json["Default"]
for i in setup_json.keys():
    if i in args.env:
        env_conf = setup_json[i]

saved_state = torch.load(
    f"{args.load_model_dir}{args.env}.dat", map_location=lambda storage, loc: storage
)


setup_logger(f"{args.env}_mon_log", rf"{args.log_dir}{args.env}_mon_log")
log = logging.getLogger(f"{args.env}_mon_log")

gpu_id = args.gpu_id

torch.manual_seed(args.seed)
if gpu_id >= 0:
    torch.cuda.manual_seed(args.seed)


d_args = vars(args)
for k in d_args.keys():
    log.info(f"{k}: {d_args[k]}")

env = atari_env(f"{args.env}", env_conf, args)
num_tests = 0
start_time = time.time()
reward_total_sum = 0
player = Agent(None, env, args, None)
player.model = A3Clstm(player.env.observation_space.shape[0], player.env.action_space, args)
player.gpu_id = gpu_id
if gpu_id >= 0:
    with torch.cuda.device(gpu_id):
        player.model = player.model.cuda()
if args.new_gym_eval:
    player.env = gym.wrappers.Monitor(
        player.env, f"{args.env}_monitor", force=True)

if gpu_id >= 0:
    with torch.cuda.device(gpu_id):
        player.model.load_state_dict(saved_state)
else:
    player.model.load_state_dict(saved_state)

player.model.eval()
try:
    for i_episode in range(args.num_episodes):
        player.state = player.env.reset()
        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                player.state = torch.from_numpy(player.state).float().cuda()
        else:
            player.state = torch.from_numpy(player.state).float()
        player.eps_len = 0
        reward_sum = 0
        while 1:
            if args.render:
                if i_episode % args.render_freq == 0:
                    player.env.render()
            player.action_test()
            reward_sum += player.reward
            if player.done and not player.env.was_real_done:
                state = player.env.reset()
                player.state = torch.from_numpy(state).float()
                if gpu_id >= 0:
                    with torch.cuda.device(gpu_id):
                         player.state = player.state.cuda()
            elif player.env.was_real_done:
                num_tests += 1
                reward_total_sum += reward_sum
                reward_mean = reward_total_sum / num_tests
                log.info(
                    f"Time {time.strftime('%Hh %Mm %Ss', time.gmtime(time.time() - start_time))}, episode reward {reward_sum}, episode length {player.eps_len}, reward mean {reward_mean:.4f}"
                )
                break
except KeyboardInterrupt:
    print("KeyboardInterrupt exception is caught")
finally:
    print("gym evalualtion process finished")

player.env.close()
