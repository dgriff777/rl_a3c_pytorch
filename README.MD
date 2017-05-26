## README still under construction

# RL A3C Pytorch

![A3C LSTM playing Breakout-v0](https://github.com/dgriff777/rl_a3c_pytorch/blob/master/demo/Breakout.gif)

This repository includes my implementation with reinforcement learning using Asynchronous Advantage Actor-Critic (A3C) in Pytorch an algorithm from Google Deep Mind's paper "Asynchronous Methods for Deep Reinforcement Learning."

### A3C LSTM

I implemented an A3C LSTM model and trained it in the atari 2600 environments provided in the Openai Gym. I have only trained model on environments for Pong-v0, MsPacman-v0, and Breakout-v0 but had very good performance and currently hold top scores on openai gym leaderboard for those games. Saved models in trained_models folder. Plan to train agent for SpaceInvaders-v0 and Asteroids-v0 in future.

Gym atari settings are more difficult to train than traditional ALE atari settings as Gym uses stochastic frame skipping and has higher number of discrete actions. Such as Breakout-v0 has 6 discrete actions in Gym but ALE is set to only 4 discrete actions.

link to the my Gym environment evaluations below

- https://gym.openai.com/envs/Breakout-v0
- https://gym.openai.com/envs/Pong-v0
- https://gym.openai.com/envs/MsPacman-v0


![A3C LSTM playing MsPacman-v0](https://github.com/dgriff777/rl_a3c_pytorch/blob/master/demo/MsPacman.gif)

## Requirements

- Python 2.7+
- Openai Gym and Universe
- Pytorch

## Training
To train agent in Pong-v0 environment with 32 different worker threads:

```
OMP_NUM_THREADS=1 python main.py --env-name Pong-v0 --num-processes 32
```

Hit Ctrl C to end training session properly

![A3C LSTM playing Pong-v0](https://github.com/dgriff777/rl_a3c_pytorch/blob/master/demo/Pong.gif)

## Evaluation
To run a 100 episode gym evaluation with trained model
```
python gym_eval.py --env-name Pong-v0 --num-episodes 100
```

## Project Reference

- https://github.com/ikostrikov/pytorch-a3c

## README still under construction
