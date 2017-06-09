
# RL A3C Pytorch

![A3C LSTM playing Breakout-v0](https://github.com/dgriff777/rl_a3c_pytorch/blob/master/demo/Breakout.gif) ![A3C LSTM playing SpaceInvadersDeterministic-v3](https://github.com/dgriff777/rl_a3c_pytorch/blob/master/demo/SpaceInvaders.gif)

This repository includes my implementation with reinforcement learning using Asynchronous Advantage Actor-Critic (A3C) in Pytorch an algorithm from Google Deep Mind's paper "Asynchronous Methods for Deep Reinforcement Learning."

### A3C LSTM

I implemented an A3C LSTM model and trained it in the atari 2600 environments provided in the Openai Gym. Included in repo are trained models for Pong-v0, MsPacman-v0, Breakout-v0, BeamRider-v0, and Asteroids-v0 which have had very good performance and currently hold the top scores on openai gym leaderboard for each of those games. Saved models in trained_models folder. Added a pre-trained SpaceInvadersDeterministic-v3 model which you can see in gif getting a score of 30,330! and scores often higher and as high as 50,000! Can see link for gym evaluation for it below..

Have optimizers using shared statistics for RMSProp and Adam available for use in training as well option to use non shared optimizer.

Gym atari settings are more difficult to train than traditional ALE atari settings as Gym uses stochastic frame skipping and has higher number of discrete actions. Such as Breakout-v0 has 6 discrete actions in Gym but ALE is set to only 4 discrete actions. Also in GYM atari they randomly repeat the previous action with probability 0.25 and there is time/step limit that limits performance.

link to the Gym environment evaluations below

[SpaceInvadersDeterministic-v3](https://gym.openai.com/evaluations/eval_ZaX8BbF5Rl6Hi6CViyMuoQ#reproducibility)

[Breakout-v0](https://gym.openai.com/envs/Breakout-v0)

[BeamRider-v0](https://gym.openai.com/envs/BeamRider-v0)

[MsPacman-v0](https://gym.openai.com/envs/MsPacman-v0)



![A3C LSTM playing MsPacman-v0](https://github.com/dgriff777/rl_a3c_pytorch/blob/master/demo/MsPacman.gif) ![A3C LSTM\
 playing BeamRider-v0](https://github.com/dgriff777/rl_a3c_pytorch/blob/master/demo/BeamRider.gif) 


## Requirements

- Python 2.7+
- Openai Gym and Universe
- Pytorch

## Training
To train agent in Pong-v0 environment with 32 different worker threads:

```
python main.py --env-name Pong-v0 --num-processes 32
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


