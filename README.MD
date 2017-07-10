# RL A3C Pytorch

![A3C LSTM playing Breakout-v0](https://github.com/dgriff777/rl_a3c_pytorch/blob/master/demo/Breakout.gif) ![A3C LSTM playing SpaceInvadersDeterministic-v3](https://github.com/dgriff777/rl_a3c_pytorch/blob/master/demo/SpaceInvaders.gif) ![A3C LSTM playing MsPacman-v0](https://github.com/dgriff777/rl_a3c_pytorch/blob/master/demo/MsPacman.gif) ![A3C LSTM\
 playing BeamRider-v0](https://github.com/dgriff777/rl_a3c_pytorch/blob/master/demo/BeamRider.gif)

This repository includes my implementation with reinforcement learning using Asynchronous Advantage Actor-Critic (A3C) in Pytorch an algorithm from Google Deep Mind's paper "Asynchronous Methods for Deep Reinforcement Learning."

### A3C LSTM

I implemented an A3C LSTM model and trained it in the atari 2600 environments provided in the Openai Gym. So far model currently has shown the best prerfomance I have seen for atari game environments.  Included in repo are trained models for SpaceInvaders-v0, MsPacman-v0, Breakout-v0, BeamRider-v0, Pong-v0 and Asteroids-v0 which have had very good performance and currently hold the top scores on openai gym leaderboard for each of those games. Saved models in trained_models folder. *Trained models may not run properly if you have recently updated gym and atari-py and have v4 atari versions as changes had effects on all other versions of atari games from update as well. To make sure they run properly u need to keep gym <= 0.8.2 and atari-py <= 0.0.21*

Have optimizers using shared statistics for RMSProp and Adam available for use in training as well option to use non shared optimizer.

Gym atari settings are more difficult to train than traditional ALE atari settings as Gym uses stochastic frame skipping and has higher number of discrete actions. Such as Breakout-v0 has 6 discrete actions in Gym but ALE is set to only 4 discrete actions. Also in GYM atari they randomly repeat the previous action with probability 0.25 and there is time/step limit that limits performance.

link to the Gym environment evaluations below


| Tables                                | Best 100 episode Avg  | Best Score  |
| ------------------------------------- |:---------------------:| -----------:|
| [SpaceInvaders-v0][1]                 | 5808.45 ± 337.28      |   13380.0   |
| [SpaceInvaders-v3][2]                 | 6944.85 ± 409.60      |   20440.0   |
| [SpaceInvadersDeterministic-v3][3]    | 79060.10 ± 5826.59    |  167330.0   | 
| [Breakout-v0][4]                      | 739.30 ± 18.43        |     864.0   |
| [Breakout-v3][5]                      | 859.57 ± 1.97         |     864.0   |
| [Pong-v0][6]                          | 20.87 ± 0.04          |      21.0   |
| [Pong-v3][7]                          | 20.94 ± 0.03          |      21.0   |
| [PongDeterministic-v3][8]             | 21.00 ± 0.00          |      21.0   |
| [BeamRider-v0][9]                     | 8441.22 ± 221.24      |   13130.0   |
| [MsPacman-v0][10]                     | 6323.01 ± 116.91      |   10181.0   |


[1]: https://gym.openai.com/evaluations/eval_K69ZjwAnSdOzN7lnUblqA#reproducibility
[2]: https://gym.openai.com/evaluations/eval_uutLMdoQ9qvlnlM01Ptkg#reproducibility
[3]: https://gym.openai.com/evaluations/eval_rZMtqVVuRe28zDIQDYGKSw#reproducibility
[4]: https://gym.openai.com/evaluations/eval_CyVPHgs0S22DiZsWXoPFw#reproducibility
[5]: https://gym.openai.com/evaluations/eval_X3ywdh8pTmWFw51ISjZvvQ#reproducibility
[6]: https://gym.openai.com/evaluations/eval_qZsD1qHORqCZWg7wPpmb0w#reproducibility
[7]: https://gym.openai.com/evaluations/eval_DOrPDl1DSImVBTjGeF9Mw
[8]: https://gym.openai.com/evaluations/eval_tM4E3BiQUOI14yMMa602A#reproducibility
[9]: https://gym.openai.com/evaluations/eval_pl5bvWR8Somu8PfFJzTryA#reproducibility
[10]: https://gym.openai.com/evaluations/eval_8Wwndzd8R62np8CxVQWEeg#reproducibility

*The 167,330 Space Invaders score is World Record Space Invaders score and game ended only due to GYM timestep limit and not from loss of life*

## Requirements

- Python 2.7+
- Openai Gym and Universe
- Pytorch

## Training
*When training model it is important to limit number of worker threads to number of cpu cores available as too many threads (e.g. more than one thread per cpu core available) will actually be detrimental in training speed and effectiveness*

To train agent in Pong-v0 environment with 32 different worker threads:

```
python main.py --env Pong-v0 --workers 32
```

Hit Ctrl C to end training session properly

![A3C LSTM playing Pong-v0](https://github.com/dgriff777/rl_a3c_pytorch/blob/master/demo/Pong.gif)

## Evaluation
To run a 100 episode gym evaluation with trained model
```
python gym_eval.py --env Pong-v0 --workers 100
```

## Project Reference

- https://github.com/ikostrikov/pytorch-a3c
