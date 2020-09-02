import spinup
from spinup import vpg_pytorch as vpg
import torch
import gym
import gridworlds

env_fn = lambda : gym.make('gridworld-v0')

ac_kwargs = dict(hidden_sizes=[32], activation=torch.nn.ReLU)

logger_kwargs = dict(output_dir='vpg_results', exp_name='experiment_name')

vpg(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=5000, epochs=250, gamma=0.9,logger_kwargs=logger_kwargs)



