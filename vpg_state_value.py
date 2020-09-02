import spinup
from spinup import vpg_pytorch as vpg
import torch
import gym
import gridworlds
import torch
import matplotlib.pyplot as plt


env_fn = gym.make('gridworld-v0')

ac_kwargs = dict(hidden_sizes=[32], activation=torch.nn.ReLU)

logger_kwargs = dict(output_dir='vpg_results', exp_name='experiment_name')

actor = torch.load('vpg_results/pyt_save/model.pt')

#vpg(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=5000, epochs=250, gamma=0.9,logger_kwargs=logger_kwargs)

v_list = []

for i in range(env_fn.env.n):
    for j in range(env_fn.env.m):
        obs_input = torch.Tensor([i,j]).float()
        action, v, log_p = actor.step(obs_input)
        v_list.append(v)

points = [
    (50, 450),
    (50, 350),
    (50, 250),
    (50, 150),
    (50, 50),
    (150, 450),
    (150, 350),
    (150, 250),
    (150, 150),
    (150, 50),
    (250, 450),
    (250, 350),
    (250, 250),
    (250, 150),
    (250, 50),
    (350, 450),
    (350, 350),
    (350, 250),
    (350, 150),
    (350, 50),
    (450, 450),
    (450, 350),
    (450, 250),
    (450, 150),
    (450, 50)]

x = list(map(lambda x: x[0], points))
y = list(map(lambda x: x[1], points))
plt.figure(figsize=(6,6))

for i in range(env_fn.env.n_states):
    plt.text(x[i], y[i],"%.2f"% v_list[i], fontsize=12,horizontalalignment='center',verticalalignment='center')

plt.rc('grid', linestyle="-", color='black')
plt.grid(True)


plt.xlim(0,500)
plt.ylim(0,500)
plt.show()
plt.save("vpg_state_value.png")
