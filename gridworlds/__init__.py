from gym.envs.registration import register

# using max_episode_steps for new API 

register(
    id='gridworld-v0',
    entry_point='gridworlds.envs:GridWorld',
    max_episode_steps=100000,
)
