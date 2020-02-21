# Reinforcement Learning Gridworld
Implement gridworld using OpenAI gym specification. Sample the dynamics of gridworld under a random policy and estimate state-value function like in example 3.5 on Page 60 of Sutton and Barto.



1. Check you class is working and meet specification by running the random agent:


```
    python3 random_agent.py 'gridworld-v0'
```
2. Sample the dynamics of your gridworld under a random policy and estimate state-value function
```
    python3 vpg_state_value.py   
```

3. (b) Read and execute the Vanilla Policy Gradient code on your gridworld environment:
```
    python3 vpg.py
```
3. (c) Supply a training artifact (a plot of either timesteps vs. total reward, trajectories sampled vs. total reward) showing that VPG converges to some score. In words, write down an explancation for this score in terms of the trajectory length and reward specified in Example 3.5
```
    python3 spinningup/spinup/utils/plot.py vpg_results/
```
3. (d) The VPG code estimates the state-value function. Plot these values.
```
    python3 vpg_state_value.py 
```
