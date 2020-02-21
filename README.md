# Reinforcement Learning Gridworld
Implement gridworld using OpenAI gym specification. Sample the dynamics of gridworld under a random policy and estimate state-value function like in example 3.5 on Page 60 of Sutton and Barto.



1. Check you class is working and meet specification by running the random agent:



Usage
-----

.. code:: shell

        $ import gym
        $ import gym_gridworlds
        $ env = gym.make('Gridworld-v0')  # substitute environment's name


.. code:: shell

        python3 random_agent.py 'gridworld-v0'

2. Sample the dynamics of your gridworld under a random policy and estimate state-value function
.. code:: shell

    python3 vpg_state_value.py

3. (b) Read and execute the Vanilla Policy Gradient code on your gridworld environment:
.. code:: shell

    python3 vpg.py

   (c)Supply a training artifact (a plot of either timesteps vs. total reward, trajectories sampled vs. total reward) showing that VPG converges to some score. In words, write down an explancation for this score in terms of the trajectory length and reward specified in Example 3.5
.. code:: shell

    python3 spinningup/spinup/utils/plot.py vpg_results/

   (d)The VPG code estimates the state-value function. Plot these values.
.. code:: shell

    python3 vpg_state_value.py 
