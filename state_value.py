import numpy as np 
import gym.spaces
import gridworlds

env = gym.make('gridworld-v0')


def policy_eval(policy, env, discount_factor=0.9, epsilon=0.000001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        epsilon: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with a random (all 0) value function
    V_old = np.zeros(env.n_states)
    env.reset()
    env.env.upper_steps = 100000
    while True:
        #new value function
        V_new = np.zeros(env.n_states)
        #stopping condition
        delta = 0

        #loop over state space
        for s in range(env.n_states):
            #To accumelate bellman expectation eqn
            v_fn = 0
            #get probability distribution over actions
            action_probs = policy[s]
            col = s // env.env.n
            row = s % env.env.n

            #loop over possible actions
            for a in range(env.n_actions):
                env.render()
                env.env.state = np.array([row, col])
                #get transitions
                [next_state, reward, done, _] = env.step(a)
                next_row, next_col = next_state
                next_state_1d = env.env.n * next_col + next_row 
                #apply bellman expectatoin eqn
                v_fn += action_probs[a] * (reward + discount_factor * V_old[next_state_1d])
                print("Current state: {}, env.state:{} ,Action: {}, Reward: {}, next_state: {}, v_fn:{}".format(s, env.state,a, reward, next_state, v_fn))
            #get the biggest difference over state space
            delta = max(delta, abs(v_fn - V_old[s]))

            #update state-value
            V_new[s] = v_fn
        print("delta", delta)    
        #the new value function
        V_old = V_new

        #if true value function
        if(delta < epsilon):
            break
    env.env.close()
    return np.array(V_old)


random_policy = np.ones([env.n_states, env.n_actions]) / env.n_actions
v = policy_eval(random_policy, env)

expected_v = np.array([3.3, 1.5, 0.1, -1.0, -1.9, 
                       8.8, 3.0, 0.7, -0.4, -1.3, 
                       4.4, 2.3, 0.7, -0.4, -1.2, 
                       5.3, 1.9, 0.4, -0.6, -1.4,
                       1.5, 0.5,-0.4, -1.2, -2.0])
np.testing.assert_array_almost_equal(v, expected_v, decimal=1)

print("State Value Function",v)
print("Expected State Value Function:", expected_v)
