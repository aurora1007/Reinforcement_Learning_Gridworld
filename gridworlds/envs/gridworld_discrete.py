import numpy as np

import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.envs.classic_control import rendering

""" n x m gridworld
"""

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


reward_matrix = np.zeros([5,5])
reward_matrix[0,1] = 10
reward_matrix[0,3] = 5

class GridWorld(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
        }
    def __init__(self, my_reward=reward_matrix, start_state=0): 
        self.n_states = my_reward.size
        self.n_actions = 4
        self.reward_matrix = my_reward
        self.done = False
        self.start_state = start_state #if not isinstance(start_state, str) else np.random.rand(n**2)
        self.reset()
        self.upper_steps = 200000
        self.steps = 0
        self.n = my_reward.shape[0]
        self.m = my_reward.shape[1]
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low = (0,0), high = (self.n-1, self.m-1),shape=(self.n.self.m), dtype=np.int) 
        self.viewer = rendering.Viewer(700, 700)
        self.gamma = 0.9
 
    def step(self, action):
        assert self.action_space.contains(action)
        self.steps +=1
        if self.steps >= self.upper_steps:
            self.done = True
            return self.state, self._get_reward(self.state, action), self.done, None

        [row, col] = self.ind2coord(self.state)
        reward = self._get_reward(self.state, action)
        
        if action == UP:
            row = max(row - 1, 0)
        elif action == DOWN:
            row = min(row + 1, self.n - 1)
        elif action == RIGHT:
            col = min(col + 1, self.m - 1)
        elif action == LEFT:
            col = max(col - 1, 0)

        new_state = self.coord2ind([row, col])
        ### NEED MODIFICATION: not general
        if self.state == 5:
            new_state = 9 
        if self.state == 15:
            new_state = 17
        self.state = new_state

        return self.state, reward, self.done, None

    def _get_reward(self, state, action):
        [row, col] = self.ind2coord(self.state)
        reward = self.reward_matrix[row,col]
        if self.at_border() and reward == 0:
            if row == 0 and action == UP:
                reward = -1.0
            if row == self.n-1 and action == DOWN:
                reward = -1.0
            if col == 0 and action == LEFT:
                reward = -1.0
            if col == self.m-1 and action == RIGHT:
                reward = -1.0
   
        return reward

    def at_border(self):
        [row, col] = self.ind2coord(self.state)
        return (row == 0 or row == self.n - 1 or col == 0 or col == self.n - 1)

    def ind2coord(self, index):
        assert(index >= 0)
        #assert(index < self.n_states - 1)

        col = index // self.n
        row = index % self.n

        return [row, col]


    def coord2ind(self, coord):
        [row, col] = coord
        assert(row < self.n)
        assert(col < self.n)

        return col * self.n + row


    def reset(self):
        self.steps = 0
        self.state = self.start_state if not isinstance(self.start_state, str) else np.random.randint(self.n_states - 1)
        self.done = False
        return self.state

    def render(self, mode='human', close=False):
        self.line1 = rendering.Line((100,100),(600,100))
        self.line2 = rendering.Line((100, 200), (600, 200))
        self.line3 = rendering.Line((100, 300), (600, 300))
        self.line4 = rendering.Line((100, 400), (600, 400))
        self.line5 = rendering.Line((100, 500), (600, 500))
        self.line6 = rendering.Line((100, 600), (600, 600))
        self.line7 = rendering.Line((100, 100), (100, 600))
        self.line8 = rendering.Line((200, 100), (200, 600))
        self.line9 = rendering.Line((300, 100), (300, 600))
        self.line10 = rendering.Line((400, 100), (400, 600))
        self.line11 = rendering.Line((500, 100), (500, 600))
        self.line12 = rendering.Line((600, 100), (600, 600))
        
        # set the position of robot
        self.robot= rendering.make_circle(30)
        self.robotrans = rendering.Transform()
        self.robot.add_attr(self.robotrans)
        self.robot.set_color(0.8, 0.6, 0.4)


        self.line1.set_color(0, 0, 0)
        self.line2.set_color(0, 0, 0)
        self.line3.set_color(0, 0, 0)
        self.line4.set_color(0, 0, 0)
        self.line5.set_color(0, 0, 0)
        self.line6.set_color(0, 0, 0)
        self.line7.set_color(0, 0, 0)
        self.line8.set_color(0, 0, 0)
        self.line9.set_color(0, 0, 0)
        self.line10.set_color(0, 0, 0)
        self.line11.set_color(0, 0, 0)
        self.line12.set_color(0, 0, 0)
        self.viewer.add_geom(self.line1)
        self.viewer.add_geom(self.line2)
        self.viewer.add_geom(self.line3)
        self.viewer.add_geom(self.line4)
        self.viewer.add_geom(self.line5)
        self.viewer.add_geom(self.line6)
        self.viewer.add_geom(self.line7)
        self.viewer.add_geom(self.line8)
        self.viewer.add_geom(self.line9)
        self.viewer.add_geom(self.line10)
        self.viewer.add_geom(self.line11)
        self.viewer.add_geom(self.line12)
        self.viewer.add_geom(self.robot)
        self.x=[150,150,150,150,150,250,250,250,250,250,350,350,350,350,350,450,450,450,450,450,550,550,550,550,550]
        self.y=[150,250,350,450,550,150,250,350,450,550,150,250,350,450,550,150,250,350,450,550,150,250,350,450,550]

        self.robotrans.set_translation(self.x[self.state], self.y[self.state])
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
      
