import numpy as np

import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.envs.classic_control import rendering
import pyglet
from pyglet import gl


""" n x m gridworld
"""

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


reward_matrix = np.zeros([5,5])
reward_matrix[0,1] = 10
reward_matrix[0,3] = 5


class DrawText:
    def __init__(self, label:pyglet.text.Label):
        self.label=label
    def render(self):
        self.label.draw()


class GridWorld(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 1
        }
    def __init__(self, my_reward=reward_matrix, start_state=np.array([0,0]), upper_steps=20): 
        self.n_states = my_reward.size
        self.n_actions = 4
        self.reward_matrix = my_reward
        self.done = False
        self.start_state = start_state 
        self.reset()
        self.upper_steps = upper_steps
        self.steps = 0
        self.n = my_reward.shape[0]
        self.m = my_reward.shape[1]
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low = np.array([0,0]), high = np.array([self.n-1, self.m-1])) 
        self.viewer = rendering.Viewer(700, 700)
        self.gamma = 0.9
 
    def step(self, action):
        assert self.action_space.contains(action)
        self.steps +=1
        if self.steps >= self.upper_steps:
            self.done = True
            return self.state, self._get_reward(self.state, action), self.done, None
       # print("state:{}  action:{}".format(self.state, action))
        row, col = self.state
        reward = self._get_reward(self.state, action)


        if row == 0 and col == 1:
            row,col = [4,1]
        elif row == 0 and col == 3:
            row,col = [2,3]
        else:
            if action == UP:
                row = max(row - 1, 0)
            elif action == DOWN:
                row = min(row + 1, self.n - 1)
            elif action == RIGHT:
                col = min(col + 1, self.m - 1)
            elif action == LEFT:
                col = max(col - 1, 0)

        new_state = np.array([row, col])
        
        self.state = new_state
        return self.state, reward, self.done, None

    def _get_reward(self, state, action):
        row, col = state
        reward = self.reward_matrix[row, col]
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
        row, col = self.state
        return (row == 0 or row == self.n - 1 or col == 0 or col == self.n - 1)

    def reset(self):
        self.steps = 0
        self.state = self.start_state 
        self.done = False
        return self.state

    def render(self, mode='human', close=False):

        if self.state.any():
            self.viewer.geoms.clear()
            self.viewer.onetime_geoms.clear()

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
        
        self.x=[150,150,150,150,150,250,250,250,250,250,350,350,350,350,350,450,450,450,450,450,550,550,550,550,550]
        self.y=[550,450,350,250,150,550,450,350,250,150,550,450,350,250,150,550,450,350,250,150,550,450,350,250,150]
        row, col = self.state
        index = col*self.n + row


        # set the position of robot
        self.robot= rendering.make_circle(30)
        self.robotrans = rendering.Transform(translation=(self.x[index], self.y[index]))
        self.robot.add_attr(self.robotrans)
        self.robot.set_color(0.1, 0.1, 0.1)


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
        if self.state is None: return None
        
        label_A = pyglet.text.Label('A', font_size=36,x=250, y=550, anchor_x='center', anchor_y='center',color=(0, 0, 139, 175))
        label_A.draw()
        self.viewer.add_geom(DrawText(label_A))
        label_B = pyglet.text.Label('B', font_size=36,x=450, y=550, anchor_x='center', anchor_y='center',color=(204, 0, 0, 175))
        label_B.draw()
        self.viewer.add_geom(DrawText(label_B))
        label_A_prime = pyglet.text.Label('A\'', font_size=36,x=250, y=150, anchor_x='center', anchor_y='center',color=(0, 0, 139, 175))
        label_A_prime.draw()
        self.viewer.add_geom(DrawText(label_A_prime))
        label_B_prime = pyglet.text.Label('B\'', font_size=36,x=450, y=350, anchor_x='center', anchor_y='center',color=(204, 0, 0, 175))
        label_B_prime.draw()
        self.viewer.add_geom(DrawText(label_B_prime))


        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
      
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
