import numpy as np
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
from rl.core import Env, Space
from ipdb import set_trace as stop
import matplotlib.pyplot as pl

class Box(Space):
    """
    A box in R^n.
    I.e., each coordinate is bounded.
    """

    def __init__(self, low, high, shape=None):
        """
        Two kinds of valid input:
            Box(-1.0, 1.0, (3,4)) # low and high are scalars, and shape is provided
            Box(np.array([-1.0,-2.0]), np.array([2.0,4.0])) # low and high are arrays of the same shape
        """
        if shape is None:
            assert low.shape == high.shape
            self.low = low
            self.high = high
        else:
            assert np.isscalar(low) and np.isscalar(high)
            self.low = low + np.zeros(shape)
            self.high = high + np.zeros(shape)

    def sample(self):
        return np.random.uniform(low=self.low, high=self.high, size=self.low.shape)

    def contains(self, x):
        return x.shape == self.shape and (x >= self.low).all() and (x <= self.high).all()

    @property
    def shape(self):
        return self.low.shape

class RL_Env(Env):
    @property
    def action_space(self):
        return Box(low=-0.1, high=0.1, shape=(2,))

    @property
    def reward_range(self):
        return (0.0, np.inf)

    @property
    def observation_space(self):
        return Box(low=-1.0, high=1.0, shape=(2,))

    def __init__(self):
        self.fig = None

    def step(self, action):
        """Run one timestep of the environment's dynamics.
        Accepts an action and returns a tuple (observation, reward, done, info).
        # Arguments
            action (object): An action provided by the environment.
        # Returns
            observation (object): Agent's observation of the current environment.
            reward (float) : Amount of reward returned after previous action.
            done (boolean): Whether the episode has ended, in which case further step() calls will return undefined results.
            info (dict): Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        self._state = self._state + np.clip(action, -0.1, 0.1)
        x, y = self._state
        self._reward = - (x**2 + y**2) ** 0.5
        done = abs(x) < 0.01 and abs(y) < 0.01
        next_observation = np.copy(self._state)
        return (next_observation, self._reward, done, {})

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.
        
        # Returns
            observation (object): The initial observation of the space. Initial reward is assumed to be 0.
        """
        self._state = np.random.uniform(-1, 1, size=(2,))
        x, y = self._state
        self._reward = - (x**2 + y**2) ** 0.5
        observation = np.copy(self._state)
        return observation

    def render(self, mode='human', close=False):
        """Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) 
        
        # Arguments
            mode (str): The mode to render with.
            close (bool): Close all open renderings.
        """
        if (self.fig is None):
            self.fig, self.ax = pl.subplots()

        self.ax.plot(self._state[0], self._state[1], '.', color='black')
        # print(" State : {0}".format(self._state))

    def close(self):
        """Override in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        pass

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).
        
        # Returns
            Returns the list of seeds used in this env's random number generators
        """
        np.random.seed(seed)
        return

    def configure(self, *args, **kwargs):
        """Provides runtime configuration to the environment.
        This configuration should consist of data that tells your
        environment how to run (such as an address of a remote server,
        or path to your ImageNet data). It should not affect the
        semantics of the environment.
        """
        pass
