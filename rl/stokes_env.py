import numpy as np
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
from rl.core import Env, Space
from ipdb import set_trace as stop
import matplotlib.pyplot as pl
import pyiacsun as ps

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
    # @property
    # def action_space(self):
    #     return Box(low=-0.1, high=0.1, shape=(2,))

    # @property
    # def reward_range(self):
    #     return (0.0, np.inf)

    # @property
    # def observation_space(self):
    #     return Box(low=-1.0, high=1.0, shape=(2,))

    def _physical_to_transformed(self, x):
        """
        Transform from physical parameters to transformed (unconstrained) ones
        
        Args:
            x (TYPE): vector of parameters
        
        Returns:
            TYPE: transformed vector of parameters
        """
        return (x-self.lower) / (self.upper - self.lower)

    def _transformed_to_physical(self, x):
        """
        Transform from transformed (unconstrained) parameters to physical ones
        
        Args:
            x (TYPE): vector of transformed parameters
        
        Returns:
            TYPE: vector of parameters
        """
        return self.lower + (self.upper - self.lower) * x

    def _vector_to_nodes(self, vector):
        """
        Transform from a vector of parameters to the structure of nodes, made of lists of lists
        
        Args:
            vector (float): model parameters
        
        Returns:
            TYPE: structure of nodes
        """
        nodes = []
        loop = 0

        for n in self.n_nodes:
            temp = []
            for i in range(n):
                temp.append(vector[loop])
                loop += 1
            nodes.append(temp)
        return nodes

    def _nodes_to_vector(self, nodes):
        """Summary
        
        Args:
            nodes (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        return np.asarray([item for sublist in nodes for item in sublist])

    def _interpolateNodes(self, logTau, variable, nodes):
        n = logTau.shape[0]
        out = np.zeros_like(variable)

        if (len(nodes) == 0):
            out = variable

        if (len(nodes) == 1):
            out = variable + nodes[0]
        
        if (len(nodes) >= 2):
            pos = np.linspace(0, n-1, len(nodes), dtype=int)
            coeff = np.polyfit(logTau[pos], nodes, len(nodes)-1)
            out = variable + np.polyval(coeff, logTau)
        return out

    def synth_profile(self, vector):

        vector_physical_units = self._transformed_to_physical(vector)
        nodes = self._vector_to_nodes(vector_physical_units)
        atmos = np.copy(self.reference_atmos)
    
        for i, n in enumerate(nodes):
            atmos[:,i+1] = self._interpolateNodes(self.log_tau, self.reference_atmos[:,i+1], n)

        return ps.radtran.synthesizeSIR(atmos, returnRF=False)[1:,:]

    def __init__(self):
        self.fig = None

        atmos = np.loadtxt('model.mod',dtype=np.float32, skiprows=1)
        lines = np.loadtxt('lines.dat')
        self.wavelength = np.loadtxt('wavelengthHinode.txt')
        self.nLambda = len(self.wavelength)

        self.reference_atmos = atmos

        self.log_tau = atmos[:,0]

        lines = [None]

        left = (np.min(self.wavelength) - 6301.5080) * 1e3
        right = (np.max(self.wavelength) - 6301.5080) * 1e3
        delta = (self.wavelength[1] - self.wavelength[0])* 1e3
        lines[0] = ['200, 201', left, delta, right]

        self.n_lambda = ps.radtran.initializeSIR(lines)

        self.nDepth = len(self.reference_atmos[:,0])
        
        self.contHSRA = ps.util.contHSRA(np.mean(self.wavelength))

# Define number of nodes and set their ranges
#                       T  vmic  B  v  thetaB  phiB
        self.n_nodes = [0,0,0,1,0,0]
        self.n_nodes_total = np.sum(self.n_nodes)
        self.n_unknowns = self.n_nodes_total

        lower = [-2000.0, 0.01, 0.0, -5.e5, 0.0, 0.0]
        upper = [2000.0, 5.0, 3000.0, 5.e5, 180.0, 180.0]

        self.lower = []
        self.upper = []

        for i in range(6):
            self.lower.append([lower[i]]*self.n_nodes[i])
            self.upper.append([upper[i]]*self.n_nodes[i])


        self.lower = np.hstack(self.lower)
        self.upper = np.hstack(self.upper)


        self.action_space = Box(low=-0.1, high=0.1, shape=(self.n_nodes_total,))
        self.reward_range = (0.0, np.inf)
        self.observation_space = Box(low=0.0, high=2.0, shape=(self.n_lambda,))
        self.noise = 1e-3

    def compute_reward(self, state):
        stokes = self.synth_profile(state)
        reward = -np.sum((stokes[0,:] - self.current_obs)**2 / self.noise**2) / (4.0*self.n_lambda)
        done = False
        if (reward < 1.5):
            done = True

        return reward, done, stokes

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
        
        self._state = self._state + action
        self._state = np.clip(action, -5.0, 5.0)
        self._reward, done, stokes = self.compute_reward(self._state)
        next_observation = stokes[0,:] - self.current_obs
        return (next_observation, self._reward, done, {})

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.
        
        # Returns
            observation (object): The initial observation of the space. Initial reward is assumed to be 0.
        """
        self._state = np.random.uniform(0.0, 1.0, size=(self.n_nodes_total,))
        self._obs_pars = np.random.uniform(0.0, 1.0, size=(self.n_nodes_total,))
        self.current_obs = self.synth_profile(self._obs_pars)[0,:]
        self.current_obs += np.random.normal(loc=0.0, scale=self.noise, size=self.current_obs.shape)
        self._reward = self.compute_reward(self._state)
        return self.current_obs

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
            self.ax.plot(self.current_obs)

        synth = self.synth_profile(self._state)[0,:]
        print (self._state)
        self.ax.plot(synth)
        pl.show()
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

if (__name__ == '__main__'):
    env = RL_Env()