import numpy as np
import matplotlib.pyplot as pl
import os
from visualize_env import visualize_env
from stokes_env import Box, RL_Env

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, merge
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from keras.utils import plot_model

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

from ipdb import set_trace as stop

if (__name__ == '__main__'):
    env = RL_Env()
    env.seed(123)
    # visualize_env(env, 'random', max_steps=200, speedup=100)

    # stop()

    assert len(env.action_space.shape) == 1
    nb_actions = env.action_space.shape[0]

# Next, we build a very simple model.
    actor = Sequential()
    actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    actor.add(Dense(16))
    actor.add(Activation('relu'))
    actor.add(Dense(nb_actions))
    actor.add(Activation('linear'))
    print(actor.summary())

    plot_model(actor, to_file='actor.png', show_shapes=True)

    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
    flattened_observation = Flatten()(observation_input)
    x = concatenate([action_input, flattened_observation])
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(1)(x)
    x = Activation('linear')(x)
    critic = Model(inputs=[action_input, observation_input], outputs=x)
    print(critic.summary())

    plot_model(critic, to_file='critic.png', show_shapes=True)

# # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# # even the metrics!
    memory = SequentialMemory(limit=10000, window_length=1)
    random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=0.15, mu=0., sigma=.3)
    agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                   memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                   random_process=random_process, gamma=.99, target_model_update=1e-3)
    agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

# # Okay, now it's time to learn something! We visualize the training here for show, but this
# # slows down training quite a lot. You can always safely abort the training prematurely using
# # Ctrl + C.
    agent.fit(env, nb_steps=25000, visualize=False, verbose=1, nb_max_episode_steps=200)

# # After training is done, we save the final weights.
    agent.save_weights('ddpg_stokes_weights.h5f', overwrite=True)

# # Finally, evaluate our algorithm for 5 episodes.
    agent.test(env, nb_episodes=5, visualize=True, nb_max_episode_steps=200)


