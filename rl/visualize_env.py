import time
import sys

def visualize_env(env, mode, max_steps=sys.maxsize, speedup=1):
    timestep = 0.05
    # step ahead with all-zero action
    if mode == 'noop':
        for _ in range(max_steps):
            env.render()
            time.sleep(timestep / speedup)
    elif mode == 'random':
        env.reset()
        env.render()
        for i in range(max_steps):
            action = env.action_space.sample()
            _, reward, done, _ = env.step(action)
            # if i % 10 == 0:
            env.render()
            # import time as ttime
            time.sleep(timestep / speedup)
            if done:
                env.reset()
    else:
        raise ValueError('Unsupported mode: %s' % mode)