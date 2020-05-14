from gibson2.envs.hand_drawer_env import HandDrawerEnv
from time import time
import numpy as np
from time import time
import gibson2
import os
from gibson2.core.render.profiler import Profiler
import logging

#logging.getLogger().setLevel(logging.DEBUG) #To increase the level of logging

def main():
    config_filename = os.path.join(os.path.dirname(gibson2.__file__),
                                   '../examples/configs/hand_drawer.yaml')
    env = HandDrawerEnv(config_file=config_filename, mode='gui')
    for j in range(1000):
        env.reset()
        # for i in range(100):
        #     with Profiler('Environment action step'):
        #         action = env.action_space.sample()
        #         state, reward, done, info = env.step(action)
        #         if done:
        #             logging.info("Episode finished after {} timesteps".format(i + 1))
        #             break

if __name__ == "__main__":
    main()
