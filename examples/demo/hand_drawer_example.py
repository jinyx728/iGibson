from gibson2.envs.hand_drawer_env import HandDrawerEnv
from time import time
import numpy as np
from time import time
import gibson2
import os
from gibson2.core.render.profiler import Profiler
import logging
import pybullet as p 
from PIL import Image

#logging.getLogger().setLevel(logging.DEBUG) #To increase the level of logging

def main():
    config_filename = os.path.join(os.path.dirname(gibson2.__file__),
                                   '../examples/configs/hand_drawer.yaml')
    env = HandDrawerEnv(config_file=config_filename, mode='gui')
    for j in range(1000):
        env.reset()
        for i in range(100):
            with Profiler('Environment action step'):
                action = env.action_space.sample()
                action[0:24] = 0
                action[-7] = 0.05
                action[-6] = 1
                action[-5] = 1.05
                action[-4:] = p.getQuaternionFromEuler([-np.pi/2,0,np.pi])
                state, reward, done, info = env.step(action) 
                # env.save_seg_image_external('seg/'+str(j*100+i)+'.jpg')
                # env.save_rgb_image('rgb/'+str(j*100+i)+'.jpg')
                # env.save_depth_image_external('depth/'+str(j*100+i)+'.jpg')
                # env.save_normal_image('normal/'+str(j*100+i)+'.jpg')
                # env.save_external_image('external/'+str(j*100+i)+'.jpg')

                if done:
                    logging.info("Episode finished after {} timesteps".format(i + 1))
                    break

if __name__ == "__main__":
    main()
