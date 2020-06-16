import gym
from hand_drawer_open_env_new import HandDrawerOpenEnv
from sawyer_drawer_open_env import SawyerDrawerOpenEnv
from gripper_drawer_open_env import GripperDrawerOpenEnv
# from metaworld.envs.mujoco.sawyer_xyz.sawyer_drawer_open_ext import SawyerDrawerOpenEnv
import numpy as np
import pybullet as p
# from env_bases import MJCFBaseBulletEnv
from PIL import Image

env = HandDrawerOpenEnv()
observation = env.reset()
for i in range(1000):
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    # image_array = env.render(mode="rgb_array")
    # im = Image.fromarray(image_array)
    # im.save("image/"+str(i)+".jpg")
    print(reward)
    if (i+1) % 100 == 0:
        observation = env.reset()
env.close()
