import gym
from metaworld.envs.mujoco.sawyer_xyz.sawyer_drawer_open_ext import SawyerDrawerOpenEnv
env = SawyerDrawerOpenEnv(rotMode='quat')
observation = env.reset()
for _ in range(1000):
  env.render()
  action = env.action_space.sample() # your agent here (this takes random actions)
  observation, reward, done, info = env.step(action)

  if done:
    observation = env.reset()
env.close()