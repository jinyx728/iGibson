import argparse

import joblib
import tensorflow as tf

from garage.sampler.utils import rollout
from hand_drawer_open_env_new import HandDrawerOpenEnv
from sawyer_drawer_open_env import SawyerDrawerOpenEnv
from gripper_drawer_open_env import GripperDrawerOpenEnv
from garage.envs import normalize
from garage.tf.envs import TfEnv
from PIL import Image

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, help='path to the snapshot file')
    parser.add_argument(
        '--max_path_length',
        type=int,
        default=1000,
        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=1, help='Speedup')
    args = parser.parse_args()

    with tf.compat.v1.Session() as sess:
        data = joblib.load(args.file)
        policy = data['algo'].policy
        env = data['env']
        deterministic=False
        animated=True

        o = env.reset()
        policy.reset()
        path_length = 0
        rcum = 0
        while path_length < (args.max_path_length or np.inf):
            o = env.observation_space.flatten(o)
            a, agent_info = policy.get_action(o)
            if deterministic and 'mean' in agent_infos:
                a = agent_info['mean']
            next_o, r, d, env_info = env.step(a)
            rcum += r
            print(r)
            path_length += 1
            if d:
                break
            o = next_o
            if animated:
                image_array = env.render(mode="rgb_array")
                im = Image.fromarray(image_array)
                im.save("image/"+str(path_length)+".jpg")
        print(rcum)
