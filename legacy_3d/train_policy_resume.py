from hand_drawer_open_env import HandDrawerOpenEnv
from sawyer_drawer_open_env import SawyerDrawerOpenEnv
from gripper_drawer_open_env import GripperDrawerOpenEnv
import tensorflow as tf
import joblib

from garage.experiment import run_experiment
from garage.envs import normalize
from garage.experiment.deterministic import set_seed
from garage.tf.algos import PPO
from garage.tf.baselines import GaussianMLPBaseline
from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.tf.policies import GaussianMLPPolicy


def run_task(snapshot_config, *_):
    """Run task."""
    with LocalTFRunner(snapshot_config=snapshot_config, max_cpus=16) as runner:
        data = joblib.load('/data/yxjin/robot/rotation/data/local/experiment/experiment_2020_05_07_06_54_13_0001/itr_2100.pkl')
        algo = data['algo']
        env = data['env']

        runner.setup(algo, env)

        runner.train(n_epochs=40000, batch_size=2048)


run_experiment(
    run_task,
    snapshot_mode='gap',
    snapshot_gap=1,
    seed=1,
)
