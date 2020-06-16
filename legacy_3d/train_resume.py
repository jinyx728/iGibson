import tensorflow as tf

from garage.experiment import run_experiment
from garage.envs import normalize
from garage.experiment.deterministic import set_seed
from garage.tf.algos import PPO
from garage.tf.baselines import GaussianMLPBaseline
from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.tf.policies import GaussianMLPPolicy

from gripper_drawer_open_env import GripperDrawerOpenEnv


def run_task(snapshot_config, *_):
    """Run task."""
    with LocalTFRunner(snapshot_config=snapshot_config, max_cpus=16) as runner:
        runner.restore('data/local/experiment/experiment_2020_05_07_06_54_13_0001')
        runner.resume(n_epochs=20000, batch_size=2048)


run_experiment(
    run_task,
    snapshot_gap=1,
    snapshot_mode='gap',
    seed=1,
)
