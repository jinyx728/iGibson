from hand_drawer_open_env_new import HandDrawerOpenEnv
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
from garage.sampler import LocalSampler, RaySampler


def run_task(snapshot_config, *_):
    """Run task."""
    with LocalTFRunner(snapshot_config=snapshot_config, max_cpus=16) as runner:
        env = TfEnv(normalize(HandDrawerOpenEnv()))

        policy = GaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=(128, 128),
            hidden_nonlinearity=tf.nn.tanh,
            output_nonlinearity=None,
        )

        baseline = GaussianMLPBaseline(
            env_spec=env.spec,
            regressor_args=dict(
                hidden_sizes=(128,128),
                hidden_nonlinearity=tf.nn.tanh,
                output_nonlinearity=None,
                use_trust_region=True,
            ),
        )

        # NOTE: make sure when setting entropy_method to 'max', set
        # center_adv to False and turn off policy gradient. See
        # tf.algos.NPO for detailed documentation.
        algo = PPO(
            env_spec=env.spec,
            policy=policy,
            baseline=baseline,
            max_path_length=1000,
            discount=0.995,
            gae_lambda=0.97,
            lr_clip_range=0.2,
            optimizer_args=dict(
                batch_size=64,
                max_epochs=10,
                tf_optimizer_args=dict(learning_rate=3e-4),
                verbose=True
            ),
            stop_entropy_gradient=True,
            entropy_method='max',
            policy_ent_coeff=0.0,
            center_adv=False,
        )

        runner.setup(algo, env, sampler_cls=RaySampler, n_workers=16)

        runner.train(n_epochs=20000, batch_size=32000)


run_experiment(
    run_task,
    snapshot_mode='gap',
    snapshot_gap=1,
    seed=1,
)

# run_experiment(
#     run_task,
#     snapshot_mode='last',
#     seed=1,
# )
