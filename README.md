#  iGibson for Dexterous hand Training


### Description

This repository is a fork from the original iGibson repository, with modifications to create a environment with a dexterous hand and a drawer for training tasks like drawer opening/closing.

Almost all files are the same as the original iGibson repository despite some additions and modifications. See my commit history to find all files that I have changed. I will described the major changes and additions below. For questions about iGibson related code, see the [original github repo](https://github.com/StanfordVL/iGibson) and [iGibson documentation](http://svl.stanford.edu/igibson/docs/intro.html).

### Legacy_3d

The `legacy_3d` folder in the root directory is my previous attempts of opening drawers using robot arm/free gripper/dexterous hand, with 3D observation instead of robot vision. One important file is `hand_drawer_open_env_new.py` where an environment with a dexterous hand and a drawer is implemented. The environment follows the OpenAI Gym API. We can use `train_policy.py` to use PPO to train a MLP with parallel sampling, and `render_policy.py` to render a trained policy. We use <https://github.com/rlworkgroup/garage>(garage) library implementation of PPO. Using the hyperparameters in the code, a decent policy can be trained, but feel free to try others. Other environments like `gripper_drawer_open_env.py` and `sawyer_drawer_open_env.py` can also be used but they are deprecated and training convergence is not guaranteed.

### Dexterous Hand Env in iGibson

Two main files that define this environment are:

- `gibson2/envs/hand_drawer_env.py`: define the environment in OpenAI Gym style.
- `gibson2/core/physics/robot_locomotors.py`: the `DexHandRobot` class defines the dexterous hand as well as APIs about it.

Other subtle code changes can be found in the commit log. In `examples/demo` folder, we can use `hand_drawer_example.py` to test the environment.

### CURL/DrQ

The algorithms that are used to train the environment in iGibson are CURL and DrQ, which are both located in the root directory, and are copied from the open source implementation of their original papers. The code CURL and DrQ maintains very similar structure because they are both based on SAC-AE implementation. See the original repo of <https://github.com/MishaLaskin/curl>(CURL), <https://github.com/denisyarats/drq>(DrQ), and <https://github.com/denisyarats/pytorch_sac_ae>(SAC-AE) for more details.
