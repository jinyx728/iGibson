#  iGibson for Dexterous hand Training


### Description

This repository is a fork from the original iGibson repository, with modifications to create a environment with a dexterous hand and a drawer for training tasks like drawer opening/closing.

Almost all files are the same as the original iGibson repository despite some additions and modifications. See my commit history to find all files that I have changed. I will described the major changes and additions below. For questions about iGibson related code, see the <https://github.com/StanfordVL/iGibson>(original github repo) and <http://svl.stanford.edu/igibson/docs/intro.html>(iGibson documentation).

### Legacy_3d

The `legacy_3d` folder in the root directory is my previous attempts of opening drawers using robot arm/free gripper/dexterous hand, with 3D observation instead of robot vision. One important file is `hand_drawer_open_env_new.py` where an environment with a dexterous hand and a drawer is implemented. The environment follows the OpenAI Gym API. We can use `train_policy.py` to train a MLP network with parallel sampling, and `render_policy.py` to render a trained policy. Using the hyperparameters in the code, a decent policy can be trained, but feel free to try others. Other environments like `gripper_drawer_open_env.py` and `sawyer_drawer_open_env.py` can also be used but they are deprecated and training convergence is not guaranteed.

