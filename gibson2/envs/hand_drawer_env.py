import gibson2
from gibson2.envs.base_env import BaseEnv
from gibson2.utils.utils import parse_config
from gibson2.core.simulator import Simulator
from gibson2.core.physics.interactive_objects import InteractiveObj
import pybullet as p
import numpy as np
import os

class HandDrawerEnv(BaseEnv):
    def __init__(
            self,
            config_file,
            model_id=None,
            mode='headless',
            action_timestep=1 / 10.0,
            physics_timestep=1 / 240.0,
            automatic_reset=False,
            device_idx=0,
            render_to_tensor=False
    ):
        """
        :param config_file: config_file path
        :param model_id: override model_id in config file
        :param mode: headless or gui mode
        :param action_timestep: environment executes action per action_timestep second
        :param physics_timestep: physics timestep for pybullet
        :param device_idx: device_idx: which GPU to run the simulation and rendering on
        """
        self.config = parse_config(config_file)
        if model_id is not None:
            self.config['model_id'] = model_id

        self.automatic_reset = automatic_reset
        self.mode = mode
        self.action_timestep = action_timestep
        self.physics_timestep = physics_timestep
        self.simulator = Simulator(mode=mode,
                                   gravity=0,
                                   timestep=physics_timestep,
                                   use_fisheye=self.config.get('fisheye', False),
                                   image_width=self.config.get('image_width', 128),
                                   image_height=self.config.get('image_height', 128),
                                   vertical_fov=self.config.get('vertical_fov', 90),
                                   device_idx=device_idx,
                                   render_to_tensor=render_to_tensor,
                                   auto_sync=False)
        self.simulator_loop = int(self.action_timestep / self.simulator.timestep)
        self.load()
        self.hand_start_pos = [0.05, 0, 1.05]
        self.hand_start_orn = p.getQuaternionFromEuler([-np.pi/2,0,np.pi])
        self.set_robot_pos_orn(self.robots[0], self.hand_start_pos, self.hand_start_orn)

        # load drawer
        self.drawer_start_pos = [0, 2.2, 1]
        self.drawer_start_orn = p.getQuaternionFromEuler([0,0,-np.pi/2])
        self.drawer = InteractiveObj(os.path.join(gibson2.assets_path, 'models', 'drawer', 'drawer_one_sided_handle.urdf'))
        self.simulator.import_articulated_object(self.drawer)
        self.drawer.set_position_orientation(self.drawer_start_pos, self.drawer_start_orn)
        self.simulator.sync()
        self._state_id = p.saveState()

    def set_robot_pos_orn(self, robot, pos, orn):
        robot.set_position_orientation(pos, orn)

    def get_state(self):
        return 0
    
    def get_reward(self):
        return 0

    def get_termination(self):
        return False

    def get_depth(self):
        """
        :return: depth sensor reading, normalized to [0.0, 1.0]
        """
        depth = -self.simulator.renderer.render_robot_cameras(modes=('3d'))[0][:, :, 2:3]
        # 0.0 is a special value for invalid entries
        depth[depth < self.depth_low] = 0.0
        depth[depth > self.depth_high] = 0.0

        # re-scale depth to [0.0, 1.0]
        depth /= self.depth_high
        return depth

    def get_rgb(self):
        """
        :return: RGB sensor reading, normalized to [0.0, 1.0]
        """
        return self.simulator.renderer.render_robot_cameras(modes=('rgb'))[0][:, :, :3]

    def get_pc(self):
        """
        :return: pointcloud sensor reading
        """
        return self.simulator.renderer.render_robot_cameras(modes=('3d'))[0]

    def get_normal(self):
        """
        :return: surface normal reading
        """
        return self.simulator.renderer.render_robot_cameras(modes='normal')

    def get_seg(self):
        """
        :return: semantic segmentation mask, normalized to [0.0, 1.0]
        """
        seg = self.simulator.renderer.render_robot_cameras(modes='seg')[0][:, :, 0:1]
        if self.num_object_classes is not None:
            seg = np.clip(seg * 255.0 / self.num_object_classes, 0.0, 1.0)
        return seg

    def load_task_setup(self):
        return
    
    def load_observation_space(self):
        return
    
    def load_action_space(self):
        self.action_space = self.robots[0].action_space

    def load_miscellaneous_variables(self):
        self.current_step = 0
        self.current_episode = 0

    def load(self):
        super(HandDrawerEnv, self).load()
        self.load_task_setup()
        self.load_observation_space()
        self.load_action_space()
        self.load_miscellaneous_variables()

    def reset_variables(self):
        self.current_episode += 1
        self.current_step = 0
    
    def reset(self):
        p.restoreState(self._state_id)
        self.simulator.sync()
        self.reset_variables()
        return self.get_state()

    def run_simulation(self):
        for _ in range(self.simulator_loop):
            self.simulator_step()
        self.simulator.sync()

    def step(self, action):
        self.current_step += 1
        if action is not None:
            self.robots[0].apply_action(action)
        self.run_simulation()

        state = self.get_state()
        info = {}
        reward = self.get_reward()
        done = self.get_termination()

        if done and self.automatic_reset:
            info['last_observation'] = state
            state = self.reset()
        return state, reward, done, info
