from gibson2.envs.base_env import BaseEnv
from gibson2.utils.utils import parse_config
from gibson2.core.simulator import Simulator
from gibson2.core.physics.interactive_objects import InteractiveObj
from gibson2.core.render.mesh_renderer.mesh_renderer_cpu import quat2rotmat, xyz2mat, xyzw2wxyz
from collections import OrderedDict
from PIL import Image
import gibson2
import gym
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

        # class: stadium=0, robot=1, drawer=2, handle=3
        self.num_object_classes = 4
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
        self.handle_idx = 3
        self.drawer = InteractiveObj(os.path.join(gibson2.assets_path, 'models', 'drawer', 'drawer_one_sided_handle.urdf'))
        self.import_drawer_handle()
        self.drawer.set_position_orientation(self.drawer_start_pos, self.drawer_start_orn)
        self.handle_init_pos = p.getLinkState(self.drawer_id, self.handle_idx)[0][1]
        self.simulator.sync()
        self._state_id = p.saveState()

    """
    modify import_articulated_object() in simulator.py to
    render drawer and drawer handle separately for semantic inforamtion
    """
    def import_drawer_handle(self):
        self.drawer_id = self.drawer.load()

        # render drawer
        class_id = self.simulator.next_class_id
        self.simulator.next_class_id += 1
        visual_objects = []
        link_ids = []
        poses_rot = []
        poses_trans = []
        for shape in p.getVisualShapeData(self.drawer_id):
            id, link_id, type, dimensions, filename, rel_pos, rel_orn, color = shape[:8]
            link_name = p.getJointInfo(self.drawer_id, link_id)[12]
            if link_name != b'handle_left' and link_name != b'handle_right' and link_name != b'handle_grip':
                filename = os.path.join(gibson2.assets_path, 'models/mjcf_primitives/cube.obj')
                self.simulator.renderer.load_object(filename,
                                          transform_orn=rel_orn,
                                          transform_pos=rel_pos,
                                          input_kd=color[:3],
                                          scale=np.array(dimensions))
                visual_objects.append(len(self.simulator.renderer.visual_objects) - 1)
                link_ids.append(link_id)
                _, _, _, _, pos, orn = p.getLinkState(id, link_id)
                poses_rot.append(np.ascontiguousarray(quat2rotmat(xyzw2wxyz(orn))))
                poses_trans.append(np.ascontiguousarray(xyz2mat(pos)))
        self.simulator.renderer.add_instance_group(object_ids=visual_objects,
                                         link_ids=link_ids,
                                         pybullet_uuid=self.drawer_id,
                                         class_id=class_id,
                                         poses_rot=poses_rot,
                                         poses_trans=poses_trans,
                                         dynamic=True,
                                         robot=None)

        # render drawer handle
        class_id = self.simulator.next_class_id
        self.simulator.next_class_id += 1
        visual_objects = []
        link_ids = []
        poses_rot = []
        poses_trans = []
        for shape in p.getVisualShapeData(self.drawer_id):
            id, link_id, type, dimensions, filename, rel_pos, rel_orn, color = shape[:8]
            link_name = p.getJointInfo(self.drawer_id, link_id)[12]
            if link_name == b'handle_left' or link_name == b'handle_right' or link_name == b'handle_grip':
                filename = os.path.join(gibson2.assets_path, 'models/mjcf_primitives/cube.obj')
                self.simulator.renderer.load_object(filename,
                                          transform_orn=rel_orn,
                                          transform_pos=rel_pos,
                                          input_kd=color[:3],
                                          scale=np.array(dimensions))
                visual_objects.append(len(self.simulator.renderer.visual_objects) - 1)
                link_ids.append(link_id)
                _, _, _, _, pos, orn = p.getLinkState(id, link_id)
                poses_rot.append(np.ascontiguousarray(quat2rotmat(xyzw2wxyz(orn))))
                poses_trans.append(np.ascontiguousarray(xyz2mat(pos)))
        self.simulator.renderer.add_instance_group(object_ids=visual_objects,
                                         link_ids=link_ids,
                                         pybullet_uuid=self.drawer_id,
                                         class_id=class_id,
                                         poses_rot=poses_rot,
                                         poses_trans=poses_trans,
                                         dynamic=True,
                                         robot=None)

    def set_robot_pos_orn(self, robot, pos, orn):
        robot.set_position_orientation(pos, orn)

    def get_drawer_handle_pos(self):
        return np.array(p.getLinkState(self.drawer_id, self.handle_idx)[0])

    def get_state(self):
        state = OrderedDict()
        if 'rgb' in self.output:
            state['rgb'] = self.get_rgb()
        if 'depth' in self.output:
            state['depth'] = self.get_depth()
        if 'normal' in self.output:
            state['normal'] = self.get_normal()
        if 'seg' in self.output:
            state['seg'] = self.get_seg()
        return state
    
    def get_reward(self):
        handle_center = self.get_drawer_handle_pos()
        grip_center = self.robots[0].get_palm_position()
        reach_dist = np.linalg.norm(grip_center - handle_center)
        pull_dist = self.handle_init_pos - handle_center[1]
        reach_rew = -self.reach_reward_factor * reach_dist
        if reach_dist < 0.05:
            self.reach_completed = True
        else:
            self.reach_completed = False
        def pull_reward():
            if self.reach_completed:
                pull_rew = self.pull_reward_factor * pull_dist
                return pull_rew
            else:
                return 0
        pull_rew = pull_reward()
        reward = reach_rew + pull_rew
        return reward

    def is_goal_pulled(self):
        epsilon = 0.01
        return (self.handle_init_pos - self.get_drawer_handle_pos()[1]) > self.max_pull_dist - epsilon

    def get_termination(self, info={}):
        done = False

        if self.is_goal_pulled() and self.reach_completed:
            done = True
            info['success'] = True
        elif self.current_step >= self.max_step:
            done = True
            info['success'] = False
        
        if done:
            info['episode_length'] = self.current_step
        
        return done, info

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

    def get_normal(self):
        """
        :return: surface normal reading
        """
        return self.simulator.renderer.render_robot_cameras(modes='normal')[0][:, :, :3]

    def get_seg(self):
        """
        :return: semantic segmentation mask, normalized to [0.0, 1.0]
        """
        seg = self.simulator.renderer.render_robot_cameras(modes='seg')[0][:, :, 0:1]
        if self.num_object_classes is not None:
            seg = np.clip(seg * 255.0 / self.num_object_classes, 0.0, 1.0)
        return seg

    def save_rgb_image(self, path):
        rgb = self.get_rgb()
        Image.fromarray((rgb * 255).astype(np.uint8)).save(path)

    def save_depth_image(self, path):
        depth = self.get_depth()
        Image.fromarray((depth * 255).astype(np.uint8)[:,:,0]).save(path)
        
    def save_seg_image(self, path):
        seg = self.get_seg()
        Image.fromarray((seg * 255).astype(np.uint8)[:,:,0]).save(path)

    def save_normal_image(self, path):
        normal = self.get_normal()
        Image.fromarray((normal * 255).astype(np.uint8)).save(path)

    def load_task_setup(self):
        self.max_pull_dist = 0.5
        self.reach_reward_factor = 1.0
        self.pull_reward_factor = 1000.0
        self.max_step = self.config.get('max_step', 500)
    
    def load_observation_space(self):
        self.output = self.config['output']
        self.image_width = self.config.get('image_width', 128)
        self.image_height = self.config.get('image_height', 128)
        observation_space = OrderedDict()
        if 'rgb' in self.output:
            self.rgb_space = gym.spaces.Box(low=0.0,
                                            high=1.0,
                                            shape=(self.image_height, self.image_width, 3),
                                            dtype=np.float32)
            observation_space['rgb'] = self.rgb_space
        if 'depth' in self.output:
            self.depth_low = self.config.get('depth_low', 0.5)
            self.depth_high = self.config.get('depth_high', 5.0)
            self.depth_space = gym.spaces.Box(low=0.0,
                                              high=1.0,
                                              shape=(self.image_height, self.image_width, 1),
                                              dtype=np.float32)
            observation_space['depth'] = self.depth_space
        if 'seg' in self.output:
            self.seg_space = gym.spaces.Box(low=0.0,
                                            high=1.0,
                                            shape=(self.image_height, self.image_width, 1),
                                            dtype=np.float32)
            observation_space['seg'] = self.seg_space
        if 'normal' in self.output:
            self.normal_space = gym.spaces.Box(low=0.0,
                                            high=1.0,
                                            shape=(self.image_height, self.image_width, 3),
                                            dtype=np.float32)
            observation_space['normal'] = self.normal_space

        self.observation_space = gym.spaces.Dict(observation_space)

    
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
        done, info = self.get_termination()

        if done and self.automatic_reset:
            info['last_observation'] = state
            state = self.reset()
        return state, reward, done, info
