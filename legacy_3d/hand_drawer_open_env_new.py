import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import subprocess
import pybullet as p2
import pybullet_data
import pybullet_utils.bullet_client as bc
from pkg_resources import parse_version

logger = logging.getLogger(__name__)

class HandDrawerOpenEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self, renders=False):
        self._renders = renders
        self._render_height = 768
        self._render_width = 1024
        self._physics_client_id = -1
        self._state_id = -1
        self._max_step = 1000
        self._current_step = 0

        self.joint_id_list = [0, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 17, 20, 21, 22, 23, 25, 26, 27, 28, 29]
        self.jointStartPos = np.zeros(len(self.joint_id_list), dtype=np.float64)
        self.palm_idx = 3
        self.tips_idx = [8, 13, 18, 24, 30]
        self.handle_idx = 3

        # observation space
        self.observation_dim = 67
        self.observation_bound = 3
        self.observation_high = np.array([self.observation_bound] * self.observation_dim)
        self.observation_space = spaces.Box(-self.observation_high, self.observation_high, dtype=np.float64)

        # action space
        self.action_dim = len(self.joint_id_list) + 7
        self.action_bound = 2
        self.action_high = np.array([self.action_bound] * self.action_dim)
        self.action_space = spaces.Box(-self.action_high, self.action_high, dtype=np.float64)

        # drawer handle specific
        self.handle_origin = 1.36
        self.handle_max = 0.9
        self.max_pull_dist = 0.46

        self.seed()
        self.viewer = None
        self._configure()

    def _configure(self, display=None):
        self.display = display

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_obs(self):
        p = self._p
        obs = []
        links = [self.palm_idx] + self.tips_idx
        # 3D Cartesian position and 4D Cartesian orientation for palm and fingertips
        for link in links:
            lpos = p.getLinkState(self.dex_hand, link)[0]
            lorn = p.getLinkState(self.dex_hand, link)[1]
            obs.extend(lpos)
            obs.extend(lorn)
        # joint position in reduced coordinate for all DOFS
        for joint in self.joint_id_list:
            jpos = p.getJointState(self.dex_hand, joint)[0]
            obs.append(jpos)
        # 1D position of drawer handle
        obs.append(p.getLinkState(self.drawer, self.handle_idx)[0][1])
        return np.array(obs)

    def _apply_action(self, action):
        p = self._p
        jointPos = np.array(action[ : len(self.joint_id_list)])
        jointPos = np.clip(jointPos, self.ll, self.ul)
        handPos = np.array(action[-7:-4])
        handOrientation = np.array(action[-4:])
        p.changeConstraint(self.hand_constraint, jointChildPivot=handPos, jointChildFrameOrientation=handOrientation, maxForce=500)
        for it in range(len(self.joint_id_list)):
            p.setJointMotorControl2(self.dex_hand, self.joint_id_list[it], p.POSITION_CONTROL, jointPos[it], force=500)

    def _compute_reward(self):
        p = self._p
        handle_center = np.array(p.getLinkState(self.drawer, self.handle_idx)[0])
        grip_center = np.array(p.getLinkState(self.dex_hand, self.palm_idx)[0])
        reachDist = np.linalg.norm(np.array(grip_center - handle_center))
        pullDist = np.abs(self.handle_max - p.getLinkState(self.drawer, self.handle_idx)[0][1])
        reachRew = -reachDist
        if reachDist < 0.05:
            self.reachCompleted = True
        else:
            self.reachCompleted = False
        def pullReward():
            if self.reachCompleted:
                # c1 = 1000 ; c2 = 0.01 ; c3 = 0.001
                # pullRew = 1000*(self.max_pull_dist - pullDist) + c1*(np.exp(-(pullDist**2)/c2) + np.exp(-(pullDist**2)/c3))
                pullRew = 1000*(self.max_pull_dist - pullDist)
                # pullRew = max(pullRew,0)
                return pullRew
            else:
                return 0
            return max(pullRew,0)
        pullRew = pullReward()
        reward = reachRew + pullRew
        return reward

    def step(self, action):
        # apply action and simulate
        self._apply_action(action)
        self._p.stepSimulation()
        # compute observation
        self.state = self._get_obs()
        # compute done
        done = False
        self._current_step += 1
        if self._current_step > self._max_step:
            done = True
        # compute reward
        reward = self._compute_reward()
        return self.state, reward, done, {}

    def render(self, mode='human'):
        # normal view
        self._view_matrix = [0.5708255171775818, -0.6403688788414001, 0.5138930082321167, 0.0, 0.821071445941925, 0.4451974034309387, -0.3572688400745392, 0.0, -0.0, 0.6258810758590698, 0.7799185514450073, 0.0, -1.0701078176498413, -0.883043646812439, -1.8267910480499268, 1.0]
        self._projection_matrix = [0.7499999403953552, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0, 0.0, 0.0, -0.02000020071864128, 0.0]

        # top down view
        # self._view_matrix = [0.9999592304229736, -0.009023100137710571, 0.00025179231306537986, 0.0, 0.009026614017784595, 0.9995699524879456, -0.027893299236893654, 0.0, -0.0, 0.027894433587789536, 0.9996107220649719, 0.0, -0.12130190432071686, -0.6463457942008972, -2.645249843597412, 1.0]
        # self._projection_matrix = [0.7499999403953552, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0, 0.0, 0.0, -0.02000020071864128, 0.0]

        if mode == "human":
            self._renders = True
        if mode != "rgb_array":
            return np.array([])
        if self._physics_client_id >= 0:
            (_, _, px, _, _) = self._p.getCameraImage(width=self._render_width, height=self._render_height, renderer=self._p.ER_TINY_RENDERER, viewMatrix=self._view_matrix, projectionMatrix=self._projection_matrix)
        else:
            px = np.array([[[255,255,255,255]]*self._render_width]*self._render_height, dtype=np.uint8)
        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(np.array(px), (self._render_height, self._render_width, -1))
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def reset(self):
        self._current_step = 0
        if self._physics_client_id < 0:
            if self._renders:
                self._p = bc.BulletClient(connection_mode=p2.GUI)
            else:
                self._p = bc.BulletClient()

            self._physics_client_id = self._p._client
            self._p.setAdditionalSearchPath(pybullet_data.getDataPath())

        p = self._p
        if self._state_id < 0:
            # load plane
            planeId = p.loadURDF("plane.urdf")
            # load hand
            self.handStartPos = [0.05, 0, 1.05]
            self.handStartOrientation = p.getQuaternionFromEuler([-np.pi/2,0,np.pi])
            self.dex_hand = p.loadURDF("/data/yxjin/robot/inmoov_ros/inmoov_description/robots/inmoov_shadow_hand_v2_3.urdf")
            p.resetBasePositionAndOrientation(self.dex_hand, self.handStartPos, self.handStartOrientation)
            self.hand_constraint = p.createConstraint(self.dex_hand, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], self.handStartPos, p.getQuaternionFromEuler([0,0,0]), self.handStartOrientation)
            self.ll = np.array([p.getJointInfo(self.dex_hand, joint)[8] for joint in self.joint_id_list])
            self.ul = np.array([p.getJointInfo(self.dex_hand, joint)[9] for joint in self.joint_id_list])
            # load drawer
            self.drawerStartPos = [0, 2.2, 1]
            self.drawerStartOrientation = p.getQuaternionFromEuler([0,0,-np.pi/2])
            self.drawer = p.loadURDF("/data/yxjin/robot/rotation/drawer_one_sided_handle.urdf", useFixedBase=1, flags=p.URDF_USE_SELF_COLLISION)
            p.resetBasePositionAndOrientation(self.drawer, self.drawerStartPos, self.drawerStartOrientation)
            self._state_id = p.saveState()
        else:
            p.restoreState(self._state_id)
        self.handle_origin = p.getLinkState(self.drawer, self.handle_idx)[0][1]
        self.max_pull_dist = self.handle_origin - self.handle_max
        # compute init state
        self.state = self._get_obs()
        return self.state


    def close(self):
        if self._physics_client_id >= 0:
            self._p.disconnect()
            self._physics_client_id = -1
