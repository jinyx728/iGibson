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

class SawyerDrawerOpenEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self, renders=False):
        self._renders = renders
        self._render_height = 768
        self._render_width = 1024
        self._physics_client_id = -1
        self._state_id = -1
        self._max_step = 1000
        self._current_step = 0

        # observation: 3D position of hand, 4D orientation of hand, 3D position of drawer
        self.observation_dim = 10
        hand_low=(-2, -2, -2)
        hand_high=(2, 2, 2)
        hand_quat_low = (0, -1, -1, -1)
        hand_quat_high = (2*np.pi, 1, 1, 1)
        obj_low=(-5, -5, -5)
        obj_high=(5, 5, 5)
        self.observation_space = spaces.Box(np.hstack((hand_low, hand_quat_low, obj_low)), np.hstack((hand_high, hand_quat_high, obj_high)))
        self.jd=[0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001]

        # action: 3D position of hand, 4D orientation of hand, 1D position of gripper
        self.action_dim = 8
        self.action_space = spaces.Box(np.array([-2, -2, -2, 0, -1, -1, -1, 0]), np.array([2, 2, 2, 2*np.pi, 1, 1, 1, 0.020833]),)

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

    def step(self, action):
        p = self._p
        # compute sawyer inverse kinematics
        pos_eff = action[0:3]
        quat_eff = action[3:7]
        pos_gripper = action[7]
        pos_new = p.calculateInverseKinematics(self.sawyer, 17, pos_eff, quat_eff, jointDamping=self.jd)
        for it in range(p.getNumJoints(self.sawyer)):
            jointInfo = p.getJointInfo(self.sawyer, it)
            qIndex = jointInfo[3]
            if qIndex > -1:
                p.setJointMotorControl2(self.sawyer, it, p.POSITION_CONTROL, pos_new[qIndex-7], force=300)
        p.setJointMotorControl2(bodyIndex=self.sawyer, jointIndex=18, controlMode=p.POSITION_CONTROL, targetPosition=pos_gripper, force=300)
        p.setJointMotorControl2(bodyIndex=self.sawyer, jointIndex=20, controlMode=p.POSITION_CONTROL, targetPosition=pos_gripper, force=300)
        p.stepSimulation()
        # compute state
        self.state = p.getLinkState(self.sawyer, 17)[0] + p.getLinkState(self.sawyer, 17)[1] + p.getLinkState(self.drawer, 3)[0]
        # compute done
        done = False
        self._current_step += 1
        if self._current_step > self._max_step:
            done = True
        # compute reward
        # handle_center = (np.array(p.getLinkState(self.drawer, 1)[0]) + np.array(p.getLinkState(self.drawer, 2)[0])) / 2
        handle_center = np.array(p.getLinkState(self.drawer, 3)[0])
        # grip_center = (np.array(p.getLinkState(self.sawyer, 18)[0]) + np.array(p.getLinkState(self.sawyer, 20)[0])) / 2
        grip_center = np.array(p.getLinkState(self.sawyer, 17)[0])
        reachDist = np.linalg.norm(np.array(grip_center - handle_center))
        #reachDist = np.linalg.norm(np.array(p.getLinkState(self.sawyer, 17)[0]) - np.array(p.getLinkState(self.drawer, 3)[0]))
        pullDist = np.abs(self.handle_max - p.getLinkState(self.drawer, 3)[0][1])
        reachRew = -reachDist
        if reachDist < 0.05:
            self.reachCompleted = True
        else:
            self.reachCompleted = False
        def pullReward():
            # if True:
            if self.reachCompleted:
                c1 = 1000 ; c2 = 0.01 ; c3 = 0.001
                # pullRew = 1000*(self.max_pull_dist - pullDist) + c1*(np.exp(-(pullDist**2)/c2) + np.exp(-(pullDist**2)/c3))
                pullRew = 1000*(self.max_pull_dist - pullDist)
                # pullRew = max(pullRew,0)
                return pullRew
            else:
                return 0
            return max(pullRew,0)
        pullRew = pullReward()
        reward = reachRew + pullRew
        return np.array(self.state), reward, done, {}

    def render(self, mode='human'):        
        # self._view_matrix = [0.5708255171775818, -0.6403688788414001, 0.5138930082321167, 0.0, 0.821071445941925, 0.4451974034309387, -0.3572688400745392, 0.0, -0.0, 0.6258810758590698, 0.7799185514450073, 0.0, -1.0701078176498413, -0.883043646812439, -1.8267910480499268, 1.0]
        # self._projection_matrix = [0.7499999403953552, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0, 0.0, 0.0, -0.02000020071864128, 0.0]

        self._view_matrix = [0.9999592304229736, -0.009023100137710571, 0.00025179231306537986, 0.0, 0.009026614017784595, 0.9995699524879456, -0.027893299236893654, 0.0, -0.0, 0.027894433587789536, 0.9996107220649719, 0.0, -0.12130190432071686, -0.6463457942008972, -2.645249843597412, 1.0]
        self._projection_matrix = [0.7499999403953552, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0, 0.0, 0.0, -0.02000020071864128, 0.0]

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
        #p.resetSimulation()
        if self._state_id < 0:
            # load plane
            planeId = p.loadURDF("plane.urdf")
            # load sawyer
            self.sawyerStartPos = [0.05, 0.5, 1.05]
            self.sawyerStartOrientation = p.getQuaternionFromEuler([0,0,np.pi/2])
            self.sawyer = p.loadURDF("/data/yxjin/robot/sawyer/sawyer_description/urdf/sawyer_with_gripper.urdf", basePosition=self.sawyerStartPos, baseOrientation=self.sawyerStartOrientation, useFixedBase=1)
            # set sawyer init pos
            pos_new = p.calculateInverseKinematics(self.sawyer, 17, [0,1,1], jointDamping=self.jd)
            for it in range(p.getNumJoints(self.sawyer)):
                jointInfo = p.getJointInfo(self.sawyer, it)
                qIndex = jointInfo[3]
                if qIndex > -1:
                    p.resetJointState(self.sawyer, it, pos_new[qIndex-7])
            # load drawer
            self.drawerStartPos = [0, 2.2, 1]
            self.drawerStartOrientation = p.getQuaternionFromEuler([0,0,-np.pi/2])
            self.drawer = p.loadURDF("/data/yxjin/robot/rotation/drawer_one_sided_handle.urdf", basePosition=self.drawerStartPos, baseOrientation=self.drawerStartOrientation, useFixedBase=1, flags=p.URDF_USE_SELF_COLLISION)
            self._state_id = p.saveState()
        else:
            p.restoreState(self._state_id)
        self.handle_origin = p.getLinkState(self.drawer, 3)[0][1]
        self.max_pull_dist = self.handle_origin - self.handle_max
        # compute init state
        self.state = p.getLinkState(self.sawyer, 17)[0] + p.getLinkState(self.sawyer, 17)[1] + p.getLinkState(self.drawer, 3)[0]
        return np.array(self.state)


    def close(self):
        if self._physics_client_id >= 0:
            self._p.disconnect()
            self._physics_client_id = -1
