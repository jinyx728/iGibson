import pybullet as p
import time
import pybullet_data
import numpy as np
import math

physicsId = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,0)

# load plane
planeId = p.loadURDF("plane.urdf")

# load hand
sawyerStartPos = [0.05, 0.5, 1.05]
sawyerStartOrientation = p.getQuaternionFromEuler([0,0,0])
sawyer = p.loadURDF("/Users/kevinjyx/Downloads/sawyer/sawyer_description/urdf/gripper.urdf", basePosition=sawyerStartPos, baseOrientation=sawyerStartOrientation, useFixedBase=1)
numJoints = p.getNumJoints(sawyer)


# print joint index and name of hand
for i in range(p.getNumJoints(sawyer)):
    print(p.getJointInfo(sawyer, i)[0], p.getJointInfo(sawyer, i)[1].decode("utf-8") )

# load drawer
drawerStartPos = [0, 2, 1]
drawerStartOrientation = p.getQuaternionFromEuler([0,0,-np.pi/2])
drawerId = p.loadURDF("/Users/kevinjyx/Downloads/rotation/drawer_one_sided_handle.urdf", basePosition=drawerStartPos, baseOrientation=drawerStartOrientation, useFixedBase=1)

# print joint index and name of drawer
# for i in range(p.getNumJoints(drawerId)):
#     print(p.getJointInfo(drawerId, i)[0], p.getJointInfo(drawerId, i)[12].decode("utf-8") )

jd=[0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001]
t=0
for i in range (10000):
    # t += 1./10.
    # for i in range (1):
    #     pos = [1.05,0.5 + 0.2*math.cos(t),1.05+0.2*math.sin(t)]
    #     jointPoses = p.calculateInverseKinematics(sawyer,16,pos,jointDamping=jd)
        
    #     for i in range (numJoints):
    #         jointInfo = p.getJointInfo(sawyer, i)
    #         qIndex = jointInfo[3]
    #         if qIndex > -1:
    #             p.setJointMotorControl2(bodyIndex=sawyer,
    #                             jointIndex=i,
    #                             controlMode=p.POSITION_CONTROL,
    #                             targetPosition=jointPoses[qIndex-7],
    #                             force=500)
    # p.setJointMotorControl2(bodyIndex=sawyer,
    #                             jointIndex=18,
    #                             controlMode=p.POSITION_CONTROL,
    #                             targetPosition=0.020833,
    #                             force=500)
    # p.setJointMotorControl2(bodyIndex=sawyer,
    #                             jointIndex=20,
    #                             controlMode=p.POSITION_CONTROL,
    #                             targetPosition=0.020833,
    #                             force=500)


    p.stepSimulation()
    time.sleep(1./240.)
p.disconnect()

