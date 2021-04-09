import pybullet as p
import os
import math


class Car:
    def __init__(self, client):
        self.client = client
        f_name = os.path.join(os.path.dirname(__file__), 'simplecar.urdf')
        self.car = p.loadURDF(fileName=f_name,basePosition=[0, 0, 0.1],physicsClientId=client)
        self.steering_joints = [0, 2] # 方向盤控制[左右旋轉]
        self.drive_joints = [1, 3, 4, 5] # 馬達控制[四顆馬達]
        self.joint_speed = 0 # joint速度
        self.c_rolling = 0.2 # 阻力常數
        self.c_drag = 0.01 # 阻力常數
        self.c_throttle = 1 # 摧一次油門增加的速度

    def get_ids(self):
        return self.car, self.client

    def apply_action(self, action):
        throttle, steering_angle = action
        # Clip 到合理範圍
        throttle = min(max(throttle, 0), 1)
        steering_angle = max(min(steering_angle, 0.9), -0.9)
        # 調整方向盤
        p.setJointMotorControlArray(self.car, self.steering_joints,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=[steering_angle] * 2,
                                    physicsClientId=self.client)

        # 計算摩擦力
        friction = -self.joint_speed * (self.joint_speed * self.c_drag + self.c_rolling)
        
        # 計算加速度
        acceleration = self.c_throttle * throttle + friction
        
        # 每個 time_step 設置為 1/240 秒
        self.joint_speed = self.joint_speed + 1/30 * acceleration
        if self.joint_speed < 0:
            self.joint_speed = 0

        # 轉動輪胎
        p.setJointMotorControlArray(
            bodyUniqueId=self.car,
            jointIndices=self.drive_joints,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=[self.joint_speed] * 4,
            forces=[1.2] * 4,
            physicsClientId=self.client)

    def get_observation(self):
        # 取得位置跟角度
        pos, ang = p.getBasePositionAndOrientation(self.car, self.client)
        ang = p.getEulerFromQuaternion(ang)
        # 角度轉成方向
        ori = (math.cos(ang[2]), math.sin(ang[2]))
        # 取得xy座標
        pos = pos[:2]
        # 取得車子加速度
        vel = p.getBaseVelocity(self.car, self.client)[0][0:2]
        # 串街 位置, 方向, 加速度 當作 observation
        observation = (pos + ori + vel)
        return observation









