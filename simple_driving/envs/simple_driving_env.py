import gym
import numpy as np
import math
import pybullet as p
from simple_driving.resources.car import Car
from simple_driving.resources.plane import Plane
import matplotlib.pyplot as plt


class SimpleDrivingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        '''
        # 動作空間
        # 第一個維度:油門 [0,1]
        # 第二個維度:方向盤 [-0.9,0.9]
        '''
        self.action_space = gym.spaces.box.Box(
            low=np.array([0, -0.9], dtype=np.float32),
            high=np.array([1, 0.9], dtype=np.float32))
        '''
        # 觀察空間
        # 索引[0,1] :車的xy座標[-10,10]
        # 索引[2,3] :車的xy方向[-1,1]
        # 索引[4,5] :車的xy加速度[-5,5]
        '''
        self.observation_space = gym.spaces.box.Box(
            low=np.array([-10, -10, -1, -1, -5, -5], dtype=np.float32),
            high=np.array([10, 10, 1, 1, 5, 5], dtype=np.float32))

        self.np_random, _ = gym.utils.seeding.np_random()

        # 選擇連結方式
        #self.client = p.connect(p.DIRECT)
        self.client = p.connect(p.GUI)
        
        # 加速訓練
        p.setTimeStep(1/30, self.client)

        # 初始化所有東西
        self.car = None
        self.done = False
        self.rendered_img = None
        self.render_rot_matrix = None
        self.reset()

    def step(self, action):
        reward = 0
        reward += 1
        self.car.apply_action(action)
        p.stepSimulation()
        car_ob = self.car.get_observation()
        pos = car_ob[:2]
        pos_x ,pos_y = pos[0] ,pos[1]
        
        # 掉落平台則遊戲結束且受到負reward
        if (car_ob[0] >= 10 or car_ob[0] <= -10 or car_ob[1] >= 10 or car_ob[1] <= -10):
            self.done = True
            reward -= 9999
        
        ob = np.array(car_ob , dtype=np.float32)
        info = {}
        
        return ob, reward, self.done, info

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        # 初始化所有東西
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -10)
        Plane(self.client)
        self.car = Car(self.client)
        self.done = False
        car_ob = self.car.get_observation()
        return np.array(car_ob,dtype=np.float32)

    def render(self, mode='human'):
        if self.rendered_img is None:
            self.rendered_img = plt.imshow(np.zeros((100, 100, 4)))
        # 基本資訊
        car_id, client_id = self.car.get_ids()
        proj_matrix = p.computeProjectionMatrixFOV(fov=80, aspect=1,nearVal=0.01, farVal=100)
        pos, ori = [list(l) for l in p.getBasePositionAndOrientation(car_id, client_id)]
        pos[2] = 0.2
        # 旋轉照相機方向
        rot_mat = np.array(p.getMatrixFromQuaternion(ori)).reshape(3, 3)
        camera_vec = np.matmul(rot_mat, [1, 0, 0])
        up_vec = np.matmul(rot_mat, np.array([0, 0, 1]))
        view_matrix = p.computeViewMatrix(pos, pos + camera_vec, up_vec)
        # 展示圖片
        frame = p.getCameraImage(100, 100, view_matrix, proj_matrix)[2]
        frame = np.reshape(frame, (100, 100, 4))
        self.rendered_img.set_data(frame)
        plt.draw()
        plt.pause(0.00001)
    
    def close(self):
        p.disconnect(self.client)
