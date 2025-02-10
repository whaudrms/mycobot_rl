import gym
import numpy as np
from gym import spaces
from utils.kinematic import Kinematics
from math import exp

class FKEnv(gym.Env):
    def __init__(self):
        super(FKEnv, self).__init__()

        # Forward Kinematics
        self.forward_kinematics = Kinematics().forward_kinematics
        
        self.num_joints = 6
        
        # action: 6개 관절 절대값 [-π, π]로 명령
        self.action_space = spaces.Box(
            low=-np.pi, high=np.pi, shape=(self.num_joints,), dtype=np.float32
        )
        
        # observation: [EE_x, EE_y, EE_z, target_x, target_y, target_z] 
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )
        
        self.max_steps = 128
        self.step_count = 0

        self.alpha = 5.0
        self.done = 0.02
        
    def reset(self):
        self.step_count = 0
        
        # 초기 관절각: 전부 0
        self.current_q = np.zeros(self.num_joints, dtype=np.float32)
        
        target_points = [
            np.array([-0.17,0,0.2], dtype=np.float32),
            np.array([-0.17,-0.17,0.2], dtype=np.float32),
            np.array([-0.17,0.17,0.2], dtype=np.float32)
        ]

        # 3개 중 무작위로 하나 선택
        self.target_pos = target_points[np.random.randint(0, len(target_points))]

        ee_pos = self.forward_kinematics(self.current_q)

        obs = np.concatenate([ee_pos, self.target_pos])
        return obs
    
    def step(self, action):
        self.step_count += 1
        
        # 여기서는 "절대 관절각" 모드로 
        # action을 그대로 current_q로 세팅
        self.current_q = np.clip(action, -np.pi, np.pi)
        
        ee_pos = self.forward_kinematics(self.current_q)
        pos_error = np.linalg.norm(ee_pos - self.target_pos)
        # reward_dist = (exp(-self.alpha*pos_error)-exp(-self.alpha)+10*(exp(-self.alpha*(pos_error/self.done))-exp(-self.alpha)))/(1-exp(-self.alpha))

        reward = -pos_error
        
        print(f"[Step {self.step_count}], reward: {reward:.4f}")


        # done 조건
        done = False
        if pos_error < 0.02:
            done = True
        if self.step_count >= self.max_steps:
            done = True
        
        # 관측 업데이트
        obs = np.concatenate([ee_pos, self.target_pos])
        
        info = {}
        
        return obs, reward, done, info
    
    def render(self, mode='human'):
        pass  # 여기서는 생략
    
    def close(self):
        pass
