#!/usr/bin/env python3

import gym
import numpy as np
from gym import spaces

import rospy
import tf2_ros
import tf.transformations as tft
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from utils.target_spawner import TargetSpawner
from math import exp

class MyCobotEnv(gym.Env):
    def __init__(self):
        """
        - Action: 6개의 관절 목표각 (라디안) → /arm_controller/command 퍼블리시
        - 관찰(Observation): [target_x, target_y, target_z, endeff_x, endeff_y, endeff_z]
          여기서 endeff_x,y,z는 tf에서 'world'→'link_flange' transform을 lookup하여 얻음
        """
        super(MyCobotEnv, self).__init__()

        # target 스포너
        self.spawner = TargetSpawner().spawn_target_model
        
        # ROS 초기화
        rospy.init_node('mycobot_env_tf', anonymous=True)
        
        # 로봇 관절 제어 퍼블리셔
        self.pub_cmd = rospy.Publisher('/arm_controller/command', JointTrajectory, queue_size=1)
        
        # TF 버퍼 & 리스너
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Action Space: 6개 관절각 [-π, π]
        self.action_space = spaces.Box(
            low=-np.pi,
            high= np.pi,
            shape=(6,),
            dtype=np.float32
        )
        
        # Observation Space: 6차원 (엔드이펙터 3 + 목표 3)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(6,),
            dtype=np.float32
        )
        
         # 내부 상태
        self.endeff_pose = np.zeros(6, dtype=np.float32)  # [x,y,z,qx,qy,qz,qw]
        self.target_pose = np.zeros(6, dtype=np.float32)  # [x,y,z,qx,qy,qz,qw]
        
        self.max_steps = 100
        self.step_count = 0
        
        self.rate = rospy.Rate(10)  # 10Hz, 필요 시 조정
        
        rospy.loginfo("MyCobotEnv using TF for end-effector position initialized.")

    def reset(self):
        """
        - 에피소드 초기화:
            1) 목표 pose 생성: x,y ∈ [-0.15, 0.15], z ∈ [1.02, 1.3] / orientation: 무작위 Euler 각
            2) 타겟 스폰 (TargetSpawner 사용)
            3) 로봇 초기화: 모든 관절 0도 (홈 포지션)
            4) TF에서 엔드이펙터 pose (translation + quaternion) 조회
            5) 이전 오차 초기화
            6) 관측(obs) 반환
        """
        self.step_count = 0
        
        target_points = [
            np.array([-0.17,0,0.2], dtype=np.float32),
            np.array([-0.17,-0.17,0.2], dtype=np.float32),
            np.array([-0.17,0.17,0.2], dtype=np.float32)
        ]

        # 3개 중 무작위로 하나 선택
        self.target_pose = target_points[np.random.randint(0, len(target_points))]
        self.spawner("my_target_cube", self.target_pose[0], self.target_pose[1], self.target_pose[2])

        # 2) 로봇 초기화: 모든 관절 0도
        home_joints = np.zeros(6, dtype=np.float32)
        self._send_joint_command(home_joints, duration=2.0)
        rospy.sleep(2.0)  # 이동 완료 대기
        
        # 3) TF 업데이트 대기 및 엔드이펙터 pose 조회
        rospy.sleep(0.5)
        current_translation = self._lookup_end_effector()      # [x,y,z]
        self.endeff_pose = current_translation
        
        # 6) 관측 반환: [endeff_pose, target_pose] (총 6차원)
        obs = np.concatenate([self.endeff_pose, self.target_pose]).astype(np.float32)
        return obs

    def step(self, action):
        """
        - action: 6개 관절 목표각
        - 관절 명령 퍼블리시 → tf로 엔드이펙터 pose 조회 → 보상 계산 (위치+orientation)
        """
        self.step_count += 1
        
        ## 1) JointTrajectory 명령
        self._send_joint_command(action, duration=1.0)
        rospy.sleep(2.0)  # 이동 대기
        rospy.sleep(0.1)  # TF 갱신 대기
        
        ## 2) 엔드이펙터 pose 조회 (translation + quaternion)
        current_translation = self._lookup_end_effector()
        self.endeff_pose = current_translation
        
        ## 3) 관측
        obs = np.concatenate([self.endeff_pose, self.target_pose]).astype(np.float32)

        ## 4) 보상 계산
        # 위치 오차
        pos_error = np.linalg.norm(self.target_pose[:3] - self.endeff_pose[:3])
        # reward_dist = (exp(-self.alpha*pos_error)-exp(-self.alpha)+10*(exp(-self.alpha*(pos_error/self.done))-exp(-self.alpha)))/(1-exp(-self.alpha))

        reward = -pos_error

        
        print(f"[Step {self.step_count}], reward: {reward:.4f}")
        
        # 5) done 조건
        done = False
        if pos_error < 0.05:  
            done = True
            print("Episode : Successfully Done!")
        if self.step_count >= self.max_steps:
            done = True

        info = {}
        return obs, reward, done, info

    def render(self, mode='human'):
        # print(f"[step={self.step_count}] target_pose={self.target_pose}, endeff_pose={self.endeff_pose}")
        pass

    def close(self):
        pass

    # -----------------------------------------------------
    #                       헬퍼 메서드
    # -----------------------------------------------------


    def _send_joint_command(self, joint_targets, duration=2.0):
        traj = JointTrajectory()
        traj.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        point = JointTrajectoryPoint()
        point.positions = joint_targets
        point.time_from_start = rospy.Duration(duration)
        traj.points.append(point)
        self.pub_cmd.publish(traj)
        # rospy.loginfo(f"Published joint command: {joint_targets}, duration={duration}")

    def _lookup_end_effector(self):
        """
        - TF에서 'world' 프레임 대비 'link6_flange' (엔드이펙터)의 translation을 lookup
        - 반환: [x, y, z]
        """
        try:
            trans = self.tf_buffer.lookup_transform(
                'world',
                'link6_flange',
                rospy.Time(0),
                rospy.Duration(1.0)
            )
            tx = trans.transform.translation.x
            ty = trans.transform.translation.y
            tz = trans.transform.translation.z
            # print(tx,ty,tz)
            return np.array([tx, ty, tz], dtype=np.float32)
        except Exception as e:
            rospy.logwarn(f"Failed to lookup TF (translation): {e}")
            return np.zeros(3, dtype=np.float32)
