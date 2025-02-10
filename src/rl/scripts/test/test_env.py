# test_env.py

import gym
from rl.scripts.env.mycobot_env import MyCobotEnv

def main():
    env = MyCobotEnv()
    obs = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        # 무작위 action
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        total_reward += reward
        env.render()

    print("Episode finished, total reward:", total_reward)

if __name__ == "__main__":
    main()
