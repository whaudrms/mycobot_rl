import rospy
from stable_baselines3 import PPO
from env.mycobot import MyCobotEnv
from live_plot_callback import LivePlotCallback

def main():    
    # 환경 초기화
    env = MyCobotEnv()
    
    # PPO 모델 초기화 (추천 하이퍼파라미터 적용)
    # model = PPO(
    #     "MlpPolicy",
    #     env,
    #     verbose=1,
    #     tensorboard_log="./tensorboard_logs",
    #     learning_rate=3e-4,
    #     n_steps=2048,
    #     batch_size=64,
    #     gamma=0.99,
    #     clip_range=0.2,
    #     ent_coef=0.0
    # )

    # 만약 저장된 모델을 불러오고 싶다면 아래 주석을 해제하여 사용하세요.
    model = PPO.load("/home/tony/mycobot_rl/src/FK_ppo.zip", env=env)
    
    # 실시간 모니터링을 위한 콜백 (에피소드마다 업데이트)
    callback = LivePlotCallback(check_freq=1)
    
    try:
        # 총 타임스텝 수는 환경 및 실험에 맞게 조정하세요.
        model.learn(total_timesteps=100000, callback=callback, tb_log_name="sim")
    except KeyboardInterrupt:
        # Ctrl+C 입력 시 중간 저장
        print("KeyboardInterrupt detected. Saving the current model...")
        model.save("sim_ppo")
    
    # 학습이 완료되었을 때 모델 저장
    print("Saving the current model...")
    model.save("sim_ppo")

if __name__ == "__main__":
    main()