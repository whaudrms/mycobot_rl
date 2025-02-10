from stable_baselines3 import PPO
from env.kinematic_env import FKEnv
from live_plot_callback import LivePlotCallback
import time

def main():    
    # 환경 초기화
    env = FKEnv()
    
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

    new_lr = 1e-4  # 원하는 낮은 러닝 레이트 값
    for param_group in model.policy.optimizer.param_groups:
        param_group['lr'] = new_lr
    
    # 실시간 모니터링을 위한 콜백 (에피소드마다 업데이트)
    callback = LivePlotCallback(check_freq=1)
    
    try:
        # 총 타임스텝 수는 환경 및 실험에 맞게 조정하세요.
        model.learn(total_timesteps=100000, callback=callback ,tb_log_name="FK")
    except KeyboardInterrupt:
        # Ctrl+C 입력 시 중간 저장
        print("KeyboardInterrupt detected. Saving the current model...")
        model.save("FK_ppo")
    
    # 학습이 완료되었을 때 모델 저장
    print("Saving the current model...")
    model.save("FK_ppo")

    time.sleep(10.0)

if __name__ == "__main__":
    main()