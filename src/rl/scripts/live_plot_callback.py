# live_plot_callback.py
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback

class LivePlotCallback(BaseCallback):
    def __init__(self, check_freq=1, verbose=0):
        super(LivePlotCallback, self).__init__(verbose)
        self.check_freq = check_freq
        
        # 에피소드별 리턴 기록용
        self.episode_rewards = []
        self.current_reward = 0.0
        self.episode_counts = []
        self.ep_count = 0
        
        # matplotlib 설정
        plt.ion()  # 인터랙티브 모드 ON
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], 'b-')
        self.ax.set_xlabel('Steps')
        self.ax.set_ylabel('Return')
        self.ax.set_title('Real-Time Episode Return')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _on_step(self) -> bool:
        # 이 스텝에서 얻은 reward (배치 학습 시 보통 배열인데, 단일 환경이면 scalar)
        reward = self.locals['rewards'][0]  # 여러 환경(병렬 VecEnv)이면 인덱스 조정
        self.current_reward += reward
        
        # done 체크
        done = self.locals['dones'][0]
        if done:
            self.ep_count += 1
            self.episode_rewards.append(self.current_reward)
            self.episode_counts.append(self.ep_count)
            
            self.current_reward = 0.0
            
            # 그래프 업데이트
            if self.ep_count % self.check_freq == 0:
                self._update_plot()
        return True

    def _update_plot(self):
        # x: episode_counts, y: episode_rewards
        x_data = np.array(self.episode_counts)
        y_data = np.array(self.episode_rewards)

        self.line.set_xdata(x_data)
        self.line.set_ydata(y_data)

        # 축 범위 자동 조정
        self.ax.relim()
        self.ax.autoscale_view()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
