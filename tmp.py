import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

# 1. 가변 길이 상태를 갖는 환경 정의
class VariableLengthStateEnv(gym.Env):
    def __init__(self):
        super(VariableLengthStateEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(2)  # 두 개의 행동: 0 또는 1
        self.observation_space = gym.spaces.Box(low=0, high=100, shape=(10,), dtype=np.float32)
        self.state = []
        self.timestep = 0

    def step(self, action):
        self.timestep += 1
        # 상태는 길이가 가변적인 리스트로 구성
        self.state.append(random.random() * 100)  # 새로운 요소 추가
        if len(self.state) > 10:
            self.state.pop(0)  # 최대 길이를 10으로 제한

        # 보상 함수: 상태의 평균값이 50에 가까울수록 높은 보상
        reward = -abs(np.mean(self.state) - 50)
        done = self.timestep >= 100  # 100 스텝 후 에피소드 종료
        return np.array(self.state, dtype=np.float32), reward, done, {}

    def reset(self):
        self.state = [random.random() * 100]  # 초기 상태는 하나의 요소를 가진 리스트
        self.timestep = 0
        return np.array(self.state, dtype=np.float32)

    def render(self, mode="human"):
        print(f"State: {self.state}, Timestep: {self.timestep}")

# 2. LSTM 기반의 정책 신경망 정의
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 입력 x의 크기를 (배치 크기, 시퀀스 길이, 입력 크기)로 맞춤
        x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, input_size)
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])
        return torch.softmax(output, dim=-1)

# 3. 강화학습 루프
def train():
    env = VariableLengthStateEnv()
    input_size = 1  # 각 상태 요소는 스칼라 값
    hidden_size = 128
    output_size = env.action_space.n

    policy = PolicyNetwork(input_size, hidden_size, output_size)
    optimizer = optim.Adam(policy.parameters(), lr=0.01)
    gamma = 0.99  # 할인율

    for episode in range(100):
        state = env.reset()
        log_probs = []
        rewards = []

        for t in range(100):
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action_probs = policy(state_tensor)
            action = np.random.choice(np.arange(output_size), p=action_probs.detach().numpy())
            log_prob = torch.log(action_probs[action])

            next_state, reward, done, _ = env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)

            state = next_state
            if done:
                break

        # 할인된 보상 계산 및 정책 업데이트
        discounted_rewards = []
        G = 0
        for reward in reversed(rewards):
            G = reward + gamma * G
            discounted_rewards.insert(0, G)

        discounted_rewards = torch.tensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

        policy_loss = []
        for log_prob, G in zip(log_probs, discounted_rewards):
            policy_loss.append(-log_prob * G)
        policy_loss = torch.cat(policy_loss).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        print(f"Episode {episode + 1}: Total Reward = {sum(rewards)}")

# 4. 실행
if __name__ == "__main__":
    train()