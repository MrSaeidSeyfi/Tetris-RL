import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from tetris2 import TetrisEnv
import matplotlib.pyplot as plt
import time  # Import time for adding delay

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim  # Already flattened
        self.action_dim = action_dim
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64
        self.model = DQN(self.state_dim, self.action_dim)
        self.target_model = DQN(self.state_dim, self.action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0)
        act_values = self.model(state)
        return torch.argmax(act_values, dim=1).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).unsqueeze(0)  # Ensure the state is a batch of size 1
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            target = reward
            if not done:
                target = (reward + self.gamma * torch.max(self.target_model(next_state)).item())
            target_f = self.model(state)
            target_f[0][action] = target
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(self.model(state), target_f)
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)

if __name__ == "__main__":
    env = TetrisEnv()
    state_dim = env.observation_space.shape[0]  # Flattened state dimension
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim)

    episodes = 1000
    rewards = []
    render_interval = 1  # Render every episode for visualization

    for e in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            agent.replay()

            # Render the environment
            env.render()
            # time.sleep(0.00001)  # Add delay to slow down the visualization

        agent.update_target_model()
        rewards.append(total_reward)
        print(f"Episode {e+1}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon}")

        if e % 10 == 0:
            agent.save("tetris-dqn.h5")

    # Plot rewards
    plt.plot(range(episodes), rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('DQN Learning Progress')
    plt.show()
