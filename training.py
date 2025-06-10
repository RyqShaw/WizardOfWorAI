import time
import gymnasium as gym
import ale_py
import random
import torch
from replaybuffer import ReplayBuffer
from dqn import DQN
import numpy as np

# Setup environment
rendering = "rgb_array"
gym.register_envs(ale_py)
env = gym.make("ALE/WizardOfWor-v5", render_mode=rendering, obs_type="ram", obs_type='ram')
        
# Get env info
obs, info = env.reset()
next_obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
print(next_obs.shape, reward, terminated, truncated, info)
state_size = obs.shape
action_size = env.action_space.n
replay_buffer = ReplayBuffer()



dqn = DQN(obs.shape[0], len(env.action_space))
mse = torch.nn.MSELoss()
optimizer = torch.optim.Adam()

def train(batch_size=64, gamma=0.999, epsilon=1, decay=.999, max_episodes=100000):
    current_epsilon = epsilon
    episode_rewards = []
        
    for episode in range(max_episodes):
        state, info = env.reset()
        episode_over = False
        total_reward = 0

        while not episode_over:
            action = 0
            if random.random() < current_epsilon:
                # get random action (explore)
                action = env.action_space.sample()
            else:
                # get best action from dqn 
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(observation).unsqueeze(0)
                    q_values = dqn.forward(state_tensor)
                    action = torch.argmax(q_values).item()

            next_obs, reward, terminated, truncated, info = env.step(action)
            next_obs = np.array(next_obs).flatten()
            done = terminated or truncated

            # Store experience
            replay_buffer.append((observation, action, reward, next_obs, done))
            observation = next_obs
            total_reward += reward

            # Train DQN
            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)
                states = torch.FloatTensor(states)
                actions = torch.LongTensor(actions).unsqueeze(1)
                rewards = torch.FloatTensor(rewards).unsqueeze(1)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones).unsqueeze(1)

                q_values = dqn.forward(states).gather(1, actions)
                with torch.no_grad():
                    next_q_values = dqn(next_states).max(1, keepdim=True)[0]
                    target_q = rewards + gamma * next_q_values * (1 - dones)
                loss = mse(q_values, target_q)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epsilon = max(0.1, epsilon * decay)

            total_reward += reward
            step -= 0.1
            
            episode_over = truncated or terminated
            

train()