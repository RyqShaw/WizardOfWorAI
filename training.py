import time
import gymnasium as gym
import ale_py
import random
import torch
from replaybuffer import ReplayBuffer
from dqn import DQN
import numpy as np
import matplotlib.pyplot as plt
import os

# Setup environment
rendering = "rgb_array"
gym.register_envs(ale_py)
env = gym.make("ALE/WizardOfWor-v5", render_mode=rendering, obs_type="ram")

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.mps.is_available():
    device = "mps"
        
# Get env info
obs, info = env.reset()
next_obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
print(next_obs.shape, reward, terminated, truncated, info)
state_size = obs.shape[0]
action_size = env.action_space.n

checkpoint_interval = 50
model_path = "nn.pth"

# Training: does 64 concurrent episodes by default, uses DQN and Replay Buffer Impl
def train(batch_size=64, gamma=0.999, epsilon=1, decay=.999, max_episodes=100):
    # Save Episodes
    current_epsilon = epsilon
    episode_rewards = []
    replay_buffer = ReplayBuffer(10000)

    # DQN and Torch Setup
    dqn = DQN(state_size, action_size, device).to(device)
    #print("Initialized DQN", dqn.parameters())
    mse = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(dqn.parameters(), lr=0.001)
    #print("Initialized Optimizer")
    
    #Load Checkpoint if one exists
    if os.path.exists("checkpoint.pth"):
        print("Loading Checkpoint")
        checkpoint = torch.load("checkpoint.pth")
        dqn.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epsilon = checkpoint['epsilon']
        episode_rewards = checkpoint['episode_rewards']
    
    # Training Loop
    print("Starting Training:")
    for episode in range(max_episodes):
        observation, info = env.reset()
        episode_over = False
        total_reward = 0

        if episode % checkpoint_interval == 0:
            print(f"Episode {episode} / {max_episodes}")
        while not episode_over:
            # Epsilon Greedy: random or optimal from DQN
            action = 0
            if random.random() < current_epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(observation).unsqueeze(0).to(device)
                    q_values = dqn.forward(state_tensor)
                    action = torch.argmax(q_values).item()

            # print(action)
            next_obs, reward, terminated, truncated, info = env.step(action)
            if action == 0:
                reward -= 0.01
            # reward = np.clip(reward, -1, 1)
            # if reward > 0:

            #     print(reward)
            next_obs = np.array(next_obs).flatten()

            # Store experience
            replay_buffer.add(observation, action, reward, next_obs)
            observation = next_obs
            total_reward += reward

            # Train DQN, Storing SARS
            if len(replay_buffer) >= batch_size:
                states, actions, rewards, next_states = replay_buffer.sample(batch_size)
                states = torch.FloatTensor(states).to(device)
                actions = torch.LongTensor(actions).unsqueeze(1).to(device)
                rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
                next_states = torch.FloatTensor(next_states).to(device)

                q_values = dqn.forward(states).gather(1, actions)
                with torch.no_grad():
                    next_q_values = dqn(next_states).max(1, keepdim=True)[0]
                    target_q = rewards + gamma * next_q_values
                loss = mse(q_values, target_q)
                # print(loss)
                # if info['frame_number'] > 3080:
                #     return

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epsilon = max(0.1, epsilon * decay)
            
            episode_over = truncated or terminated
        episode_rewards.append(total_reward)
        #print(total_reward)
        
        # Save Periodically incase of crash every 50 episodes:
        if episode % checkpoint_interval == 0 and episode != 0:
            checkpoint = {
                'episode': episode,
                'model_state_dict': dqn.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epsilon': epsilon,
                'episode_rewards': episode_rewards
            }
            torch.save(checkpoint, "checkpoint.pth")
    plt.scatter(range(max_episodes), episode_rewards)
    plt.show()
            
    return dqn
dqn = train(max_episodes=100)

torch.save(dqn.state_dict(), model_path)
