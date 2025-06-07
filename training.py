import time
import gymnasium as gym
import ale_py
import random

# Setup environment
rendering = "rgb_array"
gym.register_envs(ale_py)
env = gym.make("ALE/WizardOfWor-v5", render_mode=rendering)
        
# Get env info
obs, info = env.reset()
state_size = obs.shape
action_size = env.action_space.n

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
                action = env.action_space.sample()
            else:
                pass
                ''' Implement when Replay Buffer is in '''

            new_obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step -= 0.1
            
            episode_over = truncated or terminated
            
        epsilon = max(0.1, epsilon * decay)

train()