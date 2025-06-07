import time
import gymnasium as gym
import ale_py
import random

class Trainer:
    def __init__(self, episodes=10000, visual_render=False):
        self.episodes = episodes
        self.visual_render = visual_render
        
        # Setup environment
        rendering = "rgb_array"
        gym.register_envs(ale_py)
        self.env = gym.make("ALE/WizardOfWor-v5", render_mode=rendering)
        
        # Get env info
        obs, info = self.env.reset()
        self.state_size = obs.shape
        self.action_size = self.env.action_space.n
        
        print(f"State shape: {self.state_size}")
        print(f"Action size: {self.action_size}")

        def train(self, batch_size=64, gamma=0.999, epsilon=1, decay=.999, max_episodes=100000) :
        
            current_epsilon = epsilon
            episode_rewards = []
        
            for episode in range(max_episodes):
            
                state, info = self.env.reset()
                episode_over = False
                total_reward = 0

                while not episode_over:
                    action = 0
                    if random.random() < current_epsilon:
                        action = env.action_space.sample()
                    else:
                        pass
                    
                    new_obs, reward, terminated, truncated, info = env.step(action)

                    episode_over = truncated or terminated

        env.close()